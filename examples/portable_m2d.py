"""Masked Modeling Duo (M2D) Portable Runtime.

All you need is:
    pip install timm, einops, nnAudio
"""

import logging
import numpy as np
from pathlib import Path
from functools import partial
import re

import torch
import timm
from timm.models.layers import trunc_normal_
from einops import rearrange
import nnAudio.features


class Config:
    weight_file = ''
    feature_d = 768 * 5
    norm_type = all
    pooling_type = 'mean'
    model = ''
    input_size = [80, 208]
    patch_size = [16, 16]
    sr = '16k'
    flat_features = False


def expand_size(sz):
    if isinstance(sz, int):
        return [sz, sz]
    return sz


class PatchEmbed(torch.nn.Module):
    """ 2D Image to Patch Embedding -- borrowed from https://pypi.org/project/timm/0.4.12/"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = expand_size(img_size)
        patch_size = expand_size(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = torch.nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else torch.nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class LocalViT(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer for M2D Audio"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Workaround for PatchEmbed to avoid unintended assertion failure. ex) AssertionError: Input image width (102) doesn't match model (608).
        self.patch_embed = PatchEmbed(self.patch_embed.img_size, self.patch_embed.patch_size,
                                      self.patch_embed.proj.in_channels, self.patch_embed.proj.out_channels)
        self.norm_stats = torch.nn.Parameter(torch.tensor([-7.1, 4.2]), requires_grad=False)
        # We do not use the default head
        del self.head

    def patch_size(self):
        return np.array(self.patch_embed.patch_size)

    def grid_size(self):
        # Workaround for compatibility issue (timm 0.4.5 fails with: return self.patch_embed.grid_size)
        img_size = np.array(self.patch_embed.img_size)
        patch_size = self.patch_size()
        grid_size = img_size // patch_size
        return grid_size

    def forward_encoder(self, x):
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        pos_embed = self.pos_embed[:, 1:, :]
        if x.shape[1] < pos_embed.shape[1]:  # shorten pos_embed for a short input
            dims = pos_embed.shape[-1]
            fbins = self.grid_size()[0]
            frames = x.shape[1] // fbins
            pos_embed = pos_embed.reshape(1, fbins, -1, dims)[:, :, :frames, :].reshape(1, fbins*frames, dims)
        x = x + pos_embed

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x


def get_MLP_projector(embed_dim, proj_hidden_dim, out_embed_dim):
    projector = torch.nn.Sequential(
        torch.nn.Linear(embed_dim, proj_hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(proj_hidden_dim, out_embed_dim),
    )
    return projector


class AudioToSemantic(torch.nn.Module):
    def __init__(self, embed_dim=768, sem_depth=1, sem_heads=1, sem_mlp_ratio=1):
        # grid_size, pos_embed
        super().__init__()
        self.sem_token = torch.nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.sem_blocks = torch.nn.ModuleList([
            timm.models.vision_transformer.Block(embed_dim, sem_heads, sem_mlp_ratio, qkv_bias=True, norm_layer=torch.nn.LayerNorm)
            for i in range(sem_depth)])
        self.norm = torch.nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)
        self.dont_average = True

    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, torch.nn.Linear) and m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # Append semantic token
        sem_token = self.sem_token  # + self.pos_embed[:, :1, :]
        sem_tokens = sem_token.expand(x.shape[0], -1, -1)
        x = torch.cat((sem_tokens, x), dim=1)

        # Apply Transformer blocks
        for blk in self.sem_blocks:
            x = blk(x)
        x = x[:, 0, :]  # Use semantic token only
        x = self.norm(x)

        return x


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


def parse_sizes_by_name(name):
    # Parse parameters. "m2d_vit_base-80x1001p16x16p16kpXXXpYYY" -> input size: 80x1001, patch size: 16x16, sr: 16k, extra parameters: XXX and YYY
    model_cls = name.split('-')[0]
    params = name.split('-')[1]
    params = params.split('p')
    params, extra = params[:3], params[3:]
    input_str, patch_str, sr = params[0], params[1], params[2] if len(params) > 2 else '16k'
    input_size = [int(a) for a in input_str.split('x')]
    patch_size = [int(a) for a in patch_str.split('x')]
    return input_size, patch_size, sr, model_cls, extra


def drop_non_model_weights(model, checkpoint, filename):
    model_keys = [n for n, p in model.named_parameters()]
    new_ckpt, dropped = {}, []
    for k in checkpoint:
        if k not in model_keys:
            dropped.append(k)
            continue
        new_ckpt[k] = checkpoint[k]
    n_org = len(checkpoint.keys())
    n_cur = len(new_ckpt.keys())
    print(f' using {n_cur} parameters, while dropped {n_org - n_cur} out of {n_org} parameters from {Path(filename).parent/Path(filename).name}'
          if n_org > n_cur else f' using {n_cur} parameters from {Path(filename).parent/Path(filename).name}')
    print(' (included audio_proj params:', [k for k in new_ckpt.keys() if 'audio_proj' in k][:5])
    print(' (included text_proj params:', [k for k in new_ckpt.keys() if 'text_proj' in k][:5])
    print(' (dropped:', dropped[:5], ')' if len(dropped) < 5 else '...)')
    return new_ckpt


def load_evar_head_parameters(checkpoint, head_norm, head):
    # Load the weights of the task head trained in the EVAR fine-tuning.
    if 'module.head.norm.running_mean' in checkpoint:
        head_norm.load_state_dict({to_k: checkpoint[k] for to_k, k in {
            'running_mean':'module.head.norm.running_mean', 'running_var':'module.head.norm.running_var'}.items()})
        head.load_state_dict({to_k: checkpoint[k] for to_k, k in {
            'weight':'module.head.mlp.mlp.0.weight', 'bias':'module.head.mlp.mlp.0.bias'}.items()})
    else:
        print(' Not an EVAR checkpoint for loading head weights.')


def reformat_ckpt_keys(checkpoint):
    # In case: checkpoint['model']
    checkpoint = checkpoint['model'] if 'model' in checkpoint else checkpoint
    # The checkpoints saved in a EVAR fine-tuning has a prefix of "module.ar.runtime.backbone", the following removes it.
    new_ckpt = {}
    for k in checkpoint:
        new_k = k.replace('module.ar.runtime.backbone.', '')  # replace
        new_ckpt[new_k] = checkpoint[k]
    return new_ckpt


def extract_weight(checkpoint, root_name):
    # If no key matches the root_name, return the checkpoint unchanged.
    if not any(k.startswith(root_name) for k in checkpoint.keys()):
        return checkpoint
    # Keep only the items starts with the root_name
    new_ckpt = {k[len(root_name):]: v for k, v in checkpoint.items() if k.startswith(root_name)}
    return new_ckpt


def add_semantic_audio_proj(sem_mode, embed_dim):
    sem_params = {
        1: {'sem_depth': 1, 'sem_heads': 1, 'sem_mlp_ratio': 1},
        2: {'sem_depth': 2, 'sem_heads': 1, 'sem_mlp_ratio': 1},
        3: {'sem_depth': 3, 'sem_heads': 1, 'sem_mlp_ratio': 2},
        4: {'sem_depth': 4, 'sem_heads': 1, 'sem_mlp_ratio': 2},
    }[sem_mode]
    audio_proj = AudioToSemantic(embed_dim=embed_dim, **sem_params)
    return audio_proj


def make_it_CLAP(model, checkpoint):
    # Return if already a CLAP model
    if hasattr(model, 'audio_proj') or checkpoint is None: return
    # Add projectors if needed
    if 'audio_proj.0.weight' in checkpoint.keys():
        proj_hidden_dim, embed_dim = checkpoint['audio_proj.0.weight'].shape
        model.audio_proj = get_MLP_projector(embed_dim, embed_dim, embed_dim)
    if 'audio_proj.sem_token' in checkpoint.keys():
        embed_dim = checkpoint['audio_proj.sem_token'].shape[-1]
        sem_blocks_nums = [int(k.split('.')[2]) for k in checkpoint.keys() if k.startswith('audio_proj.sem_blocks.')]
        sem_mode = max(sem_blocks_nums) + 1
        model.audio_proj = add_semantic_audio_proj(sem_mode, embed_dim)
    if 'text_proj.weight' in checkpoint.keys():
        dim = checkpoint['text_proj.weight'].shape
        model.text_proj = torch.nn.Linear(dim[1], dim[0])
    if 'text_proj.2.weight' in checkpoint.keys():
        dim = checkpoint['text_proj.2.weight'].shape
        model.text_proj = get_MLP_projector(dim[1], dim[1], dim[0])
    ## For M2D-CLAP (2025) ablations
    if hasattr(model, 'text_proj') and not hasattr(model, 'audio_proj'):
        model.audio_proj = torch.nn.Identity()


def parse_clap_type(name):
    # Parse parameters. "m2d_clap_base-80x1001p16x16p16kpA" -> input size: 80x1001, patch size: 16x16, sr: 16k, extra: A
    params = str(name).split('-')[1]
    params = params.split('p')
    params, extra = params[:3], params[3:]
    if len(extra) == 0:
        return 'A'
    assert extra[0] in 'ABN'
    text_encoder_name = {'A': 'GTE base', 'B': 'BERT base', 'N': 'NV-Embed-v2'}
    logging.info(f' using text encoder: {text_encoder_name[extra[0]]}')
    return extra[0]


def clap_off_emb_dim(param_extra):
    if len(param_extra) == 0:
        return 768
    return {'A': 768, 'B': 768, 'N': 4096}[param_extra[0]]


def parse_clap_text_encoder_weight(param_extra, cfg, ckpt_cfg=None):
    # param_extra[0]=text encoder type, param_extra[1]=text encoder included (TI)
    if len(param_extra) <= 1:
        return None
    if len(param_extra) > 1 and param_extra[1] == 'TI':  # text encoder included
        return cfg.weight_file
    assert False, f'unknown extra parameters: {param_extra}'


def get_backbone(args, weight_file):
    args.input_size, args.patch_size, args.sr, args.model, extra = parse_sizes_by_name(Path(weight_file).parent.name)

    # Load checkpoint.
    checkpoint = torch.load(weight_file, weights_only=False, map_location='cpu')
    ckpt_cfg = checkpoint['args'] if 'args' in checkpoint else None
    checkpoint = checkpoint['model'] if 'model' in checkpoint else checkpoint
    checkpoint = extract_weight(checkpoint, 'backbone.')  # Convert from RuntimeM2D weights
    checkpoint = reformat_ckpt_keys(checkpoint)
    # Set normalization statistics for backward compatibility. The [-7.1, 4.2] is for 2022 models.
    if 'norm_stats' not in checkpoint:
        checkpoint['norm_stats'] = torch.tensor([-7.1, 4.2])
        print(' using default norm_stats:', checkpoint['norm_stats'])

    # Create a ViT.
    model = LocalViT(
        in_chans=1, img_size=args.input_size, patch_size=args.patch_size, embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4, norm_layer=partial(torch.nn.LayerNorm, eps=1e-6))

    # Modify the model if it should be a M2D-CLAP.
    make_it_CLAP(model, checkpoint)

    # Load weights.
    dropped = drop_non_model_weights(model, checkpoint, weight_file)
    msg = model.load_state_dict(dropped)
    print(msg); logging.info(msg)

    # Get text encoder weights for M2D-CLAP models.
    args.text_encoder_weight = parse_clap_text_encoder_weight(extra, args, ckpt_cfg)

    # Make normalization statistics for the model easy to use in the downstream task.
    args.mean, args.std = model.state_dict()['norm_stats'].to('cpu').numpy()
    print(f' using norm_stats: {args.mean}, {args.std}')

    model.eval()
    return model, checkpoint


def get_to_melspec(cfg):
    if cfg.sr == '16k':
        cfg.sample_rate, cfg.n_fft, cfg.window_size, cfg.hop_size = 16000, 400, 400, 160
        cfg.n_mels, cfg.f_min, cfg.f_max = 80, 50, 8000
    elif cfg.sr == '32k':
        cfg.sample_rate, cfg.n_fft, cfg.window_size, cfg.hop_size = 32000, 800, 800, 320
        cfg.n_mels, cfg.f_min, cfg.f_max = 80, 50, 16000
    else:
        assert False, f'Unknown input size: {cfg.input_size}'

    to_spec = nnAudio.features.MelSpectrogram(
        sr=cfg.sample_rate,
        n_fft=cfg.n_fft,
        win_length=cfg.window_size,
        hop_length=cfg.hop_size,
        n_mels=cfg.n_mels,
        fmin=cfg.f_min,
        fmax=cfg.f_max,
        center=True,
        power=2,
        verbose=False,
    )
    logging.info(f'Runtime MelSpectrogram({cfg.sample_rate}, {cfg.n_fft}, {cfg.window_size}, {cfg.hop_size}, '
                 + f'{cfg.n_mels}, {cfg.f_min}, {cfg.f_max}):')
    logging.info(to_spec)
    return to_spec


def get_timestamps(cfg, batch_audio, x):  # Returns timestamps in milliseconds.
    audio_len = len(batch_audio[0])
    sec = audio_len / cfg.sample_rate
    x_len = len(x[0])
    step = sec / x_len * 1000 # sec -> ms
    ts = torch.tensor([step * i for i in range(x_len)]).unsqueeze(0)
    ts = ts.repeat(len(batch_audio), 1)
    return ts


class PortableM2D(torch.nn.Module):
    def __init__(self, weight_file, num_classes=None, freeze_embed=False, flat_features=None):
        super().__init__()
        self.cfg = Config()
        self.cfg.weight_file = weight_file
        self.cfg.freeze_embed = freeze_embed
        self.cfg.flat_features = self.cfg.flat_features if flat_features is None else flat_features

        # Create backbone model.
        self.backbone, checkpoint = get_backbone(self.cfg, self.cfg.weight_file)
        # Finalize feature dimension.
        d = self.backbone.pos_embed.shape[-1]
        if num_classes is not None and 'module.head.mlp.mlp.0.weight' in checkpoint and checkpoint['module.head.mlp.mlp.0.weight'].shape[-1] == d:
            self.cfg.flat_features = True
        n_stack_feature = 1 if self.cfg.flat_features else (self.cfg.input_size[0] // self.cfg.patch_size[0])
        self.cfg.feature_d = d * n_stack_feature  # 768 if flat_features else 768*5=3840
        # Create head.
        if num_classes is not None:
            self.head_norm = torch.nn.BatchNorm1d(self.cfg.feature_d, affine=False)
            self.head = torch.nn.Linear(self.cfg.feature_d, num_classes)
            trunc_normal_(self.head.weight, std=2e-5)
            load_evar_head_parameters(checkpoint, self.head_norm, self.head)
        # Option: freeze patch embedding ([2211.09359] How to Fine-Tune Vision Models with SGD)
        if self.cfg.freeze_embed:
            set_requires_grad(self.backbone.patch_embed, False)
            logging.info(' ** Freeze patch_embed **')
            logging.info(self.backbone.patch_embed)

        logging.info(f'Model input size: {self.cfg.input_size}')
        logging.info(f'Using weights: {self.cfg.weight_file}')
        logging.info(f'Feature dimension: {self.cfg.feature_d}')
        logging.info(f'Norm stats: {self.cfg.mean}, {self.cfg.std}')

        self.to_spec = get_to_melspec(self.cfg)
        self.eval()

    def to_log_mel_spec(self, batch_audio):
        x = self.to_spec(batch_audio)
        x = (x + torch.finfo().eps).log()
        x = x.unsqueeze(1)
        return x

    def normalize_batch(self, x):
        x = (x - self.cfg.mean) / self.cfg.std
        return x

    def to_normalized_feature(self, batch_audio):
        x = self.to_log_mel_spec(batch_audio)
        x = self.normalize_batch(x)
        return x

    def encode_lms(self, x, average_per_time_frame=False):
        patch_fbins = self.backbone.grid_size()[0]
        unit_frames = self.cfg.input_size[1]
        patch_frames = self.backbone.patch_size()[1]
        embed_d = self.backbone.patch_embed.proj.out_channels
        n_chunk = (x.shape[-1] + unit_frames - 1) // unit_frames
        pad_frames = (patch_frames - (x.shape[-1] % unit_frames % patch_frames)) % patch_frames
        if pad_frames > 0:
            x = torch.nn.functional.pad(x, (0, pad_frames))

        embeddings = []
        if self.cfg.flat_features:
            # Fatten all patch embeddings
            for i in range(n_chunk):
                emb = self.backbone.forward_encoder(x[..., i*unit_frames:(i+1)*unit_frames])
                emb = emb[..., 1:, :]
                if average_per_time_frame:
                    emb = rearrange(emb, 'b (f t) d -> b t d f', f=patch_fbins, d=embed_d).mean(-1)
                embeddings.append(emb)
        else:
            # Stack embeddings along time frame
            for i in range(n_chunk):
                emb = self.backbone.forward_encoder(x[..., i*unit_frames:(i+1)*unit_frames])
                emb = emb[..., 1:, :]
                emb = rearrange(emb, 'b (f t) d -> b t (f d)', f=patch_fbins, d=embed_d)
                embeddings.append(emb)
        # Concatenate embedding chunks in the time axis
        x = torch.cat(embeddings, axis=-2)
        return x

    def encode(self, batch_audio, average_per_time_frame=False):
        x = self.to_normalized_feature(batch_audio)
        return self.encode_lms(x, average_per_time_frame=average_per_time_frame)

    def forward(self, batch_audio, average_per_time_frame=False):
        x = self.encode(batch_audio, average_per_time_frame=average_per_time_frame)
        if hasattr(self, 'head'):
            x = x.mean(1)  # B, D
            x = self.head_norm(x.unsqueeze(-1)).squeeze(-1)
            x = self.head(x)
        return x

    def get_scene_embeddings(self, batch_audio):
        x = self.encode(batch_audio)
        x = torch.mean(x, dim=1)
        return x

    def get_timestamp_embeddings(self, batch_audio):
        x = self.encode(batch_audio, average_per_time_frame=True)
        ts = get_timestamps(self.cfg, batch_audio, x)
        return x, ts

    def forward_frames(self, batch_audio):
        x, ts = self.get_timestamp_embeddings(batch_audio)
        if hasattr(self, 'head'):
            x = self.head_norm(x.transpose(-1, -2)).transpose(-2, -1)
            x = self.head(x)
        return x, ts

    def encode_clap_audio(self, batch_audio):
        audio_embeddings = self.forward(batch_audio)
        if not hasattr(self.backbone.audio_proj, 'dont_average'):
            audio_embeddings = audio_embeddings.mean(dim=-2)
        audio_embeddings = self.backbone.audio_proj(audio_embeddings)
        return audio_embeddings

    def encode_clap_text(self, batch_text, truncate=False):
        if not hasattr(self, 'text_encoder'):
            self.get_clap_text_encoder()
        text_embeddings = self.text_encoder(batch_text, truncate=truncate)
        if hasattr(self.backbone, 'text_proj'):
            text_embeddings = self.backbone.text_proj(text_embeddings)
        text_embeddings = text_embeddings.detach().cpu().to(torch.float)
        return text_embeddings

    def get_clap_text_encoder(self):
        text_encoder_weight = self.cfg.text_encoder_weight if hasattr(self.cfg, 'text_encoder_weight') else None
        self.text_encoder = get_text_encoder(self.cfg.weight_file, text_encoder_weight=text_encoder_weight)
        self.text_encoder = self.text_encoder.to(next(self.backbone.parameters()).device)


# For the CLAP models

class GTETextEncoder(torch.nn.Module):
    def __init__(self, clip_weight="thenlper/gte-base"):
        super().__init__()
        from transformers import AutoTokenizer, AutoModel
        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "true"  # To suppress warnings.

        self.tokenizer = AutoTokenizer.from_pretrained(clip_weight)
        self.model = AutoModel.from_pretrained(clip_weight)

    def __call__(self, texts, truncate=True, max_length=512):
        def average_pool(last_hidden_states, attention_mask):
            last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
            return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

        device = next(self.model.parameters()).device
        batch_dict = self.tokenizer(texts, max_length=max_length, padding=True, truncation=truncate, return_tensors='pt')
        batch_dict['input_ids'] = batch_dict['input_ids'].to(device)
        batch_dict['token_type_ids'] = batch_dict['token_type_ids'].to(device)
        batch_dict['attention_mask'] = batch_dict['attention_mask'].to(device)
        outputs = self.model.to(device)(**batch_dict)
        embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        return embeddings


class NVEmbedV2Encoder(torch.nn.Module):
    def __init__(self, clip_weight="nvidia/NV-Embed-v2"):
        # https://huggingface.co/spaces/mteb/leaderboard https://huggingface.co/nvidia/NV-Embed-v2
        # https://arxiv.org/pdf/2405.17428
        super().__init__()
        from sentence_transformers import SentenceTransformer
        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "true"  # To suppress warnings.

        self.model = SentenceTransformer(clip_weight, trust_remote_code=True)
        self.model.max_seq_length = 32768
        self.model.tokenizer.padding_side="right"

    def __call__(self, texts, **kwargs):
        def add_eos(input_examples):
            input_examples = [input_example + self.model.tokenizer.eos_token for input_example in input_examples]
            return input_examples
        texts = add_eos(texts)
        embeddings = self.model.encode(texts, batch_size=len(texts), show_progress_bar=False, convert_to_tensor=True)
        # normalize_embeddings=True
        return embeddings


class BertXEncoder(torch.nn.Module):
    def __init__(self, clip_weight="google-bert/bert-base-uncased"):
        super().__init__()
        from transformers import AutoTokenizer, AutoModel
        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "true"  # To suppress warnings.

        self.tokenizer = AutoTokenizer.from_pretrained(clip_weight)
        self.text_encoder = AutoModel.from_pretrained(clip_weight)

    def forward(self, batch_text, truncate=True, max_length=512):
        device = next(self.text_encoder.parameters()).device
        text_input = self.tokenizer(batch_text,
            padding='longest',
            truncation=truncate,
            max_length=max_length,
            return_tensors="pt").to(device)
        text_feats = self.text_encoder(input_ids=text_input.input_ids,
            attention_mask=text_input.attention_mask)[0]
        text_feats = text_feats[:, 0, :]
        return text_feats


def get_text_encoder(weight, text_encoder_weight=None):
    try:
        clap_type = parse_clap_type(Path(weight).parent.name)
    except:
        clap_type = parse_clap_type(Path(weight).stem)

    if clap_type == 'A':
        text_model = GTETextEncoder()
    if clap_type == 'B':
        text_model = BertXEncoder()
    if clap_type == 'N':
        text_model = NVEmbedV2Encoder()

    if text_encoder_weight is not None:
        weights = torch.load(text_encoder_weight, weights_only=False, map_location='cpu')
        weights = weights['model']
        weights = extract_weight(weights, 'text_encoder.')
        print(f' using model.text_encoder from {text_encoder_weight}')
        text_model.load_state_dict(weights)
    return text_model
