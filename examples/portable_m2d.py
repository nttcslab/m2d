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


def parse_sizes_by_name(name):
    # Parse parameters. "m2d_vit_base-80x1001p16x16p16k" -> input size: 80x1001, patch size: 16x16, sr: 16k
    model_cls = name.split('-')[0]
    params = name.split('-')[1]
    params = params.split('p')[:3]
    input_str, patch_str, sr = params[0], params[1], params[2] if len(params) > 2 else '16k'
    input_size = [int(a) for a in input_str.split('x')]
    patch_size = [int(a) for a in patch_str.split('x')]
    return input_size, patch_size, sr, model_cls


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


def make_it_CLAP(model, checkpoint):
    # Add projectors if needed
    if 'audio_proj.0.weight' in checkpoint.keys():
        proj_hidden_dim = embed_dim = checkpoint['audio_proj.0.weight'].shape[1]
        model.audio_proj = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, proj_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(proj_hidden_dim, embed_dim),
        )
        if 'text_proj.weight' in checkpoint.keys():
            dim = checkpoint['text_proj.weight'].shape
            model.text_proj = torch.nn.Linear(dim[1], dim[0])
        else:
            model.text_proj = torch.nn.Identity()


def get_backbone(args, weight_file):
    args.input_size, args.patch_size, args.sr, args.model = parse_sizes_by_name(Path(weight_file).parent.name)

    # Load checkpoint.
    checkpoint = torch.load(weight_file, map_location='cpu')
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

    # Make normalization statistics for the model easy to use in the downstream task.
    args.mean, args.std = model.state_dict()['norm_stats'].to('cpu').numpy()

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
            models_mae.set_requires_grad(self.backbone.patch_embed, False)
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
            # flatten all patch embeddings
            for i in range(n_chunk):
                emb = self.backbone.forward_encoder(x[..., i*unit_frames:(i+1)*unit_frames])
                emb = emb[..., 1:, :]
                if average_per_time_frame:
                    emb = rearrange(emb, 'b (f t) d -> b t d f', f=patch_fbins, d=embed_d).mean(-1)
                embeddings.append(emb)
        else:
            # stack embeddings along time frame
            for i in range(n_chunk):
                emb = self.backbone.forward_encoder(x[..., i*unit_frames:(i+1)*unit_frames])
                emb = emb[..., 1:, :]
                emb = rearrange(emb, 'b (f t) d -> b t (f d)', f=patch_fbins, d=embed_d)
                embeddings.append(emb)
        # concatenate embedding chunks in the time axis
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
        audio_embeddings = audio_embeddings.mean(dim=-2)
        audio_embeddings = self.backbone.audio_proj(audio_embeddings)
        return audio_embeddings

    def encode_clap_text(self, batch_text, truncate=False):
        if not hasattr(self, 'text_encoder'):
            self.text_encoder = GTETextEncoder()
        text_embeddings = self.text_encoder(batch_text, truncate=truncate)
        text_embeddings = self.backbone.text_proj(text_embeddings)
        text_embeddings = text_embeddings.detach().cpu().to(torch.float)
        return text_embeddings


# For the CLAP models

class GTETextEncoder:
    def __init__(self, clip_weight="thenlper/gte-base"):
        from transformers import AutoTokenizer, AutoModel
        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "true"  # To suppress warnings.

        self.tokenizer = AutoTokenizer.from_pretrained(clip_weight)
        self.model = AutoModel.from_pretrained(clip_weight)

    def __call__(self, texts, truncate=True, max_length=512):
        def average_pool(last_hidden_states, attention_mask):
            last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
            return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

        with torch.no_grad():
            device = next(self.model.parameters()).device
            batch_dict = self.tokenizer(texts, max_length=max_length, padding=True, truncation=truncate, return_tensors='pt')
            batch_dict['input_ids'] = batch_dict['input_ids'].to(device)
            batch_dict['token_type_ids'] = batch_dict['token_type_ids'].to(device)
            batch_dict['attention_mask'] = batch_dict['attention_mask'].to(device)
            outputs = self.model.to(device)(**batch_dict)
        embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        return embeddings
