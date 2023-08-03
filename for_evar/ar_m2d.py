"""Masked Modeling Duo (M2D) Wrapper for EVAR.

Masked Modeling Duo: Learning Representations by Encouraging Both Networks to Model the Input
https://ieeexplore.ieee.org/document/10097236/

Masked Modeling Duo for Speech: Specializing General-Purpose Audio Representation to Speech using Denoising Distillation
https://arxiv.org/abs/2305.14079
"""

import sys
sys.path.append('..')
from evar.ar_base import BaseAudioRepr, calculate_norm_stats, normalize_spectrogram
import torch
from m2d.runtime_audio import RuntimeM2D


class AR_M2D_BatchNormStats(BaseAudioRepr):

    def __init__(self, cfg):
        super().__init__(cfg=cfg)

        self.backbone = RuntimeM2D(cfg=cfg, weight_file=cfg.weight_file)
        self.backbone.eval()

    def encode_frames(self, batch_audio):
        with torch.no_grad():
            x = self.backbone.get_timestamp_embeddings(batch_audio)
        return x.transpose(1, 2) # [B, T, D] -> [B, D, T]

    def forward(self, batch_audio):
        with torch.no_grad():
            x = self.backbone.get_scene_embeddings(batch_audio)
        return x


class AR_M2D(BaseAudioRepr):

    def __init__(self, cfg, make_runtime=True):
        super().__init__(cfg=cfg)

        if make_runtime:
            self.runtime = RuntimeM2D(cfg=cfg, weight_file=cfg.weight_file)
            self.runtime.eval()

    def precompute(self, device, data_loader):
        self.norm_stats = calculate_norm_stats(device, data_loader, self.runtime.to_feature)

    def encode_frames(self, batch_audio):
        x = self.runtime.to_feature(batch_audio)
        x = normalize_spectrogram(self.norm_stats, x)
        x = self.augment_if_training(x)
        hidden_states = self.runtime.encode_lms(x, return_layers=True)
        # stack layer outputs
        states_to_stack = [hidden_states[index] for index in self.cfg.output_layers] if self.cfg.output_layers else [h for h in hidden_states]
        features = torch.cat(states_to_stack, axis=-1)
        return features.transpose(1, 2) # [B, T, D] -> [B, D, T]

    def forward(self, batch_audio):
        if hasattr(self, 'lms_mode'):
            x = self.encode_frames_lms(batch_audio)
        else:
            x = self.encode_frames(batch_audio)
        return x.mean(dim=-1) # [B, D, T] -> [B, D]

    def precompute_lms(self, device, data_loader):
        self.norm_stats = calculate_norm_stats(device, data_loader, lambda x: x)
        self.lms_mode = True

    def encode_frames_lms(self, batch_lms):
        x = normalize_spectrogram(self.norm_stats, batch_lms)
        x = self.augment_if_training(x)
        hidden_states = self.runtime.encode_lms(x, return_layers=True)
        # stack layer outputs
        states_to_stack = [hidden_states[index] for index in self.cfg.output_layers] if self.cfg.output_layers else [h for h in hidden_states]
        features = torch.cat(states_to_stack, axis=-1)
        return features.transpose(1, 2) # [B, T, D] -> [B, D, T]


class AR_M2DwDuration(AR_M2D):

    def __init__(self, cfg):
        super().__init__(cfg=cfg, make_runtime=False)

        self.runtime = RuntimeM2DwithDuration(cfg=cfg, weight_file=cfg.weight_file, )
        self.runtime.eval()
