"""Dataset for Speech

Masked Modeling Duo for Speech: Specializing General-Purpose Audio Representation to Speech using Denoising Distillation
https://arxiv.org/abs/2305.14079

## Data files

All the data samples used here are expected to be `.npz` preprocessed contents.
Please find the details in `README.md` and preprocessor `extract_offline_ls960.py`.

"""

import numpy as np
from pathlib import Path
import torch

from audio_dataset import SpectrogramDataset, get_files


def log_mixup_exp(xa, xb, alpha):
    xa = xa.exp()
    xb = xb.exp()
    x = alpha * xa + (1. - alpha) * xb
    return torch.log(torch.max(x, torch.finfo(x.dtype).eps*torch.ones_like(x)))


class SpeechHybridDataset(SpectrogramDataset):
    def __init__(self, folder, files, crop_size, norm_stats=None,
                 random_crop=True, n_norm_calc=20000, 
                 patch_len=None):
        assert (crop_size[1] % 2) == 0, f'Crop frames has to be multiple of 2 (frames=100Hz vs embeddings=50Hz): {crop_size}'
        self.raw_emb_len = crop_size[1] // 2  # frames=100Hz vs embeddings=50Hz
        self.emb_len = crop_size[1] // patch_len
        self.patch_len = patch_len

        super().__init__(folder=folder, files=files, crop_frames=crop_size[1], norm_stats=norm_stats,
                 random_crop=random_crop, n_norm_calc=n_norm_calc)

    def get_raw_data(self, index):
        filename = self.folder/self.df.file_name.values[index]
        try:
            hybrid = np.load(str(filename))
        except:
            assert False, f'Failed to load: {filename}'
        lms = torch.tensor(hybrid['arr_0'])
        emb = torch.tensor(hybrid['arr_1'])
        raw_emb_len = torch.tensor(hybrid['arr_2'])

        # original sample is shorter than crop duration
        if raw_emb_len < self.raw_emb_len:
            raw_emb_len = self.raw_emb_len
        # emb_len has to be the multiple of patch_len
        if (raw_emb_len % self.patch_len) > 0:
            raw_emb_len = int(raw_emb_len / self.patch_len) * self.patch_len
            assert raw_emb_len >= self.raw_emb_len, f'{raw_emb_len} {self.raw_emb_len}'

        emb = emb[:, :raw_emb_len, :]
        lms = lms[:, :, :raw_emb_len * 2]  # ensure lms length matches emb length, *2 = frames=100Hz vs embeddings=50Hz

        return lms, emb.transpose(-1, -2)  # emb: [1, T, D] -> [1, D, T] to make the same shape with lms.

    def complete_data(self, lms, emb):
        # crop & normalize
        x = super().complete_audio(lms)
        j = self.last_crop_start
        if not hasattr(self, 'norm_stats'):
            return x  # for norm_stats calculation

        # rescale the cut position j from LMS frame length to embedding length
        emb_j = (self.raw_emb_len * j) // self.crop_frames
 
        # crop embedding
        emb = emb[..., emb_j:emb_j + self.raw_emb_len]

        # shrink embeddings to match the patch length only when needed
        n_emb_per_patch = self.patch_len // 2  # 20ms per offline embedding
        if n_emb_per_patch > 1:
            _, D, T = emb.shape
            assert (T % n_emb_per_patch) == 0, f'T:{T} self.emb_len:{self.emb_len} n_emb_per_patch:{n_emb_per_patch} emb.shape:{emb.shape}'
            new_len = T // n_emb_per_patch
            emb = emb.reshape(1, D, new_len, n_emb_per_patch).mean(axis=-1)
            if new_len == 0:
                print(f'T:{T} self.emb_len:{self.emb_len} n_emb_per_patch:{n_emb_per_patch} emb.shape:{emb.shape}')

        # reshape to make it useful
        y = emb.transpose(-1, -2).squeeze(0)  # [1, D, T] to [T, D]

        return x, y

    def __getitem__(self, index):
        lms, emb = self.get_raw_data(index)
        items = self.complete_data(lms, emb)
        return items


class MixedSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, base_folder, files_speech, files_bg_noise, crop_size, patch_len, noise_ratio=0.0,
                 random_crop=True, n_norm_calc=10000) -> None:
        super().__init__()

        self.ds1 = SpeechHybridDataset(folder=base_folder, files=files_speech, crop_size=crop_size,
                random_crop=random_crop, norm_stats=None, n_norm_calc=n_norm_calc//2,
                patch_len=patch_len)
        # disable normalizion scaling in the ds1
        self.norm_std = self.ds1.norm_stats[1]
        self.ds1.norm_stats = (self.ds1.norm_stats[0], 1.0)

        if noise_ratio > 0.0:
            self.ds2 = SpectrogramDataset(folder=base_folder, files=files_bg_noise, crop_frames=crop_size[1],
                    random_crop=random_crop, norm_stats=None, n_norm_calc=n_norm_calc//2, repeat_short=True)
            self.ds2.norm_stats = (self.ds2.norm_stats[0], 1.0) # disable normalizion scaling in the ds2

        self.noise_ratio = noise_ratio
        self.bg_index = []

    def __len__(self):
        return len(self.ds1)

    def __getitem__(self, index, fixed_noise=False):
        # load index sample
        sig, label = self.ds1[index]
        if self.noise_ratio > 0.0:
            # load random noise sample ### , while making noise floor zero
            noise = self.ds2[index if fixed_noise else self.get_next_bgidx()][0]
            # mix
            sig = log_mixup_exp(noise, sig, self.noise_ratio) if self.noise_ratio < 1.0 else noise
        # finish normalization. sig and noise were averaged to zero. the following will scale to 1.0 using ds1 std.
        sig = sig / self.norm_std
        return sig, label


    def get_next_bgidx(self):
        if len(self.bg_index) == 0:
            self.bg_index = torch.randperm(len(self.ds2)).tolist()
            # print(f'Refreshed the bg index list with {len(self.bg_index)} items: {self.bg_index[:5]}...')
        return self.bg_index.pop(0)

    def __repr__(self):
        format_string = self.__class__.__name__ + f'(crop_frames={self.ds1.crop_frames}, '
        format_string += f'folder_sp={self.ds1.df.file_name.values[0].split("/")[0]}, '
        if self.noise_ratio > 0.: format_string += f'folder_bg={self.ds2.df.file_name.values[0].split("/")[0]}, '
        return format_string


def build_mixed_speech_dataset(cfg):
    ds = MixedSpeechDataset(
        base_folder=cfg.data_path, files_speech=get_files(cfg.csv_main),
        files_bg_noise=get_files(cfg.csv_bg_noise) if cfg.noise_ratio > 0. else [],
        crop_size=cfg.input_size, patch_len=cfg.patch_size[1],
        noise_ratio=cfg.noise_ratio)
    return ds


def build_viz_dataset(cfg):
    files = [str(f).replace(str(cfg.data_path) + '/', '') for f in sorted(Path(cfg.data_path).glob('vis_speect_samples/*.npy'))]
    if len(files) == 0:
        return None, []
    norm_stats = cfg.norm_stats if 'norm_stats' in cfg else None
    ds = SpectrogramDataset(folder=cfg.data_path, files=files, crop_size=cfg.input_size, norm_stats=norm_stats)
    return ds, files
