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


import pandas as pd
class SpeechHybridLabelDataset(SpeechHybridDataset):
    def __init__(self, folder, files, crop_size, norm_stats=None,
                 random_crop=True, n_norm_calc=20000, 
                 label_csv='data/ls960_train_hubert_base_ls960_L9_km500.csv', n_classes=500,
                 patch_len=None):
        super().__init__(folder, files, crop_size, norm_stats, random_crop, n_norm_calc, patch_len)

        df = pd.read_csv(label_csv)
        df['id_'] = [x.split('/')[-1][:-5] for x in df.file_name.values]
        df = df.sort_values('file_name')

        files_id_ = [f.split('/')[-1][:-4] for f in files]
        assert all(files_id_ == df.id_.values), 'Mismatch between LMS files and labels.'

        # convert label text into list of labels
        df['labels'] = [[int(x) for x in label.split(' ')] for label in df.labels.values]
        df['file_name'] = files
        self.df = df
        self.label_len = crop_size[1] // patch_len
        self.n_classes = n_classes

    def complete_data(self, lms, label):
        # crop & normalize
        x = super().complete_audio(lms)
        j = self.last_crop_start
        if not hasattr(self, 'norm_stats'):
            return x  # for norm_stats calculation

        label_j = (self.label_len * j) // self.crop_frames
        assert (self.crop_frames % self.label_len) == 0, f'LMS frame length has to be multiple of label length.'
        # convert label into one-hot encoding and shrink the label length
        # repeat the last label for short labels to ensure that the frame length matches the label length
        padded_frames = max(lms.shape[-1], self.crop_frames)
        n_patches = (padded_frames + self.patch_len - 1) // self.patch_len
        n_frames = n_patches * self.patch_len
        n_labels = (n_frames + 1) // 2  # frames=100Hz vs labels=50Hz
        if len(label) < n_labels:
            n_repeat = n_labels - len(label)
            label = label + ([label[-1]] * n_repeat)
            assert len(label) == n_labels
        # shrink labels to match the patch length
        onehot = np.eye(self.n_classes)[label]
        n_label_per_patch = self.patch_len // 2
        cur_len = len(label)
        new_len = cur_len // n_label_per_patch
        onehot = onehot.T.reshape(-1, new_len, n_label_per_patch).sum(axis=-1).T
        onehot = onehot / n_label_per_patch  # values in a one-hot label should sum to 1.
        # crop label
        onehot = onehot[label_j:label_j + self.label_len, :]

        if onehot.shape[0] < self.label_len:
            print(onehot.shape, lms.shape, i, j, h, w, n_patches, n_frames, n_labels, cur_len, new_len, label_j, label_j + self.label_len)
        return x, torch.tensor(onehot).to(float)

    def __getitem__(self, index):
        filename = self.folder/self.df.file_name.values[index]
        try:
            hybrid = np.load(str(filename))
        except:
            assert False, f'Failed to load: {filename}'
        lms = torch.tensor(hybrid['arr_0'])

        label = self.df.labels.values[index] if hasattr(self, 'norm_stats') else ['not needed']
        items = self.complete_data(lms, label)
        return items


class MixedSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, base_folder, files_speech, files_bg_noise, crop_size, patch_len, noise_ratio=0.0,
                 random_crop=True, n_norm_calc=10000, use_label=False) -> None:
        super().__init__()

        ds_cls = SpeechHybridLabelDataset if use_label else SpeechHybridDataset
        self.ds1 = ds_cls(folder=base_folder, files=files_speech, crop_size=crop_size,
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
        noise_ratio=cfg.noise_ratio, use_label=(cfg.model in ['m2d_s_vit_label_base', 'm2d_s_vit_label_bce_base',
         'm2d_s_vit_label2_base', 'm2d_s_vit_label2_bce_base', 'm2d_s_vit_hubert_base']))

    val_ds = SpectrogramDataset(folder=cfg.data_path, files=get_files(cfg.csv_val), crop_frames=cfg.input_size[1], random_crop=True) \
        if cfg.csv_val else None

    return ds, val_ds


def build_viz_dataset(cfg):
    files = [str(f).replace(str(cfg.data_path) + '/', '') for f in sorted(Path(cfg.data_path).glob('vis_speect_samples/*.npy'))]
    if len(files) == 0:
        return None, []
    norm_stats = cfg.norm_stats if 'norm_stats' in cfg else None
    ds = SpectrogramDataset(folder=cfg.data_path, files=files, crop_size=cfg.input_size, norm_stats=norm_stats)
    return ds, files
