"""Dataset for Spectrogram Audio.

## Data files
All the data samples used here are expected to be `.npy` pre-converted spectrograms.
Please find instructions in `README.md`.

## Data folder structure
We expect the following data folder structure.
Note that our training pipeline uses samples from the folder `vis_samples` for visualization.
Make a folder named `vis_samples` under the root folder of the dataset, and put some samples for visualization in the `vis_samples`.

    (data root)/(any sub-folder)/(data samples).npy
      :
    (data root)/vis_samples/(data samples for visualization).npy
      :
"""

import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F


class SpectrogramDataset(torch.utils.data.Dataset):
    """Spectrogram audio dataset class.
    Args:
        folder: Root folder that stores audio samples.
        files: List of relative path names from the root folder for all samples.
        crop_frames: Number of time frames of a data which this class outputs.
        norm_stats: Normalization statistics comprising mean and standard deviation.
            If None, statistics are calculated at runtime.
            If a pathname, the precomputed statistics will be loaded.
        tfms: Transform functions for data augmentation.
        random_crop: Set True to randomly crop data of length crop_frames,
            or always crop from the beginning of a sample.
        n_norm_calc: Number of samples to calculate normalization statistics at runtime.
    """

    def __init__(self, folder, files, crop_frames, norm_stats=None,
                 tfms=None, random_crop=True, n_norm_calc=10000, repeat_short=False):
        super().__init__()
        self.folder = Path(folder)
        self.df = pd.DataFrame({'file_name': files})
        self.crop_frames = crop_frames
        self.tfms = tfms
        self.random_crop = random_crop
        self.repeat_short = repeat_short

        # Norm stats
        if norm_stats is None:
            # Calculate norm stats runtime
            lms_vectors = [self[i][0] for i in np.random.randint(0, len(files), size=n_norm_calc)]
            lms_vectors = torch.stack(lms_vectors)
            norm_stats = lms_vectors.mean(), lms_vectors.std() + torch.finfo().eps
        elif isinstance(norm_stats, (str)):
            # Lpoad from a file
            if Path(norm_stats).exists():
                norm_stats = torch.FloatTensor(np.load(norm_stats))
            else:
                # Create a norm stat file and save it. The created file will be loaded at the next runtime.
                lms_vectors = [self[i][0] for i in np.random.randint(0, len(files), size=n_norm_calc)]
                lms_vectors = torch.vstack(lms_vectors)
                new_stats = lms_vectors.mean(axis=(0, 2), keepdims=True), lms_vectors.std(axis=(0, 2), keepdims=True) + torch.finfo().eps
                np.save(norm_stats, torch.stack(new_stats).numpy())
                norm_stats = new_stats
        self.norm_stats = norm_stats

        print(f'Dataset contains {len(self.df)} files with a normalizing stats {self.norm_stats}.')

    def __len__(self):
        return len(self.df)

    def get_audio_file(self, filename):
        lms = torch.tensor(np.load(filename))
        return lms

    def get_audio(self, index):
        filename = self.folder/self.df.file_name.values[index]
        return self.get_audio_file(filename)

    def complete_audio(self, lms, dont_tfms=False, org_index=None):
        # Repeat if short
        l = lms.shape[-1]
        if self.repeat_short and l < self.crop_frames:
            while l < self.crop_frames:
                lms = torch.cat([lms, lms], dim=-1)
                l = lms.shape[-1]
            # print(f'Repeated short sample (< {self.crop_frames}) at {org_index} as {lms.shape}')

        # Trim or pad
        start = 0
        if l > self.crop_frames:
            start = int(torch.randint(l - self.crop_frames, (1,))[0]) if self.random_crop else 0
            lms = lms[..., start:start + self.crop_frames]
            # if org_index is not None and org_index % 1000 == 0:
            #     print(org_index, 'trimmed from', start)
        elif l < self.crop_frames:
            pad_param = []
            for i in range(len(lms.shape)):
                pad_param += [0, self.crop_frames - l] if i == 0 else [0, 0]
            lms = F.pad(lms, pad_param, mode='constant', value=0)
        self.last_crop_start = start
        lms = lms.to(torch.float)

        # Normalize
        if hasattr(self, 'norm_stats'):
            lms = (lms - self.norm_stats[0]) / self.norm_stats[1]

        # Apply transforms
        if self.tfms is not None:
            if not dont_tfms:
                lms = self.tfms(lms)

        return lms

    def __getitem__(self, index):
        lms = self.get_audio(index)
        return self.complete_audio(lms, org_index=index)

    def __repr__(self):
        format_string = self.__class__.__name__ + f'(crop_frames={self.crop_frames}, random_crop={self.random_crop}, '
        format_string += f'tfms={self.tfms}\n'
        return format_string


def get_files(dataset_name):
    files = pd.read_csv(str(dataset_name)).file_name.values
    files = sorted(files)
    return files


def get_files_no_sort(dataset_name):
    return pd.read_csv(str(dataset_name)).file_name.values


def build_dataset(cfg):
    """The followings configure the training dataset details.
        - data_path: Root folder of the training dataset.
        - dataset: The _name_ of the training dataset, an stem name of a `.csv` training data list.
        - norm_stats: Normalization statistics, a list of [mean, std].
        - input_size: Input size, a list of [# of freq. bins, # of time frames].
    """

    transforms = None # Future options: torch.nn.Sequential(*transforms) if transforms else None
    norm_stats = cfg.norm_stats if 'norm_stats' in cfg else None
    ds = SpectrogramDataset(folder=cfg.data_path, files=get_files(cfg.dataset), crop_frames=cfg.input_size[1],
            tfms=transforms, norm_stats=norm_stats)
    return ds


def build_viz_dataset(cfg):
    files = [str(f).replace(str(cfg.data_path) + '/', '') for f in sorted(Path(cfg.data_path).glob('vis_samples/*.npy'))]
    if len(files) == 0:
        return None, []
    norm_stats = cfg.norm_stats if 'norm_stats' in cfg else None
    ds = SpectrogramDataset(folder=cfg.data_path, files=files, crop_frames=cfg.input_size[1], tfms=None, norm_stats=norm_stats)
    return ds, files


# Mixed dataset

def log_mixup_exp(xa, xb, alpha):
    xa = xa.exp()
    xb = xb.exp()
    x = alpha * xa + (1. - alpha) * xb
    return torch.log(torch.max(x, torch.finfo(x.dtype).eps*torch.ones_like(x)))


class MixedSpecDataset(torch.utils.data.Dataset):
    def __init__(self, base_folder, files_main, files_bg_noise, crop_size, noise_ratio=0.0,
                 random_crop=True, n_norm_calc=10000) -> None:
        super().__init__()

        self.ds1 = SpectrogramDataset(folder=base_folder, files=files_main, crop_frames=crop_size[1],
                random_crop=random_crop, norm_stats=None,
                n_norm_calc=n_norm_calc//2)
        self.norm_stats = self.ds1.norm_stats  # for compatibility with SpectrogramDataset
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
        clean = self.ds1[index]
        if self.noise_ratio > 0.0:
            # load random noise sample ### , while making noise floor zero
            noise = self.ds2[index if fixed_noise else self.get_next_bgidx()]
            # mix
            mixed = log_mixup_exp(noise, clean, self.noise_ratio) if self.noise_ratio < 1.0 else noise
        else:
            mixed = clean.clone()
        # finish normalization. clean and noise were averaged to zero. the following will scale to 1.0 using ds1 std.
        clean = clean / self.norm_std
        mixed = mixed / self.norm_std
        return clean, mixed


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


def inflate_files(files, desired_size):
    if len(files) == 0:
        return files
    files = list(files)  # make sure `files`` is a list
    while len(files) < desired_size:
        files = (files + files)[:desired_size]
    return files


def build_mixed_dataset(cfg):
    """The followings configure the training dataset details.
        - data_path: Root folder of the training dataset.
        - dataset: The _name_ of the training dataset, an stem name of a `.csv` training data list.
        - norm_stats: Normalization statistics, a list of [mean, std].
        - input_size: Input size, a list of [# of freq. bins, # of time frames].
    """

    # get files and inflate the number of files (by repeating the list) if needed
    files_main = get_files(cfg.csv_main)
    files_bg = get_files(cfg.csv_bg_noise) if cfg.noise_ratio > 0. else []
    desired_min_size = 0
    if 'min_ds_size' in cfg and cfg.min_ds_size > 0:
        desired_min_size = cfg.min_ds_size
    if desired_min_size > 0:
        old_sizes = len(files_main), len(files_bg)
        files_main, files_bg = inflate_files(files_main, desired_min_size), inflate_files(files_bg, desired_min_size)
        print('The numbers of data files are increased from', old_sizes, 'to', (len(files_main), len(files_bg)))

    ds = MixedSpecDataset(
        base_folder=cfg.data_path, files_main=files_main,
        files_bg_noise=files_bg,
        crop_size=cfg.input_size,
        noise_ratio=cfg.noise_ratio,
        random_crop=True)
    if 'weighted' in cfg and cfg.weighted:
        assert desired_min_size == 0
        ds.weight = pd.read_csv(cfg.csv_main).weight.values

    val_ds = SpectrogramDataset(folder=cfg.data_path, files=get_files(cfg.csv_val), crop_frames=cfg.input_size[1], random_crop=True) \
        if cfg.csv_val else None

    return ds, val_ds


def build_mixed_viz_dataset(cfg):
    files = [str(f).replace(str(cfg.data_path) + '/', '') for f in sorted(Path(cfg.data_path).glob('vis_samples/*.npy'))]
    if len(files) == 0:
        return None, []
    norm_stats = cfg.norm_stats if 'norm_stats' in cfg else None
    ds = SpectrogramDataset(folder=cfg.data_path, files=files, crop_frames=cfg.input_size[1], tfms=None, norm_stats=norm_stats)
    return ds, files


if __name__ == '__main__':
    # Test
    ds = MixedSpecDataset(base_folder='data', files_main=get_files('data/files_gtzan.csv'),
                          files_bg_noise=get_files('data/files_audioset.csv'),
                          crop_size=[80, 608], noise_ratio=0.2, random_crop=True, n_norm_calc=10)
    for i in range(0, 10):
        clean, mixed = ds[i]
        print(clean.shape, mixed.shape)
