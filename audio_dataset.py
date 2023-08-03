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
