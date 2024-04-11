"""Data cutter.

Run under the app/icbhi_sprs folder.
"""

import sys
import os
import fire
import torch
import torchaudio
import librosa
from pathlib import Path
import pandas as pd
import numpy as np
sys.path.append('../..')

from dataset import SPRS
from m2d.runtime_audio import RuntimeM2D, Config

args_device = 'cuda'
args_metalabel ='sa'
args_samplerate = 16000
args_duration = 8
args_pad = 'circular'


def convert(to_dir='../../data/sprsound_lms', data_dir='./data/SPRS', metadata_csv='metadata.csv'):
    rt = RuntimeM2D(weight_file='m2d_vit_base-80x100p16x4-random')
    train_ds = SPRS(data_path=data_dir, metadatafile=metadata_csv, duration=args_duration, split='train', device="cpu", samplerate=args_samplerate, pad_type=args_pad, meta_label=args_metalabel)
    val_ds = SPRS(data_path=data_dir, metadatafile=metadata_csv, duration=args_duration, split='inter_test', device="cpu", samplerate=args_samplerate, pad_type=args_pad, meta_label=args_metalabel)
    to_dir = Path(to_dir)

    for split, ds in [('val', val_ds), ('train', train_ds)]:
        print(split)
        to_split = to_dir/split
        to_split.mkdir(parents=True, exist_ok=True)
        for i in range(len(ds)):
            sample, *_ = ds[i]
            with torch.no_grad():
                lms = rt.to_feature(sample).cpu().numpy()[0] # 1,1,80,801 -> 1,80,801
            np.save(to_split/f'{i:04d}.npy', lms)
            print('.', end=' ')
        print(i)


fire.Fire(convert)

