"""Wave to log-mel spectrogram (LMS) audio file converter.

This program converts the original audio files recursively found in the source folder,
then stores them in the destination folder while holding the same relative path structure.

The conversion includes the following processes:
    - Stereo to mono
    - Resampling to a sampling rate
    - Converting to a log-mel spectrogram

Example:
    python wav_to_lms.py /your/local/fsd50k/FSD50K.dev_audio /your/msm_mae/fsd50kdev_lms
"""

import numpy as np
from pathlib import Path
import librosa
from multiprocessing import Pool
import torch.multiprocessing as mp
import torch
import fire
from tqdm import tqdm
import nnAudio.features
import warnings
warnings.simplefilter('ignore')


class FFT_parameters:
    # We extract log-mel spectrograms with 80 features using a window size of 25 ms and a stride of 10 ms from a waveform sampled at 16kHz.
    sample_rate = 16000
    window_size = 400
    n_fft       = 400
    hop_size    = 160
    n_mels      = 80
    f_min       = 50
    f_max       = 8000


def _converter_worker(args):
    subpathname, from_dir, to_dir, prms, to_lms, suffix, min_length, verbose = args
    from_dir, to_dir = Path(from_dir), Path(to_dir)
    to_name = to_dir/(subpathname[:-len(suffix)]+'.npy')

    if to_name.exists():
        print('already exist', subpathname)
        return ''

    # load and convert to a log-mel spectrogram
    try:
        wav, org_sr = librosa.load(str(from_dir/subpathname), mono=True, sr=prms.sample_rate)

        # pad if short
        if min_length is not None:
            min_length = int(FFT_parameters.sample_rate * min_length)
            if wav.shape[-1] < min_length:
                print('from', wav.shape)
                wav = np.pad(wav, (0, min_length - wav.shape[-1]))
                print('to', wav.shape)

        lms = to_lms(wav)
    except Exception as e:
        print('ERROR failed to open or convert', subpathname, '-', str(e))
        return ''

    to_name.parent.mkdir(parents=True, exist_ok=True)
    np.save(to_name, lms)

    if verbose:
        print(from_dir, '->', to_name, lms.shape)

    return to_name.name


class ToLogMelSpec:
    def __init__(self, cfg):
        # Spectrogram extractor
        self.cfg = cfg
        self.to_spec = nnAudio.features.MelSpectrogram(
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

    def __call__(self, audio):
        x = self.to_spec(torch.tensor(audio))
        x = (x + torch.finfo().eps).log()
        return x


def convert_wav(from_dir, to_dir, suffix='.wav', skip=0, min_length=6.1, verbose=False) -> None:
    from_dir = str(from_dir)
    files = [str(f).replace(from_dir, '') for f in Path(from_dir).glob(f'**/*{suffix}')]
    files = [f[1:] if f[0] == '/' else f for f in files]
    files = sorted(files)
    if skip > 0:
        files = files[skip:]

    prms = FFT_parameters()
    to_lms = ToLogMelSpec(prms)

    print(f'Processing {len(files)} {suffix} files at a sampling rate of {prms.sample_rate} Hz...')
    assert len(files) > 0

    with Pool() as p:
        args = [[f, from_dir, to_dir, prms, to_lms, suffix, min_length, verbose] for f in files]
        shapes = list(tqdm(p.imap(_converter_worker, args), total=len(args)))

    print('finished.')


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    fire.Fire(convert_wav)
