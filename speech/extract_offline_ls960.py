"""Offline Teacher Model Feature Extractor for M2D-S

Masked Modeling Duo for Speech: Specializing General-Purpose Audio Representation to Speech using Denoising Distillation
https://arxiv.org/abs/2305.14079

This script prepares offline features obtained from a teacher model. An example follows:

    (cd to the root folder of your M2D copy)
    python speech/extract_offline_ls960.py /path/to/LibriSpeech

This example will create `data/ls960_hybrid7s_hubaseL9` and `data/files_ls960_hybrid.csv`.

## Data file details

`data/ls960_hybrid7s_hubaseL9` will have converted files in .npz format. Each .npz file consists of three contents:

- arr_0: Log-mel spectrogram converted from the raw wave. The speech shorter than 7 seconds will be padded with zeros.
- arr_1: Features (hidden_states) extracted from the teacher model.
- arr_2: The length of the original hidden states excluding paddings.

Find the details for how these contents are used in SpeechHybridDataset class in speech/speech_dataset.py.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import librosa
import fire
from transformers import Wav2Vec2Processor, HubertModel
from tqdm import tqdm

sys.path.append('.')  # for running under your `m2d` folder to find wav_to_lms
from wav_to_lms import ToLogMelSpec, FFT_parameters


def prepare_ls960(src, dest='data/ls960_hybrid7s_hubaseL9', dest_csv='data/files_ls960_hybrid.csv', min_seconds=7):
    """
    Args:
        src: Source LibriSpeech 960h dataset folder.
        dest: Destination folder to store .npz files.
        dest_csv: The name of the output CSV file listing the .npz file names.
    """

    dest = Path(dest)
    src = Path(src)
    files = sorted(src.rglob('train*/**/*.flac'))
    min_samples = 16000 * min_seconds

    # Teacher model
    output_layers = [9]
    device = 'cuda'
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = HubertModel.from_pretrained("facebook/hubert-base-ls960")
    model.eval()
    model.to(device)

    # Spectrogram converter (M2D default)
    to_lms = ToLogMelSpec(FFT_parameters())

    # Extract LS960 features from the teacher
    print(f'Processing {len(files)} files..')
    csv_rel_paths = []
    for i, f in tqdm(enumerate(files)):
        wav, sr = librosa.load(f, mono=True, sr=FFT_parameters.sample_rate)
        org_wav_len = len(wav)

        # pad if short
        if min_samples is not None:
            if wav.shape[-1] < min_samples:
                wav = np.pad(wav, (0, min_samples - wav.shape[-1]))

        lms = to_lms(wav).numpy()
        wav = torch.tensor(wav).unsqueeze(0)

        preprocessed = processor(wav, return_tensors="pt", sampling_rate=16000).input_values  # Batch size 1
        preprocessed = preprocessed[0].to(device) # [1, B, raw wave length] -> [B, raw wave length]
        with torch.no_grad():
            hidden_states = model(preprocessed, output_hidden_states=True).hidden_states# list of [B, T, D]
        # stack layer outputs
        states_to_stack = [hidden_states[index] for index in output_layers] if output_layers else hidden_states
        hidden_states = torch.cat(states_to_stack, axis=-1).cpu().numpy()

        rel_path = str(f.relative_to(src)).replace('.flac', '.npz')
        csv_rel_paths.append(str(dest.relative_to('data')/rel_path))
        newname = dest/rel_path
        newname.parent.mkdir(parents=True, exist_ok=True)

        org_hidden_len = (hidden_states.shape[1] * org_wav_len) // wav.shape[-1]

        np.savez(newname, lms, hidden_states, org_hidden_len)  # arr_0: lms, arr_1: hidden_states, arr_2: original hidden states length
        if (i + 1) % 100 == 0:
            print(i, f'{i/len(files)*100:.3f}%', newname, lms.shape, hidden_states.shape, org_hidden_len)

    pd.DataFrame({'file_name': csv_rel_paths}).to_csv(dest_csv, index=None)
    print('Done.')


if __name__ == '__main__':
    fire.Fire(prepare_ls960)
