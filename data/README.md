## Pre-training data

The pre-trainer (e.g., `train_audio.py` for audio) loads data from the `data` folder by default (`--data_path`), using a list of samples in a CSV  file `data/files_audioset.csv` by default (`--dataset`).

The CSV file should have a `file_name` column containing the relative pathname of the log-mel spectrogram (LMS) samples in a numpy format. Example:

```
file_name
audioset_lms/balanced_train_segments/--aE2O5G5WE_0.000.npy
audioset_lms/balanced_train_segments/--cB2ZVjpnA_30.000.npy
audioset_lms/balanced_train_segments/--aaILOrkII_200.000.npy
audioset_lms/balanced_train_segments/--ZhevVpy1s_50.000.npy
audioset_lms/balanced_train_segments/--aO5cdqSAg_30.000.npy
audioset_lms/balanced_train_segments/--PJHxphWEs_30.000.npy
audioset_lms/balanced_train_segments/--ekDLDTUXA_30.000.npy
```

### Preparation steps

1. Convert your pre-training data to LMS using [`wav_to_lms.py`](../wav_to_lms.py).
2. Place the LMS folder under the `data` folder.
