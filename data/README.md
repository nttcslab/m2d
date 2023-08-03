## Pre-training data

The pre-trainer (e.g., `train_audio.py` for audio) loads data from the `data` folder by default (`--data_path`), using a list of samples in a CSV  file `data/files_audioset.csv` by default (`--dataset`).

The CSV file should have a `file_name` column containing the relative pathname of the files containing a log-mel spectrogram (LMS) audio. Example:

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

The folders/files should look like the following:

    (Example of the folder structure)
    data/
        audioset_lms/
            balanced_train_segments/
                --aE2O5G5WE_0.000.npy
                --cB2ZVjpnA_30.000.npy
                  :

If you also have pre-processed FSD50K data, the folder will be as follows:

    (Example of the folder structure)
    data/
        audioset_lms/
          :
        fsd50k_lms/
            FSD50K.dev_audio/
                2931.npy
                408195.npy
                    :

### Example preprocessing steps (AudioSet)

If you have downloaded the AudioSet samples and converted them into .wav files in `/your/local/audioset` folder, the following example steps will preprocess and create a new folder, `data/audioset_lms`.

1. Convert your pre-training data to LMS using [`wav_to_lms.py`](../wav_to_lms.py). Example: `python wav_to_lms.py /your/local/audioset data/audioset_lms`
2. Then, make a list of files under your `data` folder. Example follows:

    ```sh
    echo file_name > data/files_audioset.csv
    (cd data && find audioset_lms -name "*.npy") >> data/files_audioset.csv
    ```

