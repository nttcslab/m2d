![key_visual](figure-github.jpg)

# Masked Modeling Duo for Speech (M2D-S)

This repository provides a demo implementation of "[Masked Modeling Duo for Speech: Specializing General-Purpose Audio Representation to Speech using Denoising Distillation](https://arxiv.org/abs/2305.14079)."

- [x] Code for pre-training and pre-processing LS-960 features.
- [x] Pre-trained weights.
- [x] [SUPERB](https://arxiv.org/abs/2105.01051) evaluation code and instructions.

## 1. Getting Started

For installation, follow the instruction in the ["1. Getting Started" in the main README.md](../README.md#1-getting-started).

For evaluating on SUPREB, refer to [superb/upstream/m2d/README.md](../superb/upstream/m2d/README.md).

## 2. Pre-trained weights

Find pre-trained weight files in [releases](https://github.com/nttcslab/m2d/releases).

- M2D-S T=4.0s: m2d_s_vit_base-80x400p80x2-230201
- M2D-S T=5.12s: m2d_s_vit_base-80x512p80x2-230301
- M2D-S T=6.08s: m2d_s_vit_base-80x608p80x2-230220

| Model    | Pre-trained dataset  | PR    | KS    | IC    | SID   | ER    | ENV       | MUS      |
|----------|----------------------|-------|-------|-------|-------|-------|-----------|----------|
| M2D-S T=4.0s  | LS-960+AS        | 5.72  | 96.47 | 97.80 | 81.97 | 66.36 | _53.22_  | _41.71_  |
| M2D-S T=5.12s | LS-960+AS        | 5.64  | 96.87 | 97.65 | 80.69 | 65.35 | _57.34_  | _43.23_  |
| M2D-S T=6.08s | LS-960+AS        | 5.33  | 96.80 | 97.63 | 81.74 | 66.13 | _54.77_  | _43.75_  |


## 3. Pre-training from Scratch

### 3-1. Pre-processing data files

M2D-S learns from the following pre-processed files using LibriSpeech (LS-960) and HuBERT-base pre-trained model.

- `data/ls960_hybrid7s_hubaseL9`: Pre-processed data consists of log-mel spectrogram samples converted from LS-960 and HuBERT layer #9 features encoded from LS-960.
- `data/files_ls960_hybrid.csv`: List of pre-processed files of the `ls960_hybrid7s_hubaseL9` folder.

M2D-S also requires AudioSet as a background noise.

- `data/audioset_lms`: Pre-processed log-mel spectrogram samples from AudioSet, as in the original M2D.
- `data/files_audioset.csv`: List of pre-processed AudioSet files, as in the original M2D.

#### 3-1-1. LS-960 data files

The following command line will create `data/ls960_hybrid7s_hubaseL9` and `data/files_ls960_hybrid.csv`.

```
python speech/extract_offline_ls960.py /path/to/LibriSpeech
```

#### 3-1-2. AudioSet data files

For preparing AudioSet data files (`data/audioset_lms` and `data/files_audioset.csv`), please follow the [data/README.md](../data/README.md).

### 3-2. Pre-training

The `train_speech.py` pre-trains for speech.

The following example would run on any affordable GPU, consuming only 7,170MiB. However, please note that it will take very long (It took over 20 minutes for one epoch).
You can also change the BG noise dataset by adding `--csv_bg_noise data/files_fsd50k.csv`, for example.

```sh
python train_speech.py --loss_m2d 1. --loss_off 1. --input_size 80x208 --patch_size 80x4 --noise_ratio 0.2 --batch_size 128 --accum_iter 16
```

The followings are for pre-training high-end models, taking 2.5-3.5 days to complete with 4 A100s.

```sh
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 train_speech.py --loss_m2d 1. --loss_off .5 --input_size 80x400 --patch_size 80x2 --noise_ratio 0.2
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 train_speech.py --loss_m2d 1. --loss_off .5 --input_size 80x512 --patch_size 80x2 --noise_ratio 0.2 --batch_size 256 --accum_iter 2
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 train_speech.py --loss_m2d 1. --loss_off .5 --input_size 80x608 --patch_size 80x2 --noise_ratio 0.2 --batch_size 256 --accum_iter 2
```

#### 3-2-1. Major pre-training options

- --batch_size: Batch size per GPU, 512 by default.
- --epochs: Training epochs, 1000 by default.
- --accum_iter: Iterations to accumulate gradients, 1 by default.
- --input_size: Input spectrogram size, 80x208 by default.
- --patch_size: Patch size, 80x4 by default.
- --mask_ratio: Masking ratio, 0.6 by default.
- --loss_m2d: Loss ratio for M2D masked prediction, 1.0 by default.
- --loss_off: Loss ratio for offline target, 0.0 by default.
- --blr: Base learning rate: absolute_lr = base_lr * total_batch_size / 256.
- --csv_main: A CSV file to list sample files in the main dataset, 'data/files_ls960_hybrid.csv' by default.
- --csv_bg_noise: A CSV file to list sample files in the BG noise dataset, 'data/files_audioset.csv' by default.
- --noise_ratio: Noise mixing ratio, 0.2 by default.

## 4. SUPERB Evaluation

We provide upstream wrapper implementation, which you can import to your [SUPERB](https://arxiv.org/abs/2105.01051) environment.

- Copy the `superb/upstream/m2d` folder under your `s3prl/upstream` folder.
- Make a symbolic link to your copy of M2D repository under your `s3prl/upstream/m2d`, making `s3prl/upstream/m2d/m2d`. The wrapper files will find M2D programs under this symbolic link.
- You will need to run `pip install -e .` under your `s3prl` folder, so that you install your local SUPERB in your Python environment.

Please refer to [superb/upstream/m2d/README.md](../superb/upstream/m2d/README.md) for more details.

## Acknowledgements

- Our code is based on the [MAE PyTorch/GPU re-implementation](https://github.com/facebookresearch/mae) of the paper [Masked Autoencoders Are Scalable Vision Learners](https://openaccess.thecvf.com/content/CVPR2022/html/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.html).
- We use [nnAudio](https://ieeexplore.ieee.org/document/9174990) ([KinWaiCheuk/nnAudio](https://github.com/KinWaiCheuk/nnAudio)) for converting raw audio into log-mel spectrogram.
- We use [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) for the implementation and pre-trained weights of the [HuBERT](https://ieeexplore.ieee.org/document/9585401) model.

We appreciate these publicly available resources.

## References

If you find our M2D-S useful in your research, please consider citing our paper:

```BibTeX
@article{niizumi2023m2d4speech,
    title   = {{Masked Modeling Duo for Speech: Specializing General-Purpose Audio Representation to Speech using Denoising Distillation}},
    author  = {Daisuke Niizumi and Daiki Takeuchi and Yasunori Ohishi and Noboru Harada and Kunio Kashino},
    journal = {to appear at Interspeech}, 
    year    = {2023},
    url     = {https://arxiv.org/abs/2305.14079}
}
```

- SUPERB: *[Shu-wen Yang, Po-Han Chi, Yung-Sung Chuang, Cheng-I Jeff Lai, Kushal Lakhotia, Yist Y. Lin, Andy T. Liu, Jiatong Shi, Xuankai Chang, Guan-Ting Lin, Tzu-Hsien Huang, Wei-Cheng Tseng, Ko-tik Lee, Da-Rong Liu, Zili Huang, Shuyan Dong, Shang-Wen Li, Shinji Watanabe, Abdelrahman Mohamed, and Hung-yi Lee, "SUPERB: Speech Processing Universal PERformance Benchmark," Interspeech, 2021](https://arxiv.org/abs/2105.01051).*
    - https://github.com/s3prl/s3prl/blob/main/s3prl/downstream/docs/superb.md
- HuBERT: *[W.-N. Hsu, B. Bolte, Y.-H. H. Tsai, K. Lakhotia, R. Salakhutdinov, and A. Mohamed, “HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units,” IEEE/ACM Trans. Audio, Speech, Language Process., p.3451–3460, 2021](https://ieeexplore.ieee.org/document/9585401).*
