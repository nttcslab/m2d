# M2D-X Applied on ICBHI2017 & SPRSound

Here are example solutions for two respiratory sound tasks.
The setup procedure looks complicated but should work fine on a typical Ubuntu-based PyTorch environment.

We will provide further details per request; [please add an issue whenever you need it](https://github.com/nttcslab/m2d/issues).

**Under construction**

![Under construction](https://upload.wikimedia.org/wikipedia/commons/d/d9/Under_construction_animated.gif)

## 1. Setup

### 1-1. Download FSD50K and setup the data

Please follow the [steps in the main README](../../README.md#3-1-preparing-pre-training-data-samples).
You should have the following files.

```sh
    data/
        files_f_s_d_5_0_k.csv
        fsd50k_lms/
            FSD50K.dev_audio/
                2931.npy
                408195.npy
                    :
```

### 1-2. Download the ICBHI2017 & SPRSound data

Make sure you are in the `app/icbhi_sprs` folder, or `cd app/icbhi_sprs`.
The following steps will download the ICBHI2017 & SPRSound data.

```sh
echo Downloading ICBHI_final_database.zip ...
wget https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_final_database.zip --no-check-certificate

echo Downloading SPRSound ...
git clone https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound.git
(cd SPRSound && git reset --hard 45b0d5d435ff320c46585762fa1090afd0ebb318)
```

### 1-3. Setup application files

**NOTE: Download the data first. The following steps extract and convert these data.**

Make sure you are in the `app/icbhi_sprs` folder, or `cd app/icbhi_sprs`.

```sh
echo Cloning and copying from https://github.com/ilyassmoummad/scl_icbhi2017.git ...
git clone https://github.com/ilyassmoummad/scl_icbhi2017.git
cd scl_icbhi2017
git reset --hard 915c1120719a9357d662c5fe484bce7fbe845139
mv dataset.py augmentations.py utils.py losses.py args.py ..
mv data ..
mv main.py ../app_main.py
mv ce.py models.py ..
cd ..
pip install torchinfo

echo Unpacking and copying data...
cd data
unzip ../ICBHI_final_database.zip | awk 'BEGIN {ORS=" "} {if(NR%10==0)print "."}'
mv ICBHI_final_database/* ICBHI
(cd ../SPRSound && git reset --hard 45b0d5d435ff320c46585762fa1090afd0ebb318)
cp -r ../SPRSound/train_wav ../SPRSound/test_wav SPRS/
cd ..

echo Creating the pre-training log-mel spectrograms in your_repository_root/data ...
python ../../wav_to_lms.py data/ICBHI ../../data/icbhi2017_lms
python cut_data_sprs.py

echo Creating list of the pre-training data
cp files_icbhi2017.csv ../../data/files_icbhi2017.csv
echo file_name > ../../data/files_sprs.csv
(cd ../../data/ && find sprsound_lms/train -name *.npy) >> ../../data/files_sprs.csv
```

## 2. ICBHI2017

### 2-1. Further Pre-training

```sh
python train_audio.py --epochs 600 --resume m2d_vit_base-80x200p16x4-230529/checkpoint-300.pth --model m2d_x_vit_base --input_size 80x200 --patch_size 16x4 --batch_size 64 --accum_iter 2 --csv_main data/files_icbhi2017.csv --csv_bg_noise data/files_f_s_d_5_0_k.csv --noise_ratio 0.3 --save_freq 100 --eval_after 600 --seed 6 --teacher m2d_vit_base-80x200p16x4-230529/checkpoint-300.pth --blr 3e-4 --loss_off 1.
```

### 2-2. Fine-tuning

An example command line using further pre-trained weight `m2d_x_vit_base-80x200p16x4p16k-240410-MdfiDdffsd50ks6bs128a2lo1nr.3-e600/checkpoint-600.pth`:

```sh
python app_main.py --method sl --backbone m2d --epochs 150 --bs 64 --weightspath m2d_x_vit_base-80x200p16x4p16k-240410-MdfiDdffsd50ks6bs128a2lo1nr.3-e600/checkpoint-600.pth --lr 5e-5 --freeze_embed --split_iter 4
```

## 3. SPRSound

### 3-1. Further Pre-training

```sh
python train_audio.py --epochs 600 --resume m2d_vit_base-80x200p16x4-230529/checkpoint-300.pth --model m2d_x_vit_base --input_size 80x200 --patch_size 16x4 --batch_size 64 --accum_iter 2 --csv_main data/files_sprs.csv --csv_bg_noise data/files_f_s_d_5_0_k.csv --noise_ratio 0.01 --save_freq 100 --eval_after 600 --seed 6 --teacher m2d_vit_base-80x200p16x4-230529/checkpoint-300.pth --blr 3e-4 --loss_off 1.
```

### 2-2. Fine-tuning

T.B.D.