#!/bin/bash
# CUDA_VISIBLE_DEVICES=0 python circor_eval.py config/byola.yaml circor1 weight_file=external/byol_a/pretrained_weights/AudioNTT2020-BYOLA-64x96d2048.pth --lr=3e-5 --freq_mask 20 --time_mask 50  -mixup 0.0 --rrc False --epochs 50 --warmup_epochs 5 --seed 7 --batch_size 256

split=$1
n_iter=$2
seed=$3
lr_prm=0.001
bs=256
gpu=0

echo Repeating $n_iter times...

for i in $(seq $n_iter); do
  weight="external/byol_a/pretrained_weights/AudioNTT2020-BYOLA-64x96d2048.pth"
  seed=$((seed + 1))
  cmdline="CUDA_VISIBLE_DEVICES=$gpu python circor_eval.py config/byola.yaml circor$split weight_file=$weight --lr=$lr_prm --freq_mask 20 --time_mask 50  -mixup 0.0 --rrc False --epochs 50 --warmup_epochs 5 --seed $seed --batch_size $bs"
  echo $cmdline
  eval $cmdline
done
