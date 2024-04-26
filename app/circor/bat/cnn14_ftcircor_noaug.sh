#!/bin/bash
# CUDA_VISIBLE_DEVICES=0 python circor_eval.py config/cnn14.yaml circor1 --lr=1e-5 --freq_mask 0 --time_mask 0  -mixup 0.0 --rrc False --epochs 50 --warmup_epochs 5 --seed 8 --batch_size 256

split=$1
n_iter=$2
seed=$3
lr_prm=1e-3
bs=256
gpu=0

echo Repeating $n_iter times...

for i in $(seq $n_iter); do
  seed=$((seed + 1))
  cmdline="CUDA_VISIBLE_DEVICES=$gpu python circor_eval.py config/cnn14.yaml circor$split --lr=$lr_prm --freq_mask 0 --time_mask 0  -mixup 0.0 --rrc False --epochs 50 --warmup_epochs 5 --seed $seed --batch_size $bs"
  echo $cmdline
  eval $cmdline
done
