#!/bin/bash

split=$2
n_iter=$3
seed=$4
#lr_prm=0.001 for bs128
lr_prm=0.00025
bs=32
gpu=0

echo Repeating $n_iter times...

for i in $(seq $n_iter); do
  weight=$1/random
  seed=$((seed + 1))
  cmdline="CUDA_VISIBLE_DEVICES=$gpu python circor_eval.py config/m2d.yaml circor$split weight_file=$weight,encoder_only=True,freeze_embed=True --lr=$lr_prm --freq_mask 0 --time_mask 0 --training_mask 0.2 --mixup 0.0 --rrc False --epochs 50 --warmup_epochs 5 --seed $seed --batch_size $bs"
  echo $cmdline
  eval $cmdline
done
