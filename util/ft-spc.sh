#!/bin/bash

# Usage:
#     bash (your m2d)/util/ft-spc.sh <weight folder path> <# of iteration> <random seed> <checkpoint epochs to test>
#
# Example: The parameter `300` will test the checkpoint-300.pth
#     cd evar
#     bash (your m2d)/util/ft-spc.sh (your m2d)/m2d_vit_base-80x608p16x16-221006-mr7 3 42 300

n_iter=$2
seed=$3

echo **SPCV2** Repeating $n_iter times...

for i in $(seq $n_iter); do
  for w in ${@:4}; do
    weight=$1/checkpoint-$w.pth
    seed=$((seed + 1))
    cmdline="python finetune.py config/m2d.yaml spcv2 weight_file=$weight,encoder_only=True --lr=0.5 --freq_mask 30 --time_mask 48 --training_mask 0.5 --mixup 0.3 --rrc True --seed $seed"
    echo $cmdline
    eval $cmdline
  done
done