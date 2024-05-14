#!/bin/bash

# Usage:
#     bash (your m2d)/util/ft-as2m.sh <weight folder path> <# of iteration> <random seed> <checkpoint epochs to test>
#
# Example: The parameter `300` will test the checkpoint-300.pth
#     cd evar
#     bash (your m2d)/util/ft-as2m.sh (your m2d)/m2d_vit_base-80x608p16x16-221006-mr7 3 42 300

# CONFIGURE HERE: Set your the same data path as pre-training here
# Fine-tuning on AS2M requires the log-mel spectrogram audio files.
# Prepare data/audioset_lms according to the [Example preprocessing steps (AudioSet)](../data/README.md#example-preprocessing-steps-audioset).
datapath=../data/audioset_lms

# Fine-tuning steps follow
n_iter=$2
seed=$3

echo **AS2M** Repeating $n_iter times...

for i in $(seq $n_iter); do
  for w in ${@:4}; do
    weight=$1/checkpoint-$w.pth
    seed=$((seed + 1))
    cmdline="python finetune.py config/m2d.yaml as weight_file=$weight,encoder_only=True,dur_frames=1001 --lr=2.0 --freq_mask 30 --time_mask 192 --training_mask 0.5 --mixup 0.5 --rrc False --epochs 70 --warmup_epochs 15 --optim lars --batch_size 64 --data_path $datapath --seed $seed"
    echo $cmdline
    eval $cmdline
  done
done