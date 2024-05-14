#!/bin/bash

if [ $# -lt 2 ]; then
  n_iter=3
else
  n_iter=$2
fi

if [ $# -lt 3 ]; then
  lr_prm=5e-6
else
  lr_prm=$3
fi

echo Repeating $n_iter times...

for i in $(seq $n_iter); do
    cmdline="CUDA_VISIBLE_DEVICES=0 python app_main.py --dataset SPRS --datapath data/SPRS --method sl --backbone m2d --epochs 50 --bs 64 --weightspath $1 --lr $lr_prm --freeze_embed --split_iter 4"
    echo $cmdline
    eval $cmdline
done
