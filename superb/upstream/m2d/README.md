# Masked Modeling Duo (M2D) upstream model for SUPERB

Masked Modeling Duo for Speech: Specializing General-Purpose Audio Representation to Speech using Denoising Distillation
https://arxiv.org/abs/2305.14079

This is an M2D wrapper for SUPERB, and evaluating M2D on SUPERB involves two steps:

- Calculating normalization statistics first, as M2D requires the average and standard deviation of the downstream task dataset.
- Evaluating M2D on SUPERB using the calculated statistics.

## Installation

- Copy the `superb/upstream/m2d` folder under your `s3prl/upstream` folder.
- Create a copy of entire M2D repository under your `s3prl/upstream/m2d`, or make a symbolic link instead. The wrapper `expert.py` will find M2D programs in the folder.
- Edit your `s3prl/hub.py` to add `from s3prl.upstream.m2d.hubconf import *`.

The expected folders/files are as follows:

    your_s3prl/
        s3prl/
            upstream/
                m2d/
                    __init__.py
                    expert.py
                    hubconf.py
                    README.md
                    m2d/
                        (all the M2D contents should be here)
            hub.py  (should have `from s3prl.upstream.m2d.hubconf import *`)

You might also need to run `pip install -e .` under your `s3prl` folder, so that you install your local SUPERB in your Python environment.

Here is an example of installing fresh SUPERB under your copy of the M2D repository.

    git clone https://github.com/s3prl/s3prl.git
    ln -s ../../../superb/upstream/m2d s3prl/s3prl/upstream/
    ln -s ../../.. s3prl/s3prl/upstream/m2d/m2d
    pip install tensorboardX catalyst
    cd s3prl/s3prl
    (Now edit hub.py to add the following line.)
        from s3prl.upstream.m2d.hubconf import *
    cd ..   (move to your_m2d/s3prl)
    pip install -e .

After these steps, your SUPERB should accept the following evaluation steps.

## Step 1. Pre-compute statistics on a downstream task

We need statistics for each downstream task.

Use the upstream  `m2d_calcnorm` to calculate statistics. Example with a downstream task `voxceleb1` (SID):

    python run_downstream.py -m train -n m2d_calcnorm_1 -u m2d_calcnorm -d voxceleb1

This will output:

    *** Running Norm has finished updates over 10000 times, using the following stats from now on. ***
    mean=-10.571270942687988, std=4.3681135177612305
    *** Please use these statistics in your model. EXIT... ***

These `-10.571270942687988` and `4.3681135177612305` are the statistics for the `voxceleb1` (SID).

## Step 2. Run your evaluation on the downstream task

Use the upstream `m2d` to evaluate your weights with the statistics calculated in the step above.
Here an example of testing m2d_s_vit_base-80x608p80x2-230220 using `voxceleb1` (SID):

    python run_downstream.py -m train -n m2d_vc1_1 -u m2d -d voxceleb1 -k /your/m2d_s_vit_base-80x608p80x2-230220/checkpoint-1000.pth,-10.571271,4.3681135
    python run_downstream.py -m evaluate -e result/downstream/m2d_vc1_1/dev-best.ckpt

## Examples

These are the scripts used for evaluating "[Masked Modeling Duo for Speech: Specializing General-Purpose Audio Representation to Speech using Denoising Distillation](https://arxiv.org/abs/2305.14079)."

For example, we run the following to evaluate a weight `m2d_s_vit_base-80x608p80x2-230220/checkpoint-1000.pth` on KS.

    ./ks.sh 0 m2d /your/m2d_s_vit_base-80x608p80x2-230220/checkpoint-1000.pth 7

The `0` is a GPU number, `m2d` is an upstream name, and the last `7` is a random seed.
This command line will run the following two Python commands:

    CUDA_VISIBLE_DEVICES=0 python run_downstream.py -m train -n m2d_s_vit_base-80x608p80x2-230220-checkpoint-1000-KS-lr1e-4-s7 -u m2d -d speech_commands -o config.optimizer.lr=1e-4 -k /your/m2d_s_vit_base-80x608p80x2-230220/checkpoint-1000.pth,-11.506255149841309,4.314857482910156 --seed 7
    CUDA_VISIBLE_DEVICES=0 python run_downstream.py -m evaluate -e result/downstream/m2d_s_vit_base-80x608p80x2-230220-checkpoint-1000-KS-lr1e-4-s7/dev-best.ckpt


### ER (er.sh)

```sh
gpu=$1
upmodel=$2
ckpt=$3
lr=1e-5
task=ER
seed=$4

parentpath=$(dirname $ckpt)
parent=$(basename $parentpath)
ckptbase=$(basename $ckpt)
ckptstem=${ckptbase%.*}
expbase=$parent-$ckptstem

for test_fold in fold1 fold2 fold3 fold4 fold5;
do
    expname=$expbase-$task-lr$lr-s$seed-$test_fold
    echo $expname
    CUDA_VISIBLE_DEVICES=$gpu python run_downstream.py -m train -n $expname -u $upmodel -d emotion -c downstream/emotion/config.yaml -o "config.optimizer.lr=$lr,, config.downstream_expert.datarc.test_fold='$test_fold'" -k $ckpt,-13.037399291992188,3.619741439819336 --seed $seed
    CUDA_VISIBLE_DEVICES=$gpu python run_downstream.py -m evaluate -e result/downstream/$expname/dev-best.ckpt
done
```

### IC (ic.sh)

```sh
gpu=$1
upmodel=$2
ckpt=$3
lr=1e-3
task=IC
seed=$4

parentpath=$(dirname $ckpt)
parent=$(basename $parentpath)
ckptbase=$(basename $ckpt)
ckptstem=${ckptbase%.*}
expbase=$parent-$ckptstem

expname=$expbase-$task-lr$lr-s$seed

echo $expname
CUDA_VISIBLE_DEVICES=$gpu python run_downstream.py -m train -n $expname -u $upmodel -d fluent_commands -o "config.optimizer.lr=$lr" -k $ckpt,-13.017439842224121,4.417759895324707 --seed $seed
CUDA_VISIBLE_DEVICES=$gpu python run_downstream.py -m evaluate -e result/downstream/$expname/dev-best.ckpt
```

### KS (ks.sh)

```sh
gpu=$1
upmodel=$2
ckpt=$3
lr=1e-4
task=KS
seed=$4

parentpath=$(dirname $ckpt)
parent=$(basename $parentpath)
ckptbase=$(basename $ckpt)
ckptstem=${ckptbase%.*}
expbase=$parent-$ckptstem

expname=$expbase-$task-lr$lr-s$seed

echo $expname
CUDA_VISIBLE_DEVICES=$gpu python run_downstream.py -m train -n $expname -u $upmodel -d speech_commands -o "config.optimizer.lr=$lr" -k $ckpt,-11.506255149841309,4.314857482910156 --seed $seed
CUDA_VISIBLE_DEVICES=$gpu python run_downstream.py -m evaluate -e result/downstream/$expname/dev-best.ckpt
```

### PR (pr.sh)

```sh
gpu=$1
upmodel=$2
ckpt=$3
lr=1e-3
task=PR
seed=$4

parentpath=$(dirname $ckpt)
parent=$(basename $parentpath)
ckptbase=$(basename $ckpt)
ckptstem=${ckptbase%.*}
expbase=$parent-$ckptstem

expname=$expbase-$task-lr$lr-s$seed

echo $expname
CUDA_VISIBLE_DEVICES=$gpu python run_downstream.py -m train -n $expname -u $upmodel -d ctc -c downstream/ctc/libriphone.yaml -o "config.optimizer.lr=$lr" -k $ckpt,-10.43253231048584,4.241369724273682 --seed $seed
CUDA_VISIBLE_DEVICES=$gpu python run_downstream.py -m evaluate -n $expname -d ctc -e result/downstream/$expname/dev-best.ckpt
```

### SID (sid.sh)

```sh
gpu=$1
upmodel=$2
ckpt=$3
lr=1e-3
task=SID
seed=$4

parentpath=$(dirname $ckpt)
parent=$(basename $parentpath)
ckptbase=$(basename $ckpt)
ckptstem=${ckptbase%.*}
expbase=$parent-$ckptstem

expname=$expbase-$task-lr$lr-s$seed

echo $expname
CUDA_VISIBLE_DEVICES=$gpu python run_downstream.py -m train -n $expname -u $upmodel -d voxceleb1 -o "config.optimizer.lr=$lr" -k $ckpt,-10.571271,4.3681135 --seed $seed
CUDA_VISIBLE_DEVICES=$gpu python run_downstream.py -m evaluate -n $expname -d voxceleb1 -e result/downstream/$expname/dev-best.ckpt
```

