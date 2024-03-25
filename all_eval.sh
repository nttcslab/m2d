cd evar
GPU=0

if [[ "$1" == *'p32k-'* ]]; then
    cfg='config/m2d_32k.yaml'
else
    cfg='config/m2d.yaml'
fi

CUDA_VISIBLE_DEVICES=$GPU python 2pass_lineareval.py $cfg cremad batch_size=16,weight_file=$1
CUDA_VISIBLE_DEVICES=$GPU python 2pass_lineareval.py $cfg gtzan batch_size=16,weight_file=$1
CUDA_VISIBLE_DEVICES=$GPU python 2pass_lineareval.py $cfg spcv2 batch_size=64,weight_file=$1
CUDA_VISIBLE_DEVICES=$GPU python 2pass_lineareval.py $cfg esc50 batch_size=64,weight_file=$1
CUDA_VISIBLE_DEVICES=$GPU python 2pass_lineareval.py $cfg us8k batch_size=64,weight_file=$1
CUDA_VISIBLE_DEVICES=$GPU python 2pass_lineareval.py $cfg vc1 batch_size=64,weight_file=$1
CUDA_VISIBLE_DEVICES=$GPU python 2pass_lineareval.py $cfg voxforge batch_size=64,weight_file=$1
CUDA_VISIBLE_DEVICES=$GPU python 2pass_lineareval.py $cfg nsynth batch_size=64,weight_file=$1
CUDA_VISIBLE_DEVICES=$GPU python 2pass_lineareval.py $cfg surge batch_size=64,weight_file=$1

python summarize.py $1
