cd evar
GPU=0

if [[ "$1" == *'p32k-'* ]]; then
    cfg='config/m2d_32k.yaml'
    cfg_clap='config/m2d_clap_32k.yaml'
else
    cfg='config/m2d.yaml'
    cfg_clap='config/m2d_clap.yaml'
fi

if [[ "$1" == *'_clap'* ]]; then
    zs_opt=',flat_features=True'
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

if [[ "$1" == *'_clap'* ]] || [[ "$1" == *'_capgte_'* ]]; then
    echo 'Zero-shot evaluation'
    CUDA_VISIBLE_DEVICES=$GPU python zeroshot.py $cfg_clap cremad batch_size=16,weight_file=$1$zs_opt
    CUDA_VISIBLE_DEVICES=$GPU python zeroshot.py $cfg_clap gtzan batch_size=16,weight_file=$1$zs_opt
    CUDA_VISIBLE_DEVICES=$GPU python zeroshot.py $cfg_clap nsynth batch_size=64,weight_file=$1$zs_opt
    CUDA_VISIBLE_DEVICES=$GPU python zeroshot.py $cfg_clap esc50 batch_size=64,weight_file=$1$zs_opt
    CUDA_VISIBLE_DEVICES=$GPU python zeroshot.py $cfg_clap us8k batch_size=64,weight_file=$1$zs_opt
    CUDA_VISIBLE_DEVICES=$GPU python zeroshot.py $cfg_clap fsd50k batch_size=64,weight_file=$1$zs_opt
    CUDA_VISIBLE_DEVICES=$GPU python zeroshot.py $cfg_clap as batch_size=64,weight_file=$1$zs_opt
fi

python summarize.py $1
