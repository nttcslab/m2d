cd evar
CUDA_VISIBLE_DEVICES=0 python 2pass_lineareval.py config/m2d.yaml cremad batch_size=16,weight_file=$1
CUDA_VISIBLE_DEVICES=0 python 2pass_lineareval.py config/m2d.yaml gtzan batch_size=16,weight_file=$1
CUDA_VISIBLE_DEVICES=0 python 2pass_lineareval.py config/m2d.yaml spcv2 batch_size=64,weight_file=$1
CUDA_VISIBLE_DEVICES=0 python 2pass_lineareval.py config/m2d.yaml esc50 batch_size=64,weight_file=$1
CUDA_VISIBLE_DEVICES=0 python 2pass_lineareval.py config/m2d.yaml us8k batch_size=64,weight_file=$1
CUDA_VISIBLE_DEVICES=0 python 2pass_lineareval.py config/m2d.yaml vc1 batch_size=64,weight_file=$1
CUDA_VISIBLE_DEVICES=0 python 2pass_lineareval.py config/m2d.yaml voxforge batch_size=64,weight_file=$1
CUDA_VISIBLE_DEVICES=0 python 2pass_lineareval.py config/m2d.yaml nsynth batch_size=64,weight_file=$1
CUDA_VISIBLE_DEVICES=0 python 2pass_lineareval.py config/m2d.yaml surge batch_size=64,weight_file=$1
python summarize.py $1
