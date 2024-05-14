gpu=$1
upmodel=$2
ckpt=$3
lr=1e-3
task=SS
seed=$4

parentpath=$(dirname $ckpt)
parent=$(basename $parentpath)
ckptbase=$(basename $ckpt)
ckptstem=${ckptbase%.*}
expbase=$parent-$ckptstem

expname=$expbase-$task-lr$lr-s$seed

echo $expname
CUDA_VISIBLE_DEVICES=$gpu python run_downstream.py -m train -n $expname -u $upmodel -d separation_stft2 -o "config.optimizer.lr=$lr" -k $ckpt,-9.58743667602539,4.168412208557129 --seed $seed -c downstream/separation_stft2/configs/cfg.yaml
CUDA_VISIBLE_DEVICES=$gpu python run_downstream.py -m evaluate -n $expname -d separation_stft2 -e result/downstream/$expname/best-states-dev.ckpt
