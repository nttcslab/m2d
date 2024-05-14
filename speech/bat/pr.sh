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
