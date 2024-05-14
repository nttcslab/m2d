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
