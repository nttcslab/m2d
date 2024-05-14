gpu=$1
upmodel=$2
ckpt=$3
lr=5e-5
task=ASV
seed=$4

parentpath=$(dirname $ckpt)
parent=$(basename $parentpath)
ckptbase=$(basename $ckpt)
ckptstem=${ckptbase%.*}
expbase=$parent-$ckptstem

expname=$expbase-$task-lr$lr-s$seed

echo $expname
CUDA_VISIBLE_DEVICES=$gpu python run_downstream.py -m train -n $expname -u $upmodel -d sv_voxceleb1 -o "config.optimizer.lr=$lr" -k $ckpt,-11.070931,4.1807961 --seed $seed
CUDA_VISIBLE_DEVICES=$gpu ./downstream/sv_voxceleb1/test_expdir.sh result/downstream/$expname /lab/data/superb/voxceleb1
