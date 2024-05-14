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
