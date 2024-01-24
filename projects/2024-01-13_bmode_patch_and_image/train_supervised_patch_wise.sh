#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00 
#SBATCH --mem=32GB
#SBATCH --output=slurm-%j.log
#SBATCH -c 16
#SBATCH --qos=m2

export TQDM_MININTERVAL=30

/h/pwilson/anaconda3/envs/ai/bin/python train_supervised_patch_wise.py \
    --model-name=resnet10t_instance_norm 
    