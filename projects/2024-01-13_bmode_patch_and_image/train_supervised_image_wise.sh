#!/bin/bash

#SBATCH --gres=gpu:a40:1
#SBATCH --time=8:00:00 
#SBATCH --mem=32GB
#SBATCH --output=slurm-%j.log
#SBATCH -c 16
#SBATCH --qos=m

export TQDM_MININTERVAL=30

/h/pwilson/anaconda3/envs/ai/bin/python train_supervised_image_wise.py \
    --model segmentation \
    --name sam_needle_region \
    --augmentations v1 \
    --benign-cancer-ratio-train 1 \
    --scheduler warmup_cosine \
    --lr 1e-5 \
    --segmentation-backbone medsam \
    --pos-weight 1 \
    --needle-mask-threshold 0.6 \
    --prostate-mask-threshold 0.9 \
    --epochs 40 \
    --compile-model