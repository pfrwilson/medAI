#!/bin/bash

#SBATCH --mem=16G
#SBATCH --gres=gpu:a40:1
#SBATCH --time 8:00:00
#SBATCH -c 16 
#SBATCH --qos=normal
#SBATCH --output=slurm-%j.log
#SBATCH --open-mode=append

# send this batch script a SIGUSR1 60 seconds
# before we hit our time limit
#SBATCH --signal=B:USR1@60

export TQDM_MININTERVAL=30
export EXP_DIR=$(realpath experiments/$SLURM_JOB_ID)
export CKPT_DIR=/checkpoint/$USER/$SLURM_JOB_ID
export WANDB_RUN_ID=$SLURM_JOB_ID

# Symbolic link to checkpoint directory
# so it is easier to find them
ln -s $CKPT_DIR $EXP_DIR

echo "EXP_DIR: $EXP_DIR"

resubmit() {
    echo "Resubmitting job"
    scontrol requeue $SLURM_JOB_ID
    exit 0
}

trap resubmit SIGUSR1

python train_medsam.py hydra.run.dir=$EXP_DIR exp_state_path=$EXP_DIR/state.pth checkpoint_dir=$CKPT_DIR


