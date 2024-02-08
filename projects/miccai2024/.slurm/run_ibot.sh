#!/bin/bash

#SBATCH --mem=16G
#SBATCH --gres=gpu:a40:1
#SBATCH --time 16:00:00
#SBATCH -c 16 
#SBATCH --qos=normal
#SBATCH --output=slurm-%j.log
#SBATCH --open-mode=append

# send this batch script a SIGUSR1 4 minutes
# before we hit our time limit
#SBATCH --signal=B:USR1@240

CENTER=UA
EXP_NAME=${CENTER}_ibot
EXP_DIR=experiments/${EXP_NAME}/$SLURM_JOB_ID
CKPT_DIR=/checkpoint/$USER/$SLURM_JOB_ID

# Set environment variables for training
export TQDM_MININTERVAL=30
export WANDB_RUN_ID=$SLURM_JOB_ID
export WANDB_RESUME=allow

# Create experiment directory
echo "EXP_DIR: $EXP_DIR"
mkdir -p $EXP_DIR

# Symbolic link to checkpoint directory
# so it is easier to find them
echo "CKPT_DIR: $CKPT_DIR"
# only do it if the directory does not exist
if [ ! -d $EXP_DIR/checkpoints ]; then
  ln -s $CKPT_DIR $(realpath $EXP_DIR)/checkpoints
fi

# Kill training process and resubmit job if it receives a SIGUSR1
handle_timeout_or_preemption() {
  date +"%Y-%m-%d %T"
  echo "Caught timeout or preemption signal"
  echo "Sending SIGINT to child process"
  scancel $SLURM_JOB_ID --signal=SIGINT
  wait $child_pid
  echo "Job step terminated gracefully"
  echo $(date +"%Y-%m-%d %T") "Resubmitting job"
  scontrol requeue $SLURM_JOB_ID
  exit 0
}
trap handle_timeout_or_preemption SIGUSR1

# Run training script
srun python train_medsam_ibot_style.py \
  --name $EXP_NAME \
  --nouse_nct_data \
  --use_ua_unlabelled_data \
  --crop_scale_1 0.2 1 \
  --crop_scale_2 0.8 1 \
  --random_gamma 0.5 1.5 \
  --ckpt_dir $CKPT_DIR \
  --log_image_freq 500 \
  --batch_size 2 \
  --out_dim 8192 \
  --patch_out_dim 8192 \
  --use_amp \
  --project miccai2024_ssl & 
  
child_pid=$!
wait $child_pid
