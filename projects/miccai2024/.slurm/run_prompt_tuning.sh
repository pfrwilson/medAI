#!/bin/bash
#SBATCH --mem=32G
#SBATCH --gres=gpu:a40:1
#SBATCH --time 16:00:00
#SBATCH -c 16 
#SBATCH --output=slurm-%j.log
#SBATCH --open-mode=append
#SBATCH --account=deadline
#SBATCH --qos=deadline

# send this batch script a SIGUSR1 240 seconds
# before we hit our time limit
#SBATCH --signal=B:USR1@240


CENTER=UVA
EXP_NAME=${CENTER}_ProFound_PromptTuning
RUN_ID=$SLURM_JOB_ID
EXP_DIR=experiments/${EXP_NAME}/$RUN_ID
CKPT_DIR=/checkpoint/$USER/$RUN_ID

# Set environment variables for training
export TQDM_MININTERVAL=30
export WANDB_RUN_ID=$RUN_ID
export WANDB_RESUME=allow
export PYTHONUNBUFFERED=1

# Create experiment directory
echo "EXP_DIR: $EXP_DIR"
mkdir -p $EXP_DIR
# Symbolic link to checkpoint directory
# so it is easier to find them
echo "CKPT_DIR: $CKPT_DIR"
ln -s $CKPT_DIR $(realpath $EXP_DIR)/checkpoints

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
srun -u python train_profound.py \
  --test_center $CENTER \
  --min_involvement_train 40 \
  --augmentations translate \
  --batch_size 4 \
  --undersample_benign_ratio 6 \
  --remove_benign_cores_from_positive_patients \
  --lr 1e-4 \
  --encoder_lr 0 \
  --freeze_image_encoder \
  --cnn_lr 0 \
  --freeze_cnn \
  --freeze_mask_decoder \
  --warmup_epochs 0 \
  --epochs 35 \
  --wd 0 \
  --prompts data_independent_prompts age psa sparse_cnn_patch_features \
  --prompt_dropout 0.0 \
  --sparse_cnn_backbone_path /h/pwilson/projects/medAI/projects/miccai2024/checkpoints/${CENTER}_patch_ssl_0.pth \
  --backbone medsam \
  --test_every_epoch \
  --loss_0_name valid_region \
  --loss_0_base_loss_name ce \
  --loss_0_pos_weight 2 \
  --loss_0_prostate_mask True \
  --loss_0_needle_mask True \
  --loss_0_weight 1 \
  --device cuda \
  --accumulate_grad_steps 2 \
  --exp_dir $EXP_DIR \
  --checkpoint_dir $CKPT_DIR \
  --name $EXP_NAME \
  --log_images \
  --use_amp \
  --seed 42 & 

child_pid=$!
wait $child_pid

