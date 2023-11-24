
python medsam_cancer_detection_v2.py \
    --cluster submitit_auto \
    --slurm_gres gpu:a40:1 \
    --timeout_min 480 \
    --use_augmentation \
    --benign_cancer_ratio_for_training 3 \
    --fold 0 \
    --epochs 30 \
    --model_config MedSAMCancerDetectorV2 \
    --min_involvement_pct_training 0 \
    --loss involvement_tolerant_loss \
    --accumulate_grad_steps 1 \
    --batch_size 8 

