
python medsam_cancer_detection_v2.py \
    --cluster slurm \
    --timeout_min 480 \
    --use_augmentation \
    --benign_cancer_ratio_for_training 2 \
    --fold 0 \
    --epochs 30 \
    --model_config MedSAMCancerDetectorV2 \
    --needle_threshold -1 


#    --medsam_checkpoint /h/pwilson/projects/medAI/projects/sam/logs/finetune_medsam/2023-11-16-10:31:49-heavy-polecat/checkpoints/medsam-finetuned_image_encoder_aligned_files_0.9322000374454065.pth \