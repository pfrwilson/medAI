# # ensemble experiment
# NUM_ENSEMBLES=5
# INSTANCE_NORM=False
# USE_BATCH_NORM=False
# GROUP="ensemble_${NUM_ENSEMBLES}mdls_gn_3ratio_loco2"

# for CENTER in "JH" "PCC" "PMCC" "UVA" "CRCEO"  
# do
#     python ensemble_experiment.py \
#         --name "${GROUP}_${CENTER}" \
#         --group "${GROUP}" \
#         --cluster "slurm" \
#         --slurm_gres "gpu:a40:1" \
#         --slurm_qos "deadline" \
#         --slurm_account "deadline" \
#         --num_ensembles $NUM_ENSEMBLES \
#         --cohort_selection_config "loco" \
#         --leave_out $CENTER \
#         --instance_norm $INSTANCE_NORM \
#         --use_batch_norm $USE_BATCH_NORM \
#         --benign_to_cancer_ratio_train 3.0       
# done 


# # sngp experiment
# INSTANCE_NORM=True
# GROUP="sngp_inst-nrm_loco"
# LR=0.001
# WEIGHT_DECAY=0.0001
# for CENTER in "UVA" # "CRCEO" "JH" "PCC" "PMCC" 
# do
#     python sngp_experiment.py \
#         --name "${GROUP}_${CENTER}_32bz_lre-3" \
#         --group "${GROUP}" \
#         --cluster "slurm" \
#         --slurm_gres "gpu:a40:1" \
#         --lr $LR \
#         --weight_decay $WEIGHT_DECAY \
#         --cohort_selection_config "loco" \
#         --leave_out $CENTER \
#         --batch_size 32 \
#         --instance_norm $INSTANCE_NORM                
# done 


# # baseline experiment
# INSTANCE_NORM=False
# USE_BATCH_NORM=True
# # GROUP="baseline_gn_avgprob_3ratio_loco"
# GROUP="baseline_bn_avgprob_3ratio_loco"
# # GROUP="baseline_gn_avgprob_3ratio_1poly_loco"
# # GROUP="sam_baseline_gn_e-4rho_loco"
# # GROUP="baseline_bn_inst-nrm_loco"

# for CENTER in "CRCEO" # "UVA" "JH" "PMCC" "PCC" 
# do
#     python baseline_experiment.py \
#         --name "${GROUP}_${CENTER}" \
#         --group "${GROUP}" \
#         --slurm_gres "gpu:a40:1" \
#         --cluster "slurm" \
#         --cohort_selection_config "loco" \
#         --leave_out $CENTER \
#         --instance_norm $INSTANCE_NORM \
#         --use_batch_norm $USE_BATCH_NORM \
#         --benign_to_cancer_ratio_train 3.0 \
#         --use_poly1_loss False \
#         --eps 1.0 \
#         --patch_size_mm 5.0 5.0 \
#         --strides 1.0 1.0 \
#         --lr 0.0001
# done



# # ttt experiment
# QUERY_PATCH=True
# SUPPORT_PATCHES=0
# GROUP="ttt_${SUPPORT_PATCHES}+1sprt_0.1beta_JT_3ratio_loco"

# for CENTER in  "JH" "PCC" "PMCC"  "CRCEO" "UVA" #
# do
#     python ttt_experiment.py \
#         --name "${GROUP}_${CENTER}" \
#         --group "${GROUP}" \
#         --cluster "slurm" \
#         --slurm_gres "gpu:a40:1" \
#         --cohort_selection_config "loco" \
#         --benign_to_cancer_ratio_train 3.0 \
#         --leave_out $CENTER \
#         --include_query_patch $QUERY_PATCH \
#         --num_support_patches $SUPPORT_PATCHES \
#         --joint_training True \
#         --adaptation_steps 1 \
#         --adaptation_lr 0.001 \
#         --beta_byol 0.1
# done 


# mt3 experiment
QUERY_PATCH=True
SUPPORT_PATCHES=0
GROUP="mt3_${SUPPORT_PATCHES}+1sprt_0.1beta_e-3innlr_3ratio_loco"

for CENTER in  "JH" "PCC" "PMCC" "UVA" "CRCEO" # 
do
    python mt3_experiment.py \
        --name "${GROUP}_${CENTER}" \
        --group "${GROUP}" \
        --cluster "slurm" \
        --slurm_gres "gpu:a40:1" \
        --slurm_qos "deadline" \
        --slurm_account "deadline" \
        --cohort_selection_config "loco" \
        --leave_out $CENTER \
        --include_query_patch $QUERY_PATCH \
        --num_support_patches $SUPPORT_PATCHES \
        --benign_to_cancer_ratio_train 3.0 \
        --inner_steps 1 \
        --inner_lr 0.001 \
        --beta_byol 0.1
done 


# # vicreg pretrain experiment
# INSTANCE_NORM=False
# USE_BATCH_NORM=False
# GROUP="vicreg_pretrn_1024zdim_gn_300ep_3ratio_loco"
# for CENTER in "UVA" "CRCEO" #"PCC" # "PMCC" #  "JH"     
# do
#     python vicreg_pretrain_experiment.py \
#         --name "${GROUP}_${CENTER}" \
#         --group "${GROUP}" \
#         --cluster "slurm" \
#         --slurm_gres "gpu:a40:1" \
#         --cohort_selection_config "loco" \
#         --leave_out $CENTER \
#         --instance_norm $INSTANCE_NORM \
#         --use_batch_norm $USE_BATCH_NORM \
#         --benign_to_cancer_ratio_train 3.0 \
#         --epochs 300 \
#         --proj_output_dim 1024 \
#         --cov_coeff 1.0 \
#         --linear_lr 0.001 \
#         --linear_epochs 15
# done


# # vicreg pretrain experiment
# INSTANCE_NORM=False
# USE_BATCH_NORM=True
# GROUP="vicreg_pretrn_5e-3-20linprob_bn_f"
# # --group "${GROUP}" \
# for FOLD in 0 # 1 2 3 4
# do
#     python vicreg_pretrain_experiment.py \
#         --name "${GROUP}_${FOLD}" \
#         --cluster "slurm" \
#         --slurm_gres "gpu:a40:1" \
#         --fold $FOLD \
#         --instance_norm $INSTANCE_NORM \
#         --use_batch_norm $USE_BATCH_NORM \
#         --cov_coeff 1.0 \
#         --linear_lr 0.005 \
#         --linear_epochs 20
# done


# # vicreg finetune core experiment
# INSTANCE_NORM=False
# USE_BATCH_NORM=False
# GROUP="vicreg_1024-300finetune_-4lr_8heads_64qk128v_8corebz_gn_loco"
# # GROUP="vicreg_finetune_1e-4backlr_1e-4headlr_8heads_transformer_gn_loco_batch10_newrunep"
# checkpoint_path_name="vicreg_pretrn_1024zdim_gn_300ep_loco"
# # checkpoint_path_name="vicreg_pretrn_2048zdim_gn_300ep_loco"
# # --group "${GROUP}" \
# for CENTER in "JH" "PCC" # "PMCC" "UVA" "CRCEO" 
# do
#     python core_finetune_experiment.py \
#         --name "${GROUP}_${CENTER}" \
#         --cluster "slurm" \
#         --slurm_gres "gpu:a40:1" \
#         --cohort_selection_config "loco" \
#         --leave_out $CENTER \
#         --instance_norm $INSTANCE_NORM \
#         --use_batch_norm $USE_BATCH_NORM \
#         --epochs 50 \
#         --core_batch_size 8 \
#         --nhead 8 \
#         --qk_dim 64 \
#         --v_dim 128 \
#         --checkpoint_path_name $checkpoint_path_name \
#         --backbone_lr 0.0001 \
#         --head_lr 0.0005 \
#         --batch_size 1 \
#         --dropout 0.0 \
#         # --prostate_mask_threshold -1
# done


# # vicreg finetune experiment
# INSTANCE_NORM=False
# USE_BATCH_NORM=False
# # GROUP="vicreg_1024-300finetune_1e-3lr_gn_loco"
# GROUP="vicreg_1024-300finetune_1e-4lr_avgprob_gn_crtd3ratio_loco"
# # checkpoint_path_name="vicreg_pretrn_1024zdim_gn_300ep_loco"
# checkpoint_path_name="vicreg_pretrn_1024zdim_gn_300ep_3ratio_loco"
# for CENTER in   "PMCC" "UVA" "CRCEO"  # "JH" "PCC"
# do
#     python finetune_experiment.py \
#         --name "${GROUP}_${CENTER}" \
#         --group "${GROUP}" \
#         --cluster "slurm" \
#         --slurm_gres "gpu:a40:1" \
#         --slurm_qos "deadline" \
#         --slurm_account "deadline" \
#         --cohort_selection_config "loco" \
#         --leave_out $CENTER \
#         --benign_to_cancer_ratio_train 3.0 \
#         --instance_norm $INSTANCE_NORM \
#         --use_batch_norm $USE_BATCH_NORM \
#         --epochs 50 \
#         --train_backbone True \
#         --checkpoint_path_name $checkpoint_path_name \
#         --backbone_lr 0.0001 \
#         --head_lr 0.0001
# done


# # Divemble experiment
# NUM_ENSEMBLES=5
# INSTANCE_NORM=False
# USE_BATCH_NORM=False
# GROUP="ensemble-shrd-fe_gn_${NUM_ENSEMBLES}mdls_3ratio_loco"
# # GROUP="Divemble-logt_gn_${NUM_ENSEMBLES}mdls_0.5var0.05cov_3ratio_loco"
# # GROUP="Divemble-shrd_gn_${NUM_ENSEMBLES}mdls_0var0.5cov_3ratio_loco"
# # GROUP="Divemble_gn_${NUM_ENSEMBLES}mdls_crctd_loco"
# for CENTER in  "PCC" "PMCC" "UVA" "CRCEO" # "JH" 
# do
#     python divemble_experiment.py \
#         --name "${GROUP}_${CENTER}" \
#         --group "${GROUP}" \
#         --cluster "slurm" \
#         --slurm_gres "gpu:a40:1" \
#         --num_ensembles $NUM_ENSEMBLES \
#         --cohort_selection_config "loco" \
#         --leave_out $CENTER \
#         --instance_norm $INSTANCE_NORM \
#         --use_batch_norm $USE_BATCH_NORM \
#         --epochs 50 \
#         --benign_to_cancer_ratio_train 3.0 \
#         --var_reg 0.0 \
#         --cov_reg 0.0
# done