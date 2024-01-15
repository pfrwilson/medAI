# # ensemble experiment
# NUM_ENSEMBLES=10
# INSTANCE_NORM=True
# USE_BATCH_NORM=True
# GROUP="ensemble_bn_${NUM_ENSEMBLES}mdls_inst-nrm_loco"

# for CENTER in "JH" "PCC" "PMCC" "UVA" "CRCEO"
# do
#     python ensemble_experiment.py \
#         --name "${GROUP}_${CENTER}" \
#         --group "${GROUP}" \
#         --cluster "slurm" \
#         --slurm_gres "gpu:a40:1" \
#         --num_ensembles $NUM_ENSEMBLES \
#         --cohort_selection_config "loco" \
#         --leave_out $CENTER \
#         --instance_norm $INSTANCE_NORM \
#         --use_batch_norm $USE_BATCH_NORM        
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
# USE_BATCH_NORM=False
# # GROUP="baseline_gn_loco"
# GROUP="sam_baseline_gn_e-4rho_loco"
# # GROUP="baseline_bn_inst-nrm_loco"

# for CENTER in "JH" #"PCC" "PMCC" "UVA" "CRCEO"
# do
#     python baseline_experiment.py \
#         --name "${GROUP}_${CENTER}" \
#         --group "${GROUP}" \
#         --cluster "slurm" \
#         --slurm_gres "gpu:rtx6000:1" \
#         --cohort_selection_config "loco" \
#         --leave_out $CENTER \
#         --instance_norm $INSTANCE_NORM \
#         --use_batch_norm $USE_BATCH_NORM \
#         --optimizer_config "sam" \
#         --rho 0.0001\
#         --lr 0.0001
# done



# # ttt experiment
# QUERY_PATCH=True
# SUPPORT_PATCHES=0
# GROUP="ttt_${SUPPORT_PATCHES}+1sprt_JT_0.1beta_loco"

# for CENTER in  "PMCC" "UVA" "CRCEO" #"JH" "PCC"
# do
#     python ttt_experiment.py \
#         --name "${GROUP}_${CENTER}" \
#         --group "${GROUP}" \
#         --cluster "slurm" \
#         --slurm_gres "gpu:a40:1" \
#         --cohort_selection_config "loco" \
#         --leave_out $CENTER \
#         --include_query_patch $QUERY_PATCH \
#         --num_support_patches $SUPPORT_PATCHES \
#         --joint_training True \
#         --adaptation_steps 1 \
#         --adaptation_lr 0.0001 \
#         --beta_byol 0.1
# done 


# # mt3 experiment
# QUERY_PATCH=True
# SUPPORT_PATCHES=0
# GROUP="mt3_${SUPPORT_PATCHES}+1sprt_0.1beta_loco"

# for CENTER in  "PMCC" "UVA" "CRCEO" # "JH" "PCC"  
# do
#     python mt3_experiment.py \
#         --name "${GROUP}_${CENTER}" \
#         --group "${GROUP}" \
#         --cluster "slurm" \
#         --slurm_gres "gpu:a40:1" \
#         --cohort_selection_config "loco" \
#         --leave_out $CENTER \
#         --include_query_patch $QUERY_PATCH \
#         --num_support_patches $SUPPORT_PATCHES \
#         --inner_steps 1 \
#         --inner_lr 0.001 \
#         --beta_byol 0.1
# done 


# vicreg pretrain experiment
INSTANCE_NORM=False
USE_BATCH_NORM=False
GROUP="vicreg_pretrn_5e-3-10linprob_100ep_gn_loco"
# --group "${GROUP}" \
for CENTER in "JH" # "PCC" "PMCC" "UVA" "CRCEO"
do
    python vicreg_pretrain_experiment.py \
        --name "${GROUP}_${CENTER}" \
        --cluster "slurm" \
        --slurm_gres "gpu:a40:1" \
        --cohort_selection_config "loco" \
        --leave_out $CENTER \
        --instance_norm $INSTANCE_NORM \
        --use_batch_norm $USE_BATCH_NORM \
        --epochs 100 \
        --cov_coeff 1.0 \
        --linear_lr 0.005 \
        --linear_epochs 10
done


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