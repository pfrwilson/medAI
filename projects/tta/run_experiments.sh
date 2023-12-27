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


# baseline experiment
INSTANCE_NORM=True
USE_BATCH_NORM=True
GROUP="baseline_bn_inst-nrm_loco"

for CENTER in "JH" "PCC" "PMCC" "UVA" "CRCEO"
do
    python baseline_experiment.py \
        --name "${GROUP}_${CENTER}" \
        --group "${GROUP}" \
        --cluster "slurm" \
        --slurm_gres "gpu:a40:1" \
        --cohort_selection_config "loco" \
        --leave_out $CENTER \
        --instance_norm $INSTANCE_NORM \
        --use_batch_norm $USE_BATCH_NORM        
done 