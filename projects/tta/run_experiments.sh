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
#         --num_ensembles $NUM_ENSEMBLES \
#         --cohort_selection_config "loco" \
#         --leave_out $CENTER \
#         --instance_norm $INSTANCE_NORM \
#         --use_batch_norm $USE_BATCH_NORM        
# done 


# sngp experiment
INSTANCE_NORM=True
GROUP="sngp_inst-nrm_loco"

for CENTER in "JH" "PCC" "PMCC" "UVA" "CRCEO"
do
    python sngp_experiment.py \
        --name "${GROUP}_${CENTER}" \
        --group "${GROUP}" \
        --cluster "slurm" \
        --cohort_selection_config "loco" \
        --leave_out $CENTER \
        --instance_norm $INSTANCE_NORM                
done 