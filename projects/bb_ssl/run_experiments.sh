# vicreg pretrain experiment
# GROUP="vicreg_pretrn_1e-2lr_1e-2-15linprob_cifar10_newtrnsfrm_bn_1000ep"
GROUP="vicreg_pretrn_test"
# --group "${GROUP}" \
for SPLIT_SEED in 0
do
    python vicreg_pretrain_experiment.py \
        --name "${GROUP}_${SPLIT_SEED}" \
        --cluster "slurm" \
        --slurm_gres "gpu:a40:1" \
        --split_seed $SPLIT_SEED \
        --use_batch_norm $USE_BATCH_NORM \
        --timeout_min 4 \
        --epochs 1000 \
        --lr 0.01 \
        --cov_coeff 1.0 \
        --linear_lr 0.01 \
        --linear_epochs 15 
done


