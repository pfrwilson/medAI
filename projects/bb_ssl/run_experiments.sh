# vicreg pretrain experiment
GROUP="vicreg_pretrn_1e-2lr_5e-3-15linprob_cifar10_newtrnsfrm_bn_1000ep"
# --group "${GROUP}" \
for SPLIT_SEED in 0
do
    python vicreg_pretrain_experiment.py \
        --name "${GROUP}_${SPLIT_SEED}" \
        --cluster "slurm" \
        --slurm_gres "gpu:a40:1" \
        --split_seed $SPLIT_SEED \
        --use_batch_norm $USE_BATCH_NORM \
        --epochs 1000 \
        --lr 0.01 \
        --cov_coeff 1.0 \
        --linear_lr 0.005 \
        --linear_epochs 15
done


