# vicreg pretrain experiment
USE_BATCH_NORM=True
GROUP="vicreg_pretrn_1e-3lr_1e-2sgdlinprob_cifar10_newtrnsfrm_bn_1000ep_8192zdim"
# GROUP="vicreg_pretrn_test2"
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
        --lr 0.001 \
        --proj_output_dim 8192 \
        --cov_coeff 1.0 \
        --linear_lr 0.01 \
        --linear_epochs 15
done


