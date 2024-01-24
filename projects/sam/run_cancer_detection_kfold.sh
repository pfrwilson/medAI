GROUP_NAME=$(python -c "from medAI.utils import generate_experiment_name; print(generate_experiment_name())")
echo "GROUP_NAME: $GROUP_NAME"
FOLD=0

for FOLD in {0..4}
do 
echo "FOLD: $FOLD"
python medsam_cancer_detection_v2.py \
    --cluster slurm \
    --slurm_gres gpu:a40:1 \
    --slurm_qos m3 \
    --timeout_min 240 \
    --use_augmentation \
    --benign_cancer_ratio_for_training 3 \
    --fold $FOLD \
    --n_folds 4 \
    --epochs 30 \
    --group "${GROUP_NAME}" \
    --name "fold${FOLD}" \
    --freeze_backbone
done