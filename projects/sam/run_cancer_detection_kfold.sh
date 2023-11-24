GROUP_NAME=$(python -c "from medAI.utils import generate_experiment_name; print(generate_experiment_name())")
echo "GROUP_NAME: $GROUP_NAME"
FOLD=0

for FOLD in 0 1 2 3 4
do 
echo "FOLD: $FOLD"
python medsam_cancer_detection_v2.py \
    --cluster submitit_auto \
    --timeout_min 480 \
    --use_augmentation \
    --benign_cancer_ratio_for_training 3 \
    --fold $FOLD \
    --epochs 30 \
    --group "${GROUP_NAME}" \
    --name "fold${FOLD}" 
done