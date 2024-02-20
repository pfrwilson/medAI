#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a40:1
#SBATCH --job-name=submitit
#SBATCH --mem=16GB
#SBATCH --nodes=1
#SBATCH --qos=deadline
#SBATCH --account=deadline
#SBATCH --time=240
#SBATCH --output=/fs01/home/abbasgln/codes/medAI/projects/tta/notebooks/logs/%J.out
#SBATCH --error=/fs01/home/abbasgln/codes/medAI/projects/tta/notebooks/logs/%J.err


# # setup
# module load pytorch2.1-cuda11.8-python3.10
# export PYTHONPATH=$PYTHONPATH:/h/pwilson/projects/medAI

# python offline_ensemble_pseudo.py
# python offline_ensemble_memo.py
python offline_memo.py
# python offline_divemble_pseudo.py