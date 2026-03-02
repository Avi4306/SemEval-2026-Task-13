#!/bin/bash
#SBATCH --job-name=xgb_stacking_task_c
#SBATCH --partition=gpu
#SBATCH --gres=shard:10
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/xgb_stacking_task_c_%j.out
#SBATCH --error=logs/xgb_stacking_task_c_%j.err

echo "Job started at $(date)"

# Use relative path to move to the project root directory
cd "$(dirname "$0")/.."

# Create necessary directories if they don't exist
mkdir -p logs predictions models

# Use the environment's python
export PYTHONPATH=.
python train/C2_stacking.py \
    --model_names unixcoder \
    --logits_dir ./logits \
    --output_dir ./ensemble_results \
    --train \
    --predict \
    --submission_name stacking_submission.csv

echo "Job finished at $(date)"