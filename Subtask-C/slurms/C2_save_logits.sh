#!/bin/bash
#SBATCH --job-name=save_logits_task_c
#SBATCH --partition=gpu
#SBATCH --gres=shard:32
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/save_logits_task_c_%j.out
#SBATCH --error=logs/save_logits_task_c_%j.err

echo "Job started at $(date)"

# Use relative path to move to the project root directory
cd "$(dirname "$0")/.."

# Create necessary directories if they don't exist
mkdir -p logs predictions models

# Use the environment's python
export PYTHONPATH=.
python train/C2_save_logits.py \
    --model_paths ./models/unixcoder-base-taskC/final\
    --model_names unixcoder \
    --val_data Task_C/validation.parquet \
    --test_data Task_C/test.parquet \
    --output_dir ./logits \
    --batch_size 32

echo "Job finished at $(date)"