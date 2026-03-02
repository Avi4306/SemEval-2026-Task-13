#!/bin/bash
#SBATCH --job-name=xgb_task_a
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/xgb_task_a_%j.out
#SBATCH --error=logs/xgb_task_a_%j.err

echo "Job started at $(date)"

# Use relative path to project root
cd "$(dirname "$0")/.."

# Ensure necessary directories exist
mkdir -p logs predictions models

# Run training using the active environment's python
export PYTHONPATH=.
python train/A1.py

echo "Job finished at $(date)"