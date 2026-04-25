#!/bin/bash
#SBATCH --job-name=A3_train
#SBATCH --partition=gpu
#SBATCH --gres=shard:32
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/A3_train_%j.out
#SBATCH --error=logs/A3_train_%j.err

echo "Job started at $(date)"

# Use relative path to project root
cd "$(dirname "$0")/.."

# Ensure necessary directories exist
mkdir -p logs predictions models

# Run training using the active environment's python
export PYTHONPATH=.
python train/A3_train.py \
    --model_name microsoft/unixcoder-base-nine \
    --train_path Task_A/train.parquet \
    --output_dir ./models/unixcoder-base-nine-embed \
    --num_epochs 1 \
    --batch_size 64 \
    --lr 2e-5 \
    --max_length 512

echo "Job finished at $(date)"