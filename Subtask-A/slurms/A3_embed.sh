#!/bin/bash
#SBATCH --job-name=extract_embeddings_task_a
#SBATCH --partition=gpu
#SBATCH --gres=shard:10
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/extract_embeddings_task_a_%j.out
#SBATCH --error=logs/extract_embeddings_task_a_%j.err

echo "Job started at $(date)"

# Use relative path to project root
cd "$(dirname "$0")/.."

# Ensure necessary directories exist
mkdir -p logs predictions models

# Run training using the active environment's python
export PYTHONPATH=.
python train/A3_embed.py \
    --model_name_or_path answerdotai/modernbert-large \
    --train_path data/train.parquet \
    --test_path data/test.parquet \
    --test_sample_path data/test_sample.parquet \
    --output_dir ./embeddings/modernbert-large \
    --batch_size 64 \
    --max_length 512 \
    --use_pooling

echo "Job finished at $(date)"