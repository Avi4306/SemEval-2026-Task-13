#!/bin/bash
#SBATCH --job-name=adversarial_filtering_task_a
#SBATCH --partition=gpu
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/adversarial_filtering_task_a_%j.out
#SBATCH --error=logs/adversarial_filtering_task_a_%j.err

echo "Job started at $(date)"

# Use relative path to project root
cd "$(dirname "$0")/.."

# Ensure necessary directories exist
mkdir -p logs predictions models

# Run training using the active environment's python
export PYTHONPATH=.
python train/A3.py \
    --train_embeddings ./embeddings/modernbert-large/train_embeddings.npy \
    --test_embeddings ./embeddings/modernbert-large/test_embeddings.npy \
    --train_parquet ./data/train.parquet \
    --test_ids ./embeddings/modernbert-large/test_ids.npy \
    --test_sample_embeddings ./embeddings/modernbert-large/test_sample_embeddings.npy \
    --test_sample_parquet ./data/test_sample.parquet \
    --output_dir new_predictions \

echo "Job finished at $(date)"