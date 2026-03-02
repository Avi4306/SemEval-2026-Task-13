#!/bin/bash
#SBATCH --job-name=modernbert_task_b
#SBATCH --partition=gpu
#SBATCH --gres=shard:32
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/modernbert_task_b_%j.out
#SBATCH --error=logs/modernbert_task_b_%j.err

echo "Job started at $(date)"

# Use relative path to project root
cd "$(dirname "$0")/.."

# Ensure necessary directories exist
mkdir -p logs predictions models

# Run training using the active environment's python
export PYTHONPATH=.
python train/B1.py \
    --model_name modernbert-large \
    --model_path answerdotai/ModernBERT-large \
    --max_length 512 \
    --num_epochs 5 \
    --batch_size 16 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --weight_decay 0.001 \
    --warmup_ratio 0.1 \
    --bf16 \
    --tf32 \
    --output_dir ./models/modernbert-large-task-b \
    --logging_steps 500 \
    --do_train \
    --do_eval \
    --do_predict \
    --submission_file ./predictions/modernbert_large_512_task_b.csv

echo "Job finished at $(date)"