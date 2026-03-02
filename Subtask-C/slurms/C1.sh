#!/bin/bash
#SBATCH --job-name=modernbert_large_task_c
#SBATCH --partition=gpu
#SBATCH --gres=shard:20
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/modernbert_large_task_c_%j.out
#SBATCH --error=logs/modernbert_large_task_c_%j.err

echo "Job started at $(date)"

# Use relative path to move to the project root directory
cd "$(dirname "$0")/.."

# Create necessary directories if they don't exist
mkdir -p logs predictions models

# Use the environment's python
export PYTHONPATH=.
python train/C1.py \
    --model_path answerdotai/ModernBERT-large \
    --max_length 512 \
    --num_epochs 3 \
    --batch_size 64 \
    --learning_rate 5e-5 \
    --weight_decay 0.001 \
    --warmup_ratio 0.1 \
    --gradient_accumulation_steps 2 \
    --bf16  \
    --gradient_checkpointing \
    --output_dir ./models/modernbert-large \
    --logging_steps 500 \
    --do_train \
    --do_eval \
    --do_predict \
    --submission_file ./predictions/modernbert_large_512.csv

echo "Job finished at $(date)"