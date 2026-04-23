#!/bin/bash
#SBATCH --job-name=train_C2
#SBATCH --partition=gpu
#SBATCH --gres=shard:32
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/train_C2_%j.out
#SBATCH --error=logs/train_C2_%j.err

# -------------------------
echo "Job started at $(date)"

# Use relative path to move to the project root directory
cd "$(dirname "$0")/.."

# Create necessary directories if they don't exist
mkdir -p logs models

# -------------------------
# Run training
# -------------------------
python train/C2_train.py \
    --model_name unixcoder \
    --model_path microsoft/unixcoder-base \
    --train_size 900000 \
    --val_size 200000 \
    --max_length 512 \
    --num_epochs 3 \
    --batch_size 64 \
    --learning_rate 5e-5 \
    --weight_decay 0.001 \
    --warmup_ratio 0.1 \
    --gradient_accumulation_steps 2 \
    --bf16  \
    --gradient_checkpointing \
    --output_dir ./models/unixcoder-base \
    --logging_steps 100 \
    --use_class_weights

echo "Job finished at $(date)"