#!/bin/bash

# Interactive script for testing continued pretraining without SLURM

# Create logs directory if it doesn't exist
mkdir -p ./logs

# Set environment variables for better performance
export OMP_NUM_THREADS=12
export TOKENIZERS_PARALLELISM=false
# export CUDA_VISIBLE_DEVICES=0,1  # Adjust based on available GPUs

# Log output to file with timestamp
LOG_FILE="./logs/cpt_interactive_$(date +%Y%m%d_%H%M%S).log"

echo "Starting continued pretraining..."
echo "Log file: $LOG_FILE"

# Run continued pretraining with default values
python cpt_gpt.py \
    --model_name "Qwen/Qwen3-4B" \
    --data_path "/mnt/parscratch/users/acr24wz/src/iclr/mineru/llm/" \
    --labels_file "/mnt/parscratch/users/acr24wz/topcon/train/label_simple.json" \
    --output_dir "/mnt/parscratch/users/acr24wz/etu/topcon/qwen3_4B/cpt_model/llm" \
    --block_size 2048 \
    --max_length 10000 \
    --epochs 3 \
    --lr 8e-6 \
    --warmup_ratio 0.1 \
    --batch_size 1 \
    --grad_accum 8 \
    --save_steps 300 \
    --logging_steps 50 \
    --bf16 \
    --num_proc 4 \
    --eval_holdout 0.1 \
    --gpu_ids 1 3 \
    --flash_attn \
    --use_dataset_builder \
    --use_cache \
    --cache_dir "/mnt/parscratch/users/acr24wz/etu/topcon/processed_dataset/cpt/llm" \
    --model_cache_dir "/mnt/parscratch/users/acr24wz/etu/topcon/qwen3_4B" \
    2>&1 | tee "$LOG_FILE"

echo "Continued pretraining completed! Check log at: $LOG_FILE"
