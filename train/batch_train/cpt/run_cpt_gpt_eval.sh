#!/bin/bash

# Script for evaluating a fine-tuned model

# Create logs directory if it doesn't exist
mkdir -p ./logs

# Set environment variables for better performance
export OMP_NUM_THREADS=12
export TOKENIZERS_PARALLELISM=false

# Log output to file with timestamp
LOG_FILE="./logs/cpt_eval_$(date +%Y%m%d_%H%M%S).log"

echo "Starting evaluation of fine-tuned model..."
echo "Log file: $LOG_FILE"

# Parse command line arguments for model size (default: 4B)
MODEL_SIZE="${1:-4B}"
if [ "$MODEL_SIZE" == "8B" ]; then
    MODEL_NAME="Qwen/Qwen3-8B"
    OUTPUT_BASE="/mnt/parscratch/users/acr24wz/etu/topcon/qwen3_8B/cpt_model"
else
    MODEL_NAME="Qwen/Qwen3-4B"
    OUTPUT_BASE="/mnt/parscratch/users/acr24wz/etu/topcon/qwen3_4B/cpt_model"
fi

# Find the most recent training output directory (without _eval suffix)
LATEST_MODEL=$(ls -dt ${OUTPUT_BASE}_* 2>/dev/null | grep -v "_eval_" | head -n1)

if [ -z "$LATEST_MODEL" ]; then
    echo "Error: No trained model found at ${OUTPUT_BASE}_*"
    echo "Please train the model first before running evaluation."
    exit 1
fi

echo "Using model from: $LATEST_MODEL"

# Run evaluation on fine-tuned model
python cpt_gpt.py \
    --model_name "$MODEL_NAME" \
    --data_path "/mnt/parscratch/users/acr24wz/src/iclr/mineru/llm/" \
    --labels_file "/mnt/parscratch/users/acr24wz/topcon/train/label_simple.json" \
    --output_dir "$LATEST_MODEL" \
    --block_size 2048 \
    --max_length 10000 \
    --batch_size 2 \
    --bf16 \
    --num_proc 4 \
    --eval_holdout 0.1 \
    --gpu_ids 0 1 \
    --flash_attn \
    --use_dataset_builder \
    --use_cache \
    --eval \
    2>&1 | tee "$LOG_FILE"

echo "Evaluation completed! Check log at: $LOG_FILE"
