#!/bin/bash

# Example usage script for continued pretraining

# Set paths - adjust these to your actual paths
DATA_FOLDER="/path/to/your/data/folder"
LABELS_FILE="/path/to/your/labels.json"
STATISTICS_FILE="/path/to/your/statistics.json"  # Optional

# Run continued pretraining
python continued_pretraining.py \
    --data_folder "$DATA_FOLDER" \
    --labels_file "$LABELS_FILE" \
    --statistics_file "$STATISTICS_FILE" \
    --output_dir "./qwen_cpt_output" \
    --batch_size 4 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --num_epochs 3 \
    --max_length 2048 \
    --warmup_ratio 0.1 \
    --save_steps 500 \
    --logging_steps 50 \
    --bf16 \
    --gradient_checkpointing
