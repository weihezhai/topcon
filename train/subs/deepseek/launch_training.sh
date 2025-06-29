#!/bin/bash

# Set the specific GPU IDs you want to use
export CUDA_VISIBLE_DEVICES="0,1,2,3"

# Launch training with accelerate
accelerate launch \
    --config_file /mnt/parscratch/users/acr24wz/topcon/train/subs/deepseek/accelerate_config.yaml \
    /mnt/parscratch/users/acr24wz/topcon/train/subs/deepseek/train_shef_abs_intro.py \
    --model_name "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B" \
    --data_folder "/mnt/parscratch/users/acr24wz/src/iclr/data/scratch/mpx602/topcon-1/conference_data/iclr_2025_data/filtered_llm_papers/llm_papers_text/" \
    --labels_file "/mnt/parscratch/users/acr24wz/topcon/train/llm_paper/label_simple.json" \
    --output_dir "/mnt/parscratch/users/acr24wz/etu/topcon/DeepSeek-R1-0528-Qwen3-8B/finetuned_model" \
    --max_length 10000
