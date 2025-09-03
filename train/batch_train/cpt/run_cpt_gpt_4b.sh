#!/bin/bash

# SLURM submission script for continued pretraining with Qwen3-8B model
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:2
#SBATCH --mem=100G
#SBATCH --cpus-per-task=16
#SBATCH --mail-user=your_email@sheffield.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=cpt_qwen3_4b
#SBATCH --output=./logs/cpt_qwen3_4b.%j.out
#SBATCH --time=1-00:00:00

# Create logs directory if it doesn't exist
mkdir -p ./logs

# Load necessary modules
module load Python/3.10.8-GCCcore-12.2.0
module load cuDNN/8.9.2.26-CUDA-12.1.1

# Activate your Python environment
source /mnt/parscratch/users/acr24wz/envs/py310/bin/activate

# Set environment variables for better performance
export OMP_NUM_THREADS=16
export TOKENIZERS_PARALLELISM=false

# Run continued pretraining with 4B model
python cpt_gpt.py \
    --model_name "Qwen/Qwen3-4B" \
    --data_path "/mnt/parscratch/users/acr24wz/src/iclr/mineru/llm/" \
    --labels_file "/mnt/parscratch/users/acr24wz/topcon/train/label_simple.json" \
    --statistics_file "/mnt/parscratch/users/acr24wz/src/iclr/data/scratch/mpx602/topcon-1/conference_data/iclr_2025_data/statistics_per_paper.json" \
    --output_dir "/mnt/parscratch/users/acr24wz/etu/topcon/qwen3_4B/cpt_model" \
    --block_size 2048 \
    --max_length 10000 \
    --epochs 2 \
    --lr 8e-6 \
    --warmup_ratio 0.05 \
    --batch_size 1 \
    --grad_accum 16 \
    --save_steps 150 \
    --logging_steps 25 \
    --bf16 \
    --num_proc 4 \
    --eval_holdout 0.1 \
    --gpu_ids 0 1 \
    --use_dataset_builder \
    --use_cache \
    --cache_dir "/mnt/parscratch/users/acr24wz/etu/topcon/processed_dataset" \
    --model_cache_dir "/mnt/parscratch/users/acr24wz/etu/topcon/models" 
    # --flash_attn

echo "Continued pretraining with 4B model completed!"
