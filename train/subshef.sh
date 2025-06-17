#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=82G
#SBATCH --cpus-per-task=12
#SBATCH --mail-user=wzhai2@sheffield.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=iclr_2025_llm_paper_qwen3_8B_lora
#SBATCH --output=./logs/iclr_2025_llm_paper_qwen3_8B_lora.%j.out
#SBATCH --time=0-12:00:00

# Load necessary modules
module load Anaconda3/2024.02-1
module load cuDNN/8.9.2.26-CUDA-12.1.1

# Activate your conda environment
source activate etu

# Run the preprocessing script using the runner
# You can modify these parameters as needed
python train_shef.py 