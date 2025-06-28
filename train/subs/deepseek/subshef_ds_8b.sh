#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:3
#SBATCH --mem=100G
#SBATCH --cpus-per-task=12
#SBATCH --mail-user=wzhai2@sheffield.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=iclr_2025_llm_paper_DeepSeek-R1-0528-Qwen3-8B
#SBATCH --output=./logs/iclr_2025_llm_paper_DeepSeek-R1-0528-Qwen3-8B.%j.out
#SBATCH --time=0-12:00:00

# Load necessary modules
# module load Anaconda3/2024.02-1
# module load GCC/11.2.0
# module load cuDNN/8.7.0.84-CUDA-11.8.0
module load Python/3.10.8-GCCcore-12.2.0
module load cuDNN/8.9.2.26-CUDA-12.1.1
# module load CUDA/12.4.0

# Activate your conda environment
# source activate etu
source /mnt/parscratch/users/acr24wz/envs/py310/bin/activate

# Run the preprocessing script using the runner
# You can modify these parameters as needed
python train_shef_abs_intro_2gpu.py --detailed_eval