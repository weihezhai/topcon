#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=90G
#SBATCH --cpus-per-task=12
#SBATCH --mail-user=wzhai2@sheffield.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=4b_non_cpt_all_imbalanced
#SBATCH --output=./logs/4b_non_cpt_all_imbalanced.%j.out
#SBATCH --time=01-10:00:00

# Load necessary modules
# module load Anaconda3/2024.02-1
# module load GCC/11.2.0
# module load cuDNN/8.7.0.84-CUDA-11.8.0
module load Python/3.10.8-GCCcore-12.2.0
module load cuDNN/8.9.2.26-CUDA-12.1.1
source /mnt/parscratch/users/acr24wz/envs/py310/bin/activate
# module load CUDA/12.4.0

# Activate your conda environment
# source activate etu

# Run the preprocessing script using the runner
# You can modify these parameters as needed
python 4B_8000_two_a100_all.py --detailed_eval