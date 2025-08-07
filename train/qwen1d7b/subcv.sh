#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=90G
#SBATCH --cpus-per-task=12
#SBATCH --mail-user=wzhai2@sheffield.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=cv_qwen1.7b
#SBATCH --output=./logs/cv_qwen1.7b.%j.out
#SBATCH --time=0-2:00:00

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
python train_shef_abs_intro_single_gpu_detailed_cv.py --detailed_eval