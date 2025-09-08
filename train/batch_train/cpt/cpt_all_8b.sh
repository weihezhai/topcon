#!/bin/bash
#SBATCH --partition=gpu-h100
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=12
#SBATCH --mail-user=wzhai2@sheffield.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=8b_cpt_all
#SBATCH --output=./logs/8b/all_8b_cpt.%j.out
#SBATCH --time=2-00:00:00

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

# python main.py \
#   --use_hier \
#   --hier_mode sentence --max_sentences 160 \
#   --k_soft_tokens 4 \
#   --stage2_unfreeze_top 8 --stage2_epochs 2 --stage2_lr 1e-4 \
#   --model_name Qwen/Qwen3-8B \
#   --output_dir /path/to/out_hier_stage2 \
#   --gpu_ids 0 1 2

# tensorboard --port=6006 --bind_all --logdir=./logs/

python cpt_gpt_8B_all.py --batch_size 1 --flash_attn 