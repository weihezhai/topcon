#!/bin/bash
#SBATCH --partition=gpu-h100-nvl
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=12
#SBATCH --mail-user=weihezhai@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=wl_8b_vl_orig_all
#SBATCH --output=./logs/wl_8b_vl_orig_all.%j.out
#SBATCH --time=0-24:00:00

module load cuDNN/8.7.0.84-CUDA-11.8.0
source /mnt/parscratch/users/lip22fh/miniconda3/etc/profile.d/conda.sh
conda activate pap

python 8B_12000_all_vl_wl.py --detailed_eval