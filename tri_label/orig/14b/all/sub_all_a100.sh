#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:3
#SBATCH --mem=140G
#SBATCH --cpus-per-task=12
#SBATCH --mail-user=weihezhai@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=wl_14b_vl_orig_all
#SBATCH --output=./logs/wl_14b_vl_orig_all.%j.out
#SBATCH --time=0-30:00:00

module load cuDNN/8.7.0.84-CUDA-11.8.0
source /mnt/parscratch/users/lip22fh/miniconda3/etc/profile.d/conda.sh
conda activate pap

python 14B_12000_all_vl_wl.py --detailed_eval