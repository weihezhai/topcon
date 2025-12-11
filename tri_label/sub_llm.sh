#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=12
#SBATCH --mail-user=wz946@bath.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=4b_vl_orig_llm_ba
#SBATCH --output=./logs/4b_vl_orig_llm_ba.%j.out
#SBATCH --time=0-11:00:00

module load cuDNN/8.7.0.84-CUDA-11.8.0
source /mnt/parscratch/users/lip22fh/miniconda3/etc/profile.d/conda.sh
conda activate pap

python tri_4B_12000_two_a100_llm_vl.py --detailed_eval