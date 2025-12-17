#!/bin/bash
#SBATCH --job-name=14B_llm_lora
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --mail-user=weihezhai@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

module load Python/3.12.3-GCCcore-13.3.0

uv run 14B_12000_all_vl_wl_lora.py --detailed_eval \
    --resume_from_checkpoint "/ceph/hpc/home/euweihez/topcon/d2025d08-005-users/models/qwen3_14b/all/all_fail/checkpoint-900"