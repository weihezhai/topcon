#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=12
#SBATCH --mail-user=wzhai2@sheffield.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=cpted_all_8b_balanced_eval
#SBATCH --output=./logs/cpted_all_8b_balanced_eval.%j.out
#SBATCH --time=8:00:00

module load Python/3.10.8-GCCcore-12.2.0
module load cuDNN/8.9.2.26-CUDA-12.1.1
source /mnt/parscratch/users/acr24wz/envs/py310/bin/activate

mkdir -p logs predictions

# Use job ID in output filename
PRED_JSON=./predictions/eval_predictions_${SLURM_JOB_ID}.json

python 8B_8000_two_a100_all_prediction.py \
  --eval \
  --detailed_eval \
  --predictions_file "${PRED_JSON}"