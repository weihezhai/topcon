#!/bin/bash
#SBATCH --job-name=14B_llm_lora
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --mail-user=weihezhai@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

module load Python/3.12.3-GCCcore-13.3.0

# Slurm typically sets CUDA_VISIBLE_DEVICES for you; torchrun will spawn 1 proc/GPU.
NPROC_PER_NODE="${SLURM_GPUS_ON_NODE:-4}"

# Give each job a unique output_dir to avoid collisions across runs.
RUN_TAG="$(date +%Y%m%d_%H%M%S)_job${SLURM_JOB_ID}"
OUTDIR="/ceph/hpc/home/euweihez/topcon/d2025d08-005-users/models/qwen3_14b/all/${RUN_TAG}"

# Recommended NCCL settings on single-node multi-GPU (tweak if your cluster requires).
export NCCL_ASYNC_ERROR_HANDLING=1

uv run torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  14B_12000_all_vl_wl_lora_ds.py \
  --use_deepspeed \
  --deepspeed_stage 3 \
  --output_dir "${OUTDIR}" \
  --detailed_eval