#!/bin/bash
#SBATCH --job-name=crawl
#SBATCH --output=./crawl/logs/%x_%j.out
#SBATCH --error=./crawl/logs/%x_%j.err
#SBATCH --cpus-per-task=16
## SBATCH --mem=80G
#SBATCH --time=12:00:00

# Create log directory if it doesn't exist
mkdir -p ./crawl/logs

# Load Python module
module load Python/3.12.3-GCCcore-13.3.0

# Print job information
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 100GB"
echo ""

# Run the crawler
uv run spider_with_pdfs_apiv2.py --conference iclr --venue 2026 --limit 0

# Print job completion
echo ""
echo "Job finished at: $(date)"