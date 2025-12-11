#!/bin/bash
#SBATCH --job-name=qwen_img_cap
#SBATCH --output=logs/%x_%A_%a.out  # %A is job ID, %a is array index
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=cpu             # Targeting the CPU Partition (768 nodes)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4           # 4 CPUs for image encoding/overhead
#SBATCH --mem=8G                    # 8GB RAM should be plenty
#SBATCH --time=48:00:00             # Adjust based on dataset size
#SBATCH --array=0-3                 # RUNS 4 PARALLEL JOBS (indices 0, 1, 2, 3)

# 1. Load necessary modules (Adjust based on your HPC environment)
module load Python/3.12.3-GCCcore-13.3.0  
# source /path/to/your/virtualenv/bin/activate

# 2. Export API Key (If not set globally)
export DASHSCOPE_API_KEY="sk-5e081662e553472788958c33a522dde2"

# 3. Define paths
SCRIPT_PATH="/ceph/hpc/home/euweihez/topcon/img_description/evaluate_image_api_sharded_cv.py"
DATA_ROOT="/ceph/hpc/home/euweihez/topcon/d2025d08-005-users/data_src/balanced/balanced_cv/cv/balanced_cv"
OUTPUT_BASE="/ceph/hpc/home/euweihez/topcon/d2025d08-005-users/img_description/cv/image_descriptions.json"

# 4. Create log dir if not exists
mkdir -p logs

# 5. Run the Python script
# SLURM_ARRAY_TASK_ID will be 0, 1, 2, or 3
# SLURM_ARRAY_TASK_COUNT will be 4 (if you set array=0-3, count is actually manual, so we hardcode 4)

echo "Starting Task ID: $SLURM_ARRAY_TASK_ID"

uv run $SCRIPT_PATH \
    --data_root "$DATA_ROOT" \
    --output_file "$OUTPUT_BASE" \
    --model_name "qwen3-vl-8b-instruct" \
    --shard_id $SLURM_ARRAY_TASK_ID \
    --num_shards 4

echo "Task $SLURM_ARRAY_TASK_ID finished"