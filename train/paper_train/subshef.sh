#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:2
#SBATCH --mem=100G
#SBATCH --cpus-per-task=12
#SBATCH --mail-user=YOUR_EMAIL@sheffield.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=llm_paper_training
#SBATCH --output=./logs/llm_paper_training.%j.out
#SBATCH --time=0-12:00:00

# Load necessary modules 
module load Python/3.10.8-GCCcore-12.2.0
module load cuDNN/8.9.2.26-CUDA-12.1.1

# Activate your conda/virtual environment
# Option 1: Conda environment
# source activate YOUR_ENV_NAME
# Option 2: Virtual environment
# source /path/to/your/venv/bin/activate

# Set environment variables (customize these paths for your setup)
export DATA_FOLDER="/path/to/your/training/data"
export LABELS_FILE="/path/to/your/labels.json"
export OUTPUT_DIR="/path/to/your/output/models"
export BASE_MODEL_CACHE="/path/to/your/model/cache"
export PROCESSED_DATASET_CACHE="/path/to/your/dataset/cache"

# Run the training script
python train_v1.py \
    --data_folder $DATA_FOLDER \
    --labels_file $LABELS_FILE \
    --output_dir $OUTPUT_DIR \
    --base_model_cache $BASE_MODEL_CACHE \
    --processed_dataset_cache $PROCESSED_DATASET_CACHE \
    --detailed_eval