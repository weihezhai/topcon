#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 2      # 2 cores
#$ -l h_rt=12:0:0  # 24 hours runtime
#$ -l h_vmem=10G      # 10G RAM per core
#$ -m be
##$ -l gpu=1         # request GPUs (commented out)
## $ -l h=sbg4
#$ -l rocky
##$ -l cluster=andrena   
#$ -N iclr_2023_crawl_api_v1

# Load environment
source /data/home/mpx602/projects/py311/bin/activate

tar -czvf iclr2025.tar.gz /data/scratch/mpx602/topcon-1/conference_data/iclr_2025_data