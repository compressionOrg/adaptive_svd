#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gpus=1
#SBATCH -t 0-3:00:00 
#SBATCH --cpus-per-task=18
#SBATCH -o uniform.out

source /home/sliu01/anaconda3/etc/profile.d/conda.sh
source activate prune_llm
cd ..


python   main_singular_value_threshold.py  --rank_thresold 0.05 --Lamda 0.3 --prune_type weight_thresold_scale   
