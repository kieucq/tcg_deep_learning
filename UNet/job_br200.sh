#!/bin/bash -l
#SBATCH -N 1
#SBATCH -t 18:00:00
#SBATCH -J segment_gen
#SBATCH -p gpu --gpus 2
#SBATCH -A r00043 
module load PrgEnv-gnu
module load python/gpu
cd /N/u/ckieu/Carbonate/model/deep-learning/UNet/
sed -i 's/0h/6h/' tcg_segment_p0.py 
python tcg_segment_p0.py

