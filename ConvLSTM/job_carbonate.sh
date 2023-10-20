#!/bin/bash -l
#SBATCH -N 1
#SBATCH -t 48:00:00
#SBATCH -J nwp_convlstm
#SBATCH -p gpu --gpus-per-node v100:1
#SBATCH -A r00043 
#module load PrgEnv-gnu
module load python/gpu
cd /N/u/ckieu/BigRed200/model/deep-learning/ConvLSTM
python nwp_convlstm_p1.py 2020
python nwp_convlstm_p1.py 2021

