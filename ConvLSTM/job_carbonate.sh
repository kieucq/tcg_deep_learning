#!/bin/bash -l
#SBATCH -N 1
#SBATCH -t 48:00:00
#SBATCH -J nwp_convlstm
#SBATCH -p gpu --gpus-per-node v100:1
#SBATCH -A r00043 
#SBATCH --mem=256G
#module load PrgEnv-gnu
module load python/gpu
cd /N/u/ckieu/BigRed200/model/deep-learning/ConvLSTM
#python nwp_convlstm_p1.py 2014
#python nwp_convlstm_p1.py 2015
#python nwp_convlstm_p1.py 2016
#python nwp_convlstm_p1.py 2017
#python nwp_convlstm_p1.py 2010
#python nwp_convlstm_p1.py 2011
#python nwp_convlstm_p1.py 2012
#python nwp_convlstm_p1.py 2013
python nwp_convlstm_p1.py 2008
python nwp_convlstm_p1.py 2009

