#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --mem-per-cpu=2G
#SBATCH --time=6:00:00
#SBATCH --job-name=reeval
#SBATCH --partition=short
#SBATCH --clusters=all

module load Anaconda3
source activate /data/PLACEHOLDER_PATH/PLACEHOLDER/conda_envs/c_gan
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/PLACEHOLDER_PATH/PLACEHOLDER/conda_envs/c_gan/lib
#source activate $DATA/conda_envs/c_gan
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$DATA/conda_envs/c_gan/lib

export PYTHONPATH=$PYTHONPATH:gather_results

use_case="url"
model="tablegan"
versions="unconstrained random corr kde"

for version in $versions ;
do
  CUDA_VISIBLE_DEVICES=-1 python gather_results/reeval_final.py ${use_case} ${model} ${version}
done
