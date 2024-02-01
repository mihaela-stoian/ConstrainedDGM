#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --mem-per-cpu=2G
#SBATCH --time=12:00:00
#SBATCH --job-name=wgan-lcld
#SBATCH --partition=short
#SBATCH --clusters=all

module load Anaconda3
source activate /data/PLACEHOLDER_PATH/PLACEHOLDER/conda_envs/c_gan
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/PLACEHOLDER_PATH/PLACEHOLDER/conda_envs/c_gan/lib
#source activate $DATA/conda_envs/c_gan
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$DATA/conda_envs/c_gan/lib

use_case="lcld"
wandbp="wandb_constrained_wgan_${use_case}_fixed_scaler"
eps=10

rep=10
lr=0.00005
pac=1
bs=64

seeds="3 5 9 21"
for seed in $seeds ;
do
  echo "**********************************************"
  CUDA_VISIBLE_DEVICES=-1 python main_wgan.py ${use_case} --wandb_project=$wandbp --seed=$seed --epochs=$eps --disc_repeats=${rep} --d_lr=${lr} --g_lr=${lr} --pac=${pac} --batch_size=${bs}
done

seeds="2 3 5 9 21"
for seed in $seeds ;
do
  echo "**********************************************"
  CUDA_VISIBLE_DEVICES=-1 python main_wgan.py ${use_case} --wandb_project=$wandbp --seed=$seed --epochs=$eps --disc_repeats=${rep} --d_lr=${lr} --g_lr=${lr} --pac=${pac} --batch_size=${bs} --version=constrained --label_ordering="random"
done
