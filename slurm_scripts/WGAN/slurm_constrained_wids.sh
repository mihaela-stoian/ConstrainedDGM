#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --mem-per-cpu=2G
#SBATCH --time=12:00:00
#SBATCH --job-name=wgan-wids
#SBATCH --partition=short
#SBATCH --clusters=all

module load Anaconda3
source activate /data/PLACEHOLDER_PATH/PLACEHOLDER/conda_envs/c_gan
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/PLACEHOLDER_PATH/PLACEHOLDER/conda_envs/c_gan/lib
#source activate $DATA/conda_envs/c_gan
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$DATA/conda_envs/c_gan/lib

# Fixed parameters
use_case="wids"
wandbp="wandb_constrained_wgan_${use_case}_fixed_scaler"
eps=30 # reduced this from 100 to 50 or 30, as the loss gets shaky later on
#eval_eps=150

rep=10
#opt="rmsprop"
lr=0.001
pac=1
bs=128


#seeds="2 3 5 9 21"
seeds="3 5 9 21"
for seed in $seeds ;
do
  echo "**********************************************"
  CUDA_VISIBLE_DEVICES=-1 python main_wgan.py ${use_case} --wandb_project=$wandbp --seed=$seed --epochs=$eps --disc_repeats=${rep} --d_lr=${lr} --g_lr=${lr} --pac=${pac} --batch_size=${bs}
#  CUDA_VISIBLE_DEVICES=-1 python main_wgan.py ${use_case} --wandb_project=$wandbp --seed=$seed --epochs=$eps --disc_repeats=${rep} --d_lr=${lr} --g_lr=${lr} --pac=${pac} --batch_size=${bs} --version=constrained --label_ordering="corr"
#  CUDA_VISIBLE_DEVICES=-1 python main_wgan.py ${use_case} --wandb_project=$wandbp --seed=$seed --epochs=$eps --disc_repeats=${rep} --d_lr=${lr} --g_lr=${lr} --pac=${pac} --batch_size=${bs} --version=constrained --label_ordering="kde"
#  CUDA_VISIBLE_DEVICES=-1 python main_wgan.py ${use_case} --wandb_project=$wandbp --seed=$seed --epochs=$eps --disc_repeats=${rep} --d_lr=${lr} --g_lr=${lr} --pac=${pac} --batch_size=${bs} --version=constrained --label_ordering="random"
done

seeds="2 3 5 9 21"
for seed in $seeds ;
do
  echo "**********************************************"
  CUDA_VISIBLE_DEVICES=-1 python main_wgan.py ${use_case} --wandb_project=$wandbp --seed=$seed --epochs=$eps --disc_repeats=${rep} --d_lr=${lr} --g_lr=${lr} --pac=${pac} --batch_size=${bs} --version=constrained --label_ordering="random"
done
