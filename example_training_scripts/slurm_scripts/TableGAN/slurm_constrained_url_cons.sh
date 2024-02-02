#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --mem-per-cpu=2G
#SBATCH --time=12:00:00
#SBATCH --job-name=c-tablegan-url
#SBATCH --partition=short
#SBATCH --clusters=all

module load Anaconda3
source activate /data/PLACEHOLDER_PATH/PLACEHOLDER/conda_envs/c_gan
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/PLACEHOLDER_PATH/PLACEHOLDER/conda_envs/c_gan/lib
#source activate $DATA/conda_envs/c_gan
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$DATA/conda_envs/c_gan/lib

# Fixed parameters
use_case="url"
server="PLACEHOLDER_SERVER2"
wandbp="TableGAN-dgm_${server}_${use_case}_NO_DISJ_fixed_scaler"
seed=9
eps=300

# Defaults for params that will change
default_optimiser="adam"
default_lr=0.0010
default_bs=128
default_random_dim=100


seeds="2 5 7 9 21"
for seed in $seeds ;
do
#  CUDA_VISIBLE_DEVICES=-1  python main_tableGAN.py ${use_case} --wandb_project=$wandbp --seed=$seed --epochs=$eps --optimiser=${default_optimiser} --lr=${default_lr} --batch_size=${default_bs} --random_dim=${default_random_dim}
  CUDA_VISIBLE_DEVICES=-1  python main_tableGAN.py ${use_case} --wandb_project=$wandbp --seed=$seed --epochs=$eps --optimiser=${default_optimiser} --lr=${default_lr} --batch_size=${default_bs} --random_dim=${default_random_dim} --version="constrained" --label_ordering="corr"
  CUDA_VISIBLE_DEVICES=-1  python main_tableGAN.py ${use_case} --wandb_project=$wandbp --seed=$seed --epochs=$eps --optimiser=${default_optimiser} --lr=${default_lr} --batch_size=${default_bs} --random_dim=${default_random_dim} --version="constrained" --label_ordering="kde"
#  CUDA_VISIBLE_DEVICES=-1  python main_tableGAN.py ${use_case} --wandb_project=$wandbp --seed=$seed --epochs=$eps --optimiser=${default_optimiser} --lr=${default_lr} --batch_size=${default_bs} --random_dim=${default_random_dim} --version="constrained" --label_ordering="random"
done

#seeds="2 5 7 9 21"
#for seed in $seeds ;
#do
##  CUDA_VISIBLE_DEVICES=-1  python main_tableGAN.py ${use_case} --wandb_project=$wandbp --seed=$seed --epochs=$eps --optimiser=${default_optimiser} --lr=${default_lr} --batch_size=${default_bs} --random_dim=${default_random_dim}
##  CUDA_VISIBLE_DEVICES=-1  python main_tableGAN.py ${use_case} --wandb_project=$wandbp --seed=$seed --epochs=$eps --optimiser=${default_optimiser} --lr=${default_lr} --batch_size=${default_bs} --random_dim=${default_random_dim} --version="constrained" --label_ordering="corr"
##  CUDA_VISIBLE_DEVICES=-1  python main_tableGAN.py ${use_case} --wandb_project=$wandbp --seed=$seed --epochs=$eps --optimiser=${default_optimiser} --lr=${default_lr} --batch_size=${default_bs} --random_dim=${default_random_dim} --version="constrained" --label_ordering="kde"
##  CUDA_VISIBLE_DEVICES=-1  python main_tableGAN.py ${use_case} --wandb_project=$wandbp --seed=$seed --epochs=$eps --optimiser=${default_optimiser} --lr=${default_lr} --batch_size=${default_bs} --random_dim=${default_random_dim} --version="constrained" --label_ordering="random"
#done
