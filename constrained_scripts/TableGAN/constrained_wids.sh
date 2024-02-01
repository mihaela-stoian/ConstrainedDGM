#!/bin/bash

# Fixed parameters
use_case="wids"
server="PLACEHOLDER_SERVER"
wandbp="TableGAN-dgm_${server}_${use_case}_constrained_tolerance_shifted"
seed=9
eps=50

# Defaults for params that will change
default_optimiser="rmsprop"
default_lr=0.0010
default_bs=128
default_random_dim=100


seeds="2 5 7 9 21"
for seed in $seeds ;
do
#  CUDA_VISIBLE_DEVICES=-1  python main_tableGAN.py ${use_case} --wandb_project=$wandbp --seed=$seed --epochs=$eps --optimiser=${default_optimiser} --lr=${default_lr} --batch_size=${default_bs} --random_dim=${default_random_dim}
#  CUDA_VISIBLE_DEVICES=-1  python main_tableGAN.py ${use_case} --wandb_project=$wandbp --seed=$seed --epochs=$eps --optimiser=${default_optimiser} --lr=${default_lr} --batch_size=${default_bs} --random_dim=${default_random_dim} --version="constrained" --label_ordering="corr"
#  CUDA_VISIBLE_DEVICES=-1  python main_tableGAN.py ${use_case} --wandb_project=$wandbp --seed=$seed --epochs=$eps --optimiser=${default_optimiser} --lr=${default_lr} --batch_size=${default_bs} --random_dim=${default_random_dim} --version="constrained" --label_ordering="kde"
  CUDA_VISIBLE_DEVICES=-1  python main_tableGAN.py ${use_case} --wandb_project=$wandbp --seed=$seed --epochs=$eps --optimiser=${default_optimiser} --lr=${default_lr} --batch_size=${default_bs} --random_dim=${default_random_dim} --version="constrained" --label_ordering="random"
done
