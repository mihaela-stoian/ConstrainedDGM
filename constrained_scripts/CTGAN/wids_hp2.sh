#!/bin/bash

# Fixed parameters
use_case="wids"
server="PLACEHOLDER_SERVER"
wandbp="CTGAN_${use_case}_hp"
eps=50
seed=2


## Defaults for params that will change
default_pac=1
default_bs=500
default_decay=0.000001

#default_optimiser="adam"
#lrs="0.0002 0.001 0.0005"
#for lr in $lrs ;
#do
#  CUDA_VISIBLE_DEVICES=-1 python main_ctgan.py  ${use_case} --wandb_project=$wandbp --seed=$seed --epochs=$eps --optimiser=${default_optimiser} --generator_lr=${lr} --discriminator_lr=${lr} --batch_size=${default_bs} --generator_decay=${default_decay} --discriminator_decay=${default_decay} --pac=${default_pac}
#  CUDA_VISIBLE_DEVICES=-1 python main_ctgan.py  ${use_case} --wandb_project=$wandbp --seed=$seed --epochs=$eps --optimiser=${default_optimiser} --generator_lr=${lr} --discriminator_lr=${lr} --batch_size=${default_bs} --generator_decay=${default_decay} --discriminator_decay=${default_decay} --pac=${default_pac} --version="constrained" --label_ordering="random"
#done

default_optimiser="rmsprop"
lrs="0.001 0.0005"
for lr in $lrs ;
do
  CUDA_VISIBLE_DEVICES=-1 python main_ctgan.py  ${use_case} --wandb_project=$wandbp --seed=$seed --epochs=$eps --optimiser=${default_optimiser} --generator_lr=${lr} --discriminator_lr=${lr} --batch_size=${default_bs} --generator_decay=${default_decay} --discriminator_decay=${default_decay} --pac=${default_pac}
  CUDA_VISIBLE_DEVICES=-1 python main_ctgan.py  ${use_case} --wandb_project=$wandbp --seed=$seed --epochs=$eps --optimiser=${default_optimiser} --generator_lr=${lr} --discriminator_lr=${lr} --batch_size=${default_bs} --generator_decay=${default_decay} --discriminator_decay=${default_decay} --pac=${default_pac} --version="constrained" --label_ordering="random"
done

#default_optimiser="rmsprop"
#lr=0.001
#default_bs=128
#CUDA_VISIBLE_DEVICES=-1 python main_ctgan.py  ${use_case} --wandb_project=$wandbp --seed=$seed --epochs=$eps --optimiser=${default_optimiser} --generator_lr=${lr} --discriminator_lr=${lr} --batch_size=${default_bs} --generator_decay=${default_decay} --discriminator_decay=${default_decay} --pac=${default_pac}
#CUDA_VISIBLE_DEVICES=-1 python main_ctgan.py  ${use_case} --wandb_project=$wandbp --seed=$seed --epochs=$eps --optimiser=${default_optimiser} --generator_lr=${lr} --discriminator_lr=${lr} --batch_size=${default_bs} --generator_decay=${default_decay} --discriminator_decay=${default_decay} --pac=${default_pac} --version="constrained" --label_ordering="random"
