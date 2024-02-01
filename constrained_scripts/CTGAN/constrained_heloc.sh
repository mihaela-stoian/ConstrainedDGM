#!/bin/bash -l
#SBATCH -N 1
#SBATCH -n 14
#SBATCH -p bigmem
#SBATCH --time=0-47:00:00
#SBATCH --qos=normal
#SBATCH -J RUN_ALL
#SBATCH --mail-type=all
#SBATCH --mail-user=PLACEHOLDER_USER
conda activate gan
git checkout unified_extended

# Fixed parameters
use_case="heloc"
server="hpc"
wandbp="CTGAN-dgm_${server}_${use_case}_constrained_NO_DISJ"
eps=20

#Defaults for params that will change
default_optimiser="adam"
default_lr=0.0002
default_bs=500
default_decay=0.000001
default_pac=1

seeds="2 5 7 9 21"
for seed in $seeds ;
do
#  CUDA_VISIBLE_DEVICES=-1 python main_ctgan.py  ${use_case} --wandb_project=$wandbp --seed=$seed --epochs=$eps --optimiser=${default_optimiser} --generator_lr=${default_lr} --discriminator_lr=${default_lr} --batch_size=${default_bs} --generator_decay=${default_decay} --discriminator_decay=${default_decay} --pac=${default_pac}
  CUDA_VISIBLE_DEVICES=-1 python main_ctgan.py  ${use_case} --wandb_project=$wandbp --seed=$seed --epochs=$eps --optimiser=${default_optimiser} --generator_lr=${default_lr} --discriminator_lr=${default_lr} --batch_size=${default_bs} --generator_decay=${default_decay} --discriminator_decay=${default_decay} --pac=${default_pac} --version="constrained" --label_ordering="corr"
  CUDA_VISIBLE_DEVICES=-1 python main_ctgan.py  ${use_case} --wandb_project=$wandbp --seed=$seed --epochs=$eps --optimiser=${default_optimiser} --generator_lr=${default_lr} --discriminator_lr=${default_lr} --batch_size=${default_bs} --generator_decay=${default_decay} --discriminator_decay=${default_decay} --pac=${default_pac} --version="constrained" --label_ordering="kde"
# CUDA_VISIBLE_DEVICES=-1 python main_ctgan.py  ${use_case} --wandb_project=$wandbp --seed=$seed --epochs=$eps --optimiser=${default_optimiser} --generator_lr=${default_lr} --discriminator_lr=${default_lr} --batch_size=${default_bs} --generator_decay=${default_decay} --discriminator_decay=${default_decay} --pac=${default_pac} --version="constrained" --label_ordering="random"
done


seeds="2 5 7 9 21"
for seed in $seeds ;
do
#  CUDA_VISIBLE_DEVICES=-1 python main_ctgan.py  ${use_case} --wandb_project=$wandbp --seed=$seed --epochs=$eps --optimiser=${default_optimiser} --generator_lr=${default_lr} --discriminator_lr=${default_lr} --batch_size=${default_bs} --generator_decay=${default_decay} --discriminator_decay=${default_decay} --pac=${default_pac}
#  CUDA_VISIBLE_DEVICES=-1 python main_ctgan.py  ${use_case} --wandb_project=$wandbp --seed=$seed --epochs=$eps --optimiser=${default_optimiser} --generator_lr=${default_lr} --discriminator_lr=${default_lr} --batch_size=${default_bs} --generator_decay=${default_decay} --discriminator_decay=${default_decay} --pac=${default_pac} --version="constrained" --label_ordering="corr"
#  CUDA_VISIBLE_DEVICES=-1 python main_ctgan.py  ${use_case} --wandb_project=$wandbp --seed=$seed --epochs=$eps --optimiser=${default_optimiser} --generator_lr=${default_lr} --discriminator_lr=${default_lr} --batch_size=${default_bs} --generator_decay=${default_decay} --discriminator_decay=${default_decay} --pac=${default_pac} --version="constrained" --label_ordering="kde"
 CUDA_VISIBLE_DEVICES=-1 python main_ctgan.py  ${use_case} --wandb_project=$wandbp --seed=$seed --epochs=$eps --optimiser=${default_optimiser} --generator_lr=${default_lr} --discriminator_lr=${default_lr} --batch_size=${default_bs} --generator_decay=${default_decay} --discriminator_decay=${default_decay} --pac=${default_pac} --version="constrained" --label_ordering="random"
done
