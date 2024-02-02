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
wandbp="TableGAN-dgm_${server}_${use_case}_NO_DISJ"
seed=9
eps=25

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
# CUDA_VISIBLE_DEVICES=-1  python main_tableGAN.py ${use_case} --wandb_project=$wandbp --seed=$seed --epochs=$eps --optimiser=${default_optimiser} --lr=${default_lr} --batch_size=${default_bs} --random_dim=${default_random_dim} --version="constrained" --label_ordering="random"
done


seeds="2 5 7 9 21"
for seed in $seeds ;
do
#  CUDA_VISIBLE_DEVICES=-1  python main_tableGAN.py ${use_case} --wandb_project=$wandbp --seed=$seed --epochs=$eps --optimiser=${default_optimiser} --lr=${default_lr} --batch_size=${default_bs} --random_dim=${default_random_dim}
#  CUDA_VISIBLE_DEVICES=-1  python main_tableGAN.py ${use_case} --wandb_project=$wandbp --seed=$seed --epochs=$eps --optimiser=${default_optimiser} --lr=${default_lr} --batch_size=${default_bs} --random_dim=${default_random_dim} --version="constrained" --label_ordering="corr"
#  CUDA_VISIBLE_DEVICES=-1  python main_tableGAN.py ${use_case} --wandb_project=$wandbp --seed=$seed --epochs=$eps --optimiser=${default_optimiser} --lr=${default_lr} --batch_size=${default_bs} --random_dim=${default_random_dim} --version="constrained" --label_ordering="kde"
 CUDA_VISIBLE_DEVICES=-1  python main_tableGAN.py ${use_case} --wandb_project=$wandbp --seed=$seed --epochs=$eps --optimiser=${default_optimiser} --lr=${default_lr} --batch_size=${default_bs} --random_dim=${default_random_dim} --version="constrained" --label_ordering="random"
done
