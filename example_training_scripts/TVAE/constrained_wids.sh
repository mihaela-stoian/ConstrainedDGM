#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --mem-per-cpu=2G
#SBATCH --time=12:00:00
#SBATCH --job-name=wids-tvae
#SBATCH --partition=short
#SBATCH --clusters=all

module load Anaconda3
source activate /data/PLACEHOLDER_PATH/PLACEHOLDER/conda_envs/c_gan
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/PLACEHOLDER_PATH/PLACEHOLDER/conda_envs/c_gan/lib

# Fixed parameters
use_case="wids"
server="PLACEHOLDER_SERVER3"
wandbp="TVAE_${use_case}"
seed=9
eps=50

bs=70
l2scale=0.000005
loss_factor=2
seeds="2 5 7 9 21"
for seed in $seeds ;
do
  CUDA_VISIBLE_DEVICES=-1 python main_tvae.py  ${use_case} --wandb_project=$wandbp --seed=$seed --epochs=$eps  --batch_size=${bs} --l2scale=${l2scale} --loss_factor=${loss_factor}
  CUDA_VISIBLE_DEVICES=-1 python main_tvae.py  ${use_case} --wandb_project=$wandbp --seed=$seed --epochs=$eps  --batch_size=${bs} --l2scale=${l2scale} --loss_factor=${loss_factor} --version="constrained" --label_ordering="corr"
  CUDA_VISIBLE_DEVICES=-1 python main_tvae.py  ${use_case} --wandb_project=$wandbp --seed=$seed --epochs=$eps  --batch_size=${bs} --l2scale=${l2scale} --loss_factor=${loss_factor} --version="constrained" --label_ordering="kde"
  CUDA_VISIBLE_DEVICES=-1 python main_tvae.py  ${use_case} --wandb_project=$wandbp --seed=$seed --epochs=$eps  --batch_size=${bs} --l2scale=${l2scale} --loss_factor=${loss_factor} --version="constrained" --label_ordering="random"
done
