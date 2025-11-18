#!/bin/bash
#SBATCH --job-name=1_nt_transformer_full_training
#SBATCH --output=%j.1_nt_transformer_full_training.out
#SBATCH --error=%j.1_nt_transformer_full_training.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --account=EUHPC_D26_076
#SBATCH --partition=boost_usr_prod

######################## 1.  加载模块 ########################
# salloc -N 1 --ntasks-per-node=1 --cpus-per-task=1 --gres=gpu:1 -A EUHPC_D26_076 --time=4:00:00 --partition=boost_usr_prod
module purge
#module profile archive
module load cuda/12.2
module load llvm/14.0.6--gcc--12.2.0-cuda-12.2
module load python/3.11.7

source glm_env/bin/activate
python scripts/train.py --config ./config/training/default.yaml