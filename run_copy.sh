#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100-pcie:1
#SBATCH --time=01:00:00
#SBATCH --job-name=gpu_run
#SBATCH --mem=10GB
#SBATCH --ntasks=1
#SBATCH --output=myjob.%j.out
#SBATCH --error=myjob.%j.err

module load gcc/11.1.0

module load anaconda3/2022.05

source /shared/centos7/anaconda3/2022.05/etc/profile.d/conda.sh

conda init bash

conda activate pytorch_env

python create_list_copy.py --num_nodes 100

python prepare_minigpt_list.py  --num_nodes 100 --problem list_copy

python train.py --num_nodes 100 --max_iter 1000  --problem list_copy --dataset list --num_of_paths 20 --n_head 6 --n_layer 6

python test_list_hinting.py --num_nodes 100 --problem list_copy  --ckpt_iter 1000