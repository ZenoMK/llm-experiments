#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100-pcie:1
#SBATCH --time=01:00:00
#SBATCH --job-name=gpu_run
#SBATCH --mem=4GB
#SBATCH --ntasks=1
#SBATCH --output=myjob.%j.out
#SBATCH --error=myjob.%j.err

module load gcc/11.1.0

module load anaconda3/2022.05

conda init bash

conda activate pytorch_env

python generate_tree_path.py --num_nodes 100

python prepare_minigpt_path.py --num_nodes 100 --problem tree --graph_type tree

python train_full.py --num_nodes 100 --problem tree --num_of_paths 20 --dataset tree --n_head 1 --n_layer 1 --use_identity_embeddings --max_iter 1000

python test_full.py --num_nodes 100 --num_of_paths 20 --graph_type tree --problem tree --ckpt_iter 1000

