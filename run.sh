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

python create_circle.py --num_nodes 100

python prepare_minigpt_path.py --num_nodes 100 --num_of_paths 20 --graph_type circle --problem path

python train.py --num_nodes 100 --problem path --dataset circle

python test_simple.py --num_nodes 100 --num_of_paths 20 --problem path --graph_type circle


