#!/bin/bash

python create_circle.py --num_nodes 100

python prepare_minigpt_path.py --num_nodes 100 --num_of_paths 20 --graph_type circle --problem path

python train.py --num_nodes 100 --problem path --dataset circle

python test_simple.py --num_nodes 100 --num_of_paths 20 --problem path --graph_type circle


