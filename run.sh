#!/bin/bash

python create_circle.py --num_nodes 95

python prepare_minigpt_path.py --num_nodes 95 --num_of_paths 20 --graph_type circle --problem path


