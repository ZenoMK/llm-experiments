import networkx as nx
import random
import os
import argparse
import numpy as np


def generate_random_rooted_tree(num_nodes):
    if num_nodes < 3:
        raise ValueError("Tree must have at least 3 nodes (1 root, 1 intermediate, 1 leaf)")

    G = nx.DiGraph()
    G.add_node(0)  # Root node

    intermediate_nodes = list(range(1, (num_nodes // 2) + 1))
    leaf_nodes = list(range((num_nodes // 2) + 1, num_nodes))

    # Ensure each intermediate node connects to exactly one leaf
    for i, intermediate in enumerate(intermediate_nodes):
        if i < len(leaf_nodes):
            G.add_edge(0, intermediate)  # Connect root to intermediate node
            G.add_edge(intermediate, leaf_nodes[i])  # Connect intermediate to one unique leaf

    return G


def get_root_to_leaf_paths(G):
    roots = [node for node in G.nodes if G.in_degree(node) == 0]
    leaves = [node for node in G.nodes if G.out_degree(node) == 0]

    paths = []
    for leaf in leaves:
        path = []
        current = leaf
        while current in G:
            path.append(current)
            preds = list(G.predecessors(current))
            if preds:
                current = preds[0]
            else:
                break
        paths.append(list(reversed(path)))

    return paths


def create_dataset(G):
    root_to_leaf_paths = get_root_to_leaf_paths(G)

    # Split 50% of paths into training and 50% into testing
    random.shuffle(root_to_leaf_paths)
    split_idx = len(root_to_leaf_paths) // 2
    train_paths = root_to_leaf_paths[:split_idx]
    test_paths = root_to_leaf_paths[split_idx:]

    train_set = set()
    for path in train_paths:
        train_set.add(tuple(path))

        # Add all intermediary node paths to training set
        for i in range(1, len(path)):
            train_set.add(tuple(path[:i]))  # Root to intermediary
            train_set.add(tuple(path[i:]))  # Intermediary to leaf

    test_set = [path for path in test_paths if tuple(path) not in train_set]  # Ensure no train-test leakage

    return list(train_set), test_set



def format_data(data):
    return f"{data[0]} {data[-1]} " + ' '.join(str(num) for num in data) + '\n'


def write_dataset(dataset, file_name):
    with open(file_name, "w") as file:
        for data in dataset:
            file.write(format_data(data))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a random rooted tree and datasets.')
    parser.add_argument('--num_nodes', type=int, default=100, help='Number of nodes in the tree')
    args = parser.parse_args()

    num_nodes = args.num_nodes
    rooted_tree = generate_random_rooted_tree(num_nodes)
    train_set, test_set = create_dataset(rooted_tree)

    folder_name = os.path.join(os.path.dirname(__file__), f'{num_nodes}_tree')
    os.makedirs(folder_name, exist_ok=True)


    write_dataset(train_set, os.path.join(os.path.dirname(__file__), f'{num_nodes}_tree/train_20.txt'))
    write_dataset(test_set, os.path.join(os.path.dirname(__file__), f'{num_nodes}_tree/test.txt'))
    nx.write_graphml(rooted_tree, os.path.join(os.path.dirname(__file__), f'{num_nodes}_tree/path_graph.graphml'))

    print(f"Dataset generated in {folder_name}/")
