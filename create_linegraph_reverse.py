import networkx as nx
import random
import os
import argparse
import numpy


def generate_undirected_linegraph(num_nodes):
    # Create an empty undirected graph
    G = nx.Graph()

    # Add nodes to the graph
    for i in range(num_nodes):
        G.add_node(i)

    # Add edges to form a line (path) graph
    for i in range(num_nodes - 1):
        G.add_edge(i, i + 1)

    return G




def walk(source_node, target_node):
    if source_node < target_node:
        lst = [i for i in range(source_node, target_node + 1)]
        return lst + lst[-2::-1]
    else:
        lst = [i for i in range(source_node, target_node - 1, -1)]
        return lst + lst[-2::-1]



def create_dataset(i):
    train_set = []
    test_set = []
    train_num_per_pair = max(i, 1)
    for target_node in range(num_nodes):
        cnt = 0  # to avoid some target not appear in training dataset
        for source_node in range(num_nodes):
            lst = walk(source_node, target_node)
            if (data[source_node][target_node] == 1):
                train_set.append([source_node, target_node] + lst)
            if (data[source_node][target_node] == -1):
                test_set.append([source_node, target_node] + lst)

    return train_set, test_set

def obtain_stats(dataset):
    max_len = 0
    pairs = set()

    for data in dataset:
        max_len = max(max_len, len(data))
        pairs.add((data[0], data[-1]))

    len_stats = [0] * (max_len + 1)

    for data in dataset:
        length = len(data)
        len_stats[length] += 1

    print('number of source target pairs:', len(pairs))
    for ii in range(3, len(len_stats)):
        print(f'There are {len_stats[ii]} paths with length {ii - 3}')


def format_data(data):
    return f"{data[0]} {data[1]} " + ' '.join(str(num) for num in data[2:]) + '\n'


def write_dataset(dataset, file_name):
    with open(file_name, "w") as file:
        for data in dataset:
            file.write(format_data(data))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a random graph based on the given parameters.')
    parser.add_argument('--num_nodes', type=int, default=100, help='Number of nodes in the graph')
    parser.add_argument('--edge_prob', type=float, default=0.1,
                        help='Probability of creating an edge between two nodes')
    parser.add_argument('--DAG', type=bool, default=True, help='Whether the graph should be a Directed Acyclic Graph')
    parser.add_argument('--chance_in_train', type=float, default=0.5, help='Chance of a pair being in the training set')
    parser.add_argument('--num_of_paths', type=int, default=20,
                        help='Number of paths per pair nodes in training dataset')

    args = parser.parse_args()

    num_nodes = args.num_nodes
    edge_prob = args.edge_prob
    DAG = args.DAG
    chance_in_train = args.chance_in_train
    num_of_paths = args.num_of_paths

    random_digraph = generate_undirected_linegraph(num_nodes)

    folder_name = os.path.join(os.path.dirname(__file__), f'{num_nodes}_reversepath')
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    data = numpy.zeros([num_nodes, num_nodes])
    for target_node in range(num_nodes):
        cnt = 0  # to avoid some target not appear in training dataset
        for source_node in range(num_nodes):
            if source_node == target_node:
                continue
            elif random.random() < chance_in_train:
                data[source_node][target_node] = 1
            else:
                data[source_node][target_node] = -1


    train_set, test_set = create_dataset(num_of_paths)

    obtain_stats(train_set)
    print('number of source target pairs:', len(test_set))

    write_dataset(train_set, os.path.join(os.path.dirname(__file__), f'{num_nodes}_reversepath/train_{num_of_paths}.txt'))
    write_dataset(test_set, os.path.join(os.path.dirname(__file__), f'{num_nodes}_reversepath/test.txt'))
    nx.write_graphml(random_digraph, os.path.join(os.path.dirname(__file__), f'{num_nodes}_reversepath/path_graph.graphml'))


