import networkx as nx
import random
import os
import argparse
import numpy


def generate_markov_chain(num_states, edge_prob, output_file="path_graph.graphml"):
    # Create a directed graph
    G = nx.DiGraph()

    # Add states (nodes)
    for i in range(num_states):
        G.add_node(i)

    # Add transitions (edges) with probabilities
    for i in range(num_states):
        total_weight = 0
        edges = []

        for j in range(num_states):
            if i != j and random.random() < edge_prob:
                weight = random.random()
                edges.append((j, weight))
                total_weight += weight

        # Normalize probabilities
        for j, weight in edges:
            G.add_edge(i, j, weight=weight / total_weight if total_weight > 0 else 1.0 / num_states)

    # Save to GraphML
    nx.write_graphml(G, output_file)
    print(f"Markov chain saved as {output_file}")

    return G


def get_reachable_nodes(TC, target_node):
    # Find the predecessors in the transitive closure (nodes that can reach the target_node)
    reachable_from = TC.predecessors(target_node)
    return list(reachable_from)


def obtain_reachability(TC):
    reachability = {}
    pairs = 0
    for node in random_digraph.nodes():
        reachability[node] = get_reachable_nodes(TC, node)
        pairs += len(reachability[node])
    return reachability, pairs


def random_walk(source_node):
    current_node = source_node
    path = [current_node]

    while len(path) < 20:
        neighbors = list(random_digraph.successors(current_node))
        if not neighbors:
            return False  # No outgoing edges, terminate walk

        weights = [random_digraph[current_node][neighbor]['weight'] for neighbor in neighbors]
        current_node = random.choices(neighbors, weights=weights)[0]
        path.append(current_node)

    return path


def create_dataset(i):
    train_set = []
    train_num_per_pair = max(i, 1)
    for source_node in range(num_nodes):
        num_paths = 0
        while num_paths < train_num_per_pair:
            path = random_walk(source_node)
            train_set.append(path)
            num_paths += 1

    return train_set



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
    return ' '.join(str(num) for num in data) + '\n'


def write_dataset(dataset, file_name):
    with open(file_name, "w") as file:
        for data in dataset:
            if len(data) > 0:
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

    random_digraph = generate_markov_chain(num_nodes, edge_prob)
    TC = nx.transitive_closure(random_digraph)
    reachability, feasible_pairs = obtain_reachability(TC)

    folder_name = os.path.join(os.path.dirname(__file__), f'{num_nodes}_path')
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


    train_set = create_dataset(num_of_paths)

    obtain_stats(train_set)

    write_dataset(train_set, os.path.join(os.path.dirname(__file__), f'{num_nodes}_path/train_{num_of_paths}.txt'))
    nx.write_graphml(random_digraph, os.path.join(os.path.dirname(__file__), f'{num_nodes}_path/path_graph.graphml'))


