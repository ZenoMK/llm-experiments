
import os
from collections import Counter
import tiktoken
import matplotlib.pyplot as plt
from model import GPTConfig, GPT
import numpy as np
import networkx as nx
import argparse
import pickle
import re
import torch
from utils_final import (
    AttentionVisualizer
)
import seaborn as sns


parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_iter', type=int, default=10000)
parser.add_argument('--graph_type', type=str, default='simple_graph')
parser.add_argument('--config', type=str, default='1_1_120')
parser.add_argument('--temperature', type=float, default=1)
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--num_nodes', type=int, default=100)
parser.add_argument('--num_of_paths', type=int, default=20)
parser.add_argument("--problem", type=str, default = "path", help ="Which algorithmic problem (path/cut)")


args = parser.parse_args()
dataset = args.graph_type
ckpt_iter = args.ckpt_iter
problem = args.problem
device = args.device
temperature = args.temperature
num_nodes = args.num_nodes
num_of_paths = args.num_of_paths
config = args.config



def evaluate_model_against_graph(model, G, tokenizer):
    #model = torch.load(model_path, torch.device('cpu'))
    #model.eval()

    num_states = len(G.nodes())
    real_probs_matrix = np.zeros((num_states, num_states))
    pred_probs_matrix = np.zeros((num_states, num_states))

    for node in G.nodes():
        neighbors = list(G.successors(node))
        if not neighbors:
            continue

        weights = [G[node][neighbor]['weight'] for neighbor in neighbors]
        input_tensor = torch.tensor([encode(node)]).unsqueeze(0)  # Example input format
        with torch.no_grad():
            output = model(input_tensor)[0]
            probabilities = torch.nn.functional.softmax(output, dim=-1).squeeze().tolist()

        for idx, neighbor in enumerate(neighbors):
            real_probs_matrix[int(node), int(neighbor)] = weights[idx]
            pred_probs_matrix[int(node), int(neighbor)] = probabilities[encode(str(idx))]

    # Compute RMSE
    rmse = np.sqrt(np.mean((real_probs_matrix - pred_probs_matrix) ** 2))
    print(f"Root Mean Squared Error: {rmse}")

    # Compute absolute and relative errors
    abs_error = np.abs(real_probs_matrix - pred_probs_matrix)
    relative_error = np.divide(abs_error, real_probs_matrix, where=real_probs_matrix != 0)
    # Plot heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(abs_error, ax=axes[0], cmap="Reds")
    axes[0].set_title("Absolute Errors")

    sns.heatmap(relative_error, ax=axes[1], cmap="Blues")
    axes[1].set_title("Relative Errors")

    plt.savefig(out_dir + f'errors.png', dpi=400)

    return True

def encode(s):
    return stoi[s]



data_path = f'data/{dataset}/{num_nodes}_{problem}'
meta_path = f'{data_path}/meta.pkl'
tokenizer = tiktoken.get_encoding("gpt2")
print(f"Loading meta from {meta_path}...")
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)

stoi, itos = meta['stoi'], meta['itos']
max_new_tokens = meta['block_size']
top_k = len(itos)
simple_format = meta['simple_format']

out_dir = f'out/{dataset}_{config}_{num_nodes}_{problem}/'

path_graph = f'{data_path}/path_graph.graphml'
path_graph = nx.read_graphml(path_graph)

if(num_of_paths == 0):
    ckpt_path = os.path.join(out_dir, f'{ckpt_iter}_ckpt.pt')
else:
    ckpt_path = os.path.join(out_dir, f'{ckpt_iter}_ckpt_{num_of_paths}.pt')

checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)

model.eval()
model.to(device)

evaluate_model_against_graph(model, path_graph, tokenizer)