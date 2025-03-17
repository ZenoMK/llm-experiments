import os
from collections import Counter
import tiktoken
import matplotlib.pyplot as plt
from model import GPTConfig, GPT
import numpy as np
import networkx as nx
import argparse
import random
import pickle
import re
import torch
from utils_final import (
    AttentionVisualizer
)

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_iter', type=int, default=10000)
parser.add_argument('--graph_type', type=str, default='simple_graph')
parser.add_argument('--config', type=str, default='1_1_120')
parser.add_argument('--temperature', type=float, default=1)
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--num_nodes', type=int, default=100)
parser.add_argument('--num_of_paths', type=int, default=20)
parser.add_argument("--problem", type=str, default="path", help="Which algorithmic problem (path/cut)")

args = parser.parse_args()
dataset = args.graph_type
ckpt_iter = args.ckpt_iter
problem = args.problem
device = args.device
temperature = args.temperature
num_nodes = args.num_nodes
num_of_paths = args.num_of_paths
config = args.config

data_path = f'data/{dataset}/{num_nodes}_{problem}'
meta_path = f'{data_path}/meta.pkl'

print(f"Loading meta from {meta_path}...")
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)

stoi, itos = meta['stoi'], meta['itos']
max_new_tokens = meta['block_size']
top_k = len(itos)
simple_format = meta['simple_format']

out_dir = f'out/{dataset}_{config}_{num_nodes}_{problem}/'

if (num_of_paths == 0):
    ckpt_path = os.path.join(out_dir, f'{ckpt_iter}_ckpt.pt')
else:
    ckpt_path = os.path.join(out_dir, f'{ckpt_iter}_ckpt_{num_of_paths}.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)

model.eval()
model.to(device)

tokenizer = tiktoken.get_encoding("gpt2")
print(type(out_dir))
viz = AttentionVisualizer(model, tokenizer, out_dir = out_dir, test_path=f'{data_path}/test.txt', meta_path=meta_path)
viz.infer_and_visualize_attention( heads=[0], layers = [0], input_text="45 99 45 92 99 92 45", problem = "path", specific_path=False)


path_graph = f'{data_path}/path_graph.graphml'
path_graph = nx.read_graphml(path_graph)


def find_third_number_position(number_string):
    """
        path = number_string.split()
    start, end = path[0], path[1]

    # Find the index where the path reaches 'end'
    try:
        split_idx = path.index(end, 2)  # Look for 'end' after the first two elements
    except ValueError:
        return 'incorrect path, does not reach end'

    position = sum(len(num) for num in path[:split_idx+1]) + split_idx #- 1
    return position
"""
    numbers = number_string.split()
    third_number_index = 2
    third_number_index = random.randint(3, len(numbers) // 2)
    position = sum(len(num) for num in numbers[:third_number_index]) + third_number_index-1
    return position

def encode(s):
    ss = s.split(" ")
    encoded_string = [stoi[ch] for ch in ss]
    return encoded_string


def decode(l):
    dec = ""
    for i in l:
        dec = dec + itos[i] + " "
    return dec[:-1]


def check_path(G, gen_str):
    path = re.findall(r'\d+', gen_str)
    if len(path) < 5:
        return 'wrong syntax'

    #path = list(map(int, path))  # Convert extracted strings to integers

    for node in path:
        if node not in G:
            return 'wrong syntax'

    start, end = path[0], path[1]

    # Find the index where the path reaches 'end'
    try:
        split_idx = path.index(end, 2)  # Look for 'end' after the first two elements
    except ValueError:
        return 'incorrect path, does not reach end'

    forward_path = path[2:split_idx + 1]
    return_path = path[split_idx:]

    if return_path[-1] != start:
        return 'incorrect return'

    # Check forward path validity
    for i in range(len(forward_path) - 1):
        if not G.has_edge(forward_path[i], forward_path[i + 1]):
            return f'non-existence path {forward_path[i], forward_path[i + 1]}'

    # Check return path validity (reverse direction)
    for i in range(len(return_path) - 1):
        if not G.has_edge(return_path[i + 1], return_path[i]):
            return f'non-existence reverse path {return_path[i + 1], return_path[i]}'

    # Check if the forward path is the reverse of the return path (excluding end node)
    if not forward_path == list(reversed(return_path)):
        print(forward_path)
        print(return_path)
        return 'forward path is not the reverse of return path'

    return ''


def check_path_unreachable(G, gen_str, gt):
    path = re.findall(r'\d+|x', gen_str)
    if 'x' in path and len(path) < 4:
        return 0 if 'x' in gt else 1

    if 'x' in gt and 'x' not in gen_str:
        return 1

    return check_path(G, gen_str)


typedata = 'test'
f = open(f'{data_path}/{typedata}.txt', encoding='gbk')
texts = []
encode_texts = []
ground_truth = []

for line in f:
    if not simple_format:
        texts.append(line.split(':')[0] + ':')
        encode_texts.append(encode(line.split(':')[0] + ':'))
    else:
        pos = find_third_number_position(line)
        if (line[:pos] != ''):
            texts.append(line[:pos])
            encode_texts.append(encode(line[:pos]))

    ground_truth.append(line)

ground_truth = np.array(ground_truth)
from torch.nn.utils.rnn import pad_sequence

# Convert list of lists to tensors (assuming each sublist is a tensor)
encode_texts = [torch.tensor(seq, dtype=torch.long) for seq in encode_texts]

# Pad sequences to max length (e.g., 4)
encode_texts = pad_sequence(encode_texts, batch_first=True, padding_value=0)

from tqdm import tqdm

batch_size = 1000
ix = torch.randint(len(encode_texts), (batch_size,))

with open(out_dir + f'pred_{typedata}_{ckpt_iter}_partialpath.txt', 'w') as f:
    pass

wrong = 0
non_reverses = 0
for i in tqdm(range(10)):
    x = encode_texts[ix]
    x_gt = ground_truth[ix]
    print(x[0])
    # x = (torch.tensor(text, dtype=torch.long, device=device))
    y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
    PAD_TOKEN = 0  # or whatever your model's padding token is
    y = [[token for token in y[t].tolist() if token != PAD_TOKEN] for t in range(batch_size)]
    y_pred = [decode(y[t]).split('\n')[0] for t in range(batch_size)]
    print(y_pred[0])

    # Lists to store path lengths
    correct_lengths = []
    incorrect_lengths = []

    with open(out_dir + f'pred_{typedata}_{ckpt_iter}_partialpath.txt', 'a') as f:
        for t, item in enumerate(y_pred):
            symbol = check_path(path_graph, item)
            path_len = len(re.findall(r'\d+', item))
            if symbol not in ["",'forward path is not the reverse of return path']:
                incorrect_lengths.append(path_len)
                wrong = wrong + 1
            else:
                if symbol == 'forward path is not the reverse of return path':
                    non_reverses += 1
                correct_lengths.append(path_len)
            f.write(item + " % " + symbol + '\n')
        f.write(f"Number of wrongs: {wrong}" + '\n')
        f.write(f"Number of valid non-reversals: {non_reverses}" + '\n')

    # Plotting
correct_counts = Counter(correct_lengths)
incorrect_counts = Counter(incorrect_lengths)

# Get all unique path lengths
all_lengths = set(correct_counts.keys()).union(incorrect_counts.keys())

# Compute proportion of incorrect paths per length
proportions = {
    length: correct_counts[length] / (correct_counts[length] + incorrect_counts[length])
    for length in all_lengths
}

# Sorting lengths for plotting
sorted_lengths = sorted(proportions.keys())
sorted_proportions = [proportions[length] for length in sorted_lengths]

# Plotting
plt.figure(figsize=(8, 5))
plt.bar(sorted_lengths, sorted_proportions, color="green", alpha=0.7)
plt.xlabel("Path Length")
plt.ylabel("Proportion of Correct Paths")
plt.title("Proportion of Correct Paths per Path Length")
plt.xticks(sorted_lengths)
plt.ylim(0, 1)  # Proportion ranges from 0 to 1
plt.savefig(out_dir + f'pathlen_proportion.png', dpi=400)

