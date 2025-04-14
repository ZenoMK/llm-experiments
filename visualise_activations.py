import torch
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import pickle
from model import GPT, GPTConfig

def load_meta(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi = meta['stoi']
    itos = meta['itos']
    return stoi, itos

def custom_encode(text, stoi):
    text = text.rstrip()
    ss = text.split(" ")
    encoded = [stoi[ch] for ch in ss]
    return encoded

def custom_decode(indices, itos):
    return " ".join(itos[i] for i in indices)

def get_activations(text, model, stoi, device):
    idx = custom_encode(text, stoi)
    idx = torch.tensor(idx).unsqueeze(0).to(device)

    activations = {}

    # Hook into the first MLP layer
    def hook_fn(module, input, output):
        activations['layer'] = output.detach().cpu()

    handle = model.transformer.h[0].mlp.c_fc.register_forward_hook(hook_fn)

    logits, loss, _ = model(idx, return_hidden_states=True)

    handle.remove()

    act = activations['layer'].squeeze(0)  # (seq_len, hidden_dim)
    return act

def visualize_activations_heatmap(activations, labels):
    # activations: (seq_len, hidden_dim)
    plt.figure(figsize=(12, 8))
    sns.heatmap(activations.numpy(), cmap="viridis", cbar=True, xticklabels=False, yticklabels=labels)
    plt.ylabel("Tokens")
    plt.xlabel("Neuron index")
    plt.title("Neuron Activation Heatmap")
    plt.tight_layout()
    plt.savefig("neuron_activation_heatmap.png")
    plt.close()

def load_custom_gpt_from_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval()
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Transformer Neuron Activations with Heatmap")
    parser.add_argument("--checkpoint_path", type=str, help="Path to the custom GPT model checkpoint (.pt file)")
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
    num_nodes = args.num_nodes
    problem = args.problem
    device = args.device
    config = args.config

    data_path = f'data/{dataset}/{num_nodes}_{problem}'
    meta_path = f'{data_path}/meta.pkl'

    model = load_custom_gpt_from_checkpoint(args.checkpoint_path)
    model.to(args.device)

    stoi, itos = load_meta(meta_path)

    text = "0 1 2 50 51 52 53 15 85 60 23"
    activations = get_activations(text, model, stoi, args.device)

    tokenized_text = text.split(" ")

    visualize_activations_heatmap(activations[:len(tokenized_text)], tokenized_text)
