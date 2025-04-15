import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import argparse
import pickle
from model import GPT, GPTConfig
import seaborn as sns


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

    # Define a forward hook to grab activations from a specific layer
    def hook_fn(module, input, output):
        activations['layer'] = output.detach().cpu()

    # Register the hook (you can change to a specific layer if you want)
    handle = model.transformer.h[0].mlp.c_fc.register_forward_hook(hook_fn)

    # Forward pass
    logits, loss, _ = model(idx, return_hidden_states=True)

    handle.remove()  # Clean up hook

    # activations['layer'] will have shape [batch, seq_len, hidden_dim]
    act = activations['layer'].squeeze(0)  # remove batch dim -> (seq_len, hidden_dim)
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


def print_model_structure(model):
    print("\n[INFO] Model Structure Overview:")

    if hasattr(model, 'config'):
        if hasattr(model.config, 'n_layer'):
            print(f" - Number of layers: {model.config.n_layer}")
        if hasattr(model.config, 'n_head'):
            print(f" - Number of heads per layer: {model.config.n_head}")
        if hasattr(model.config, 'n_embd'):
            print(f" - Hidden size (embedding dim): {model.config.n_embd}")
    else:
        print(" - Model config not found. Trying manual inspection...")

        if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            num_layers = len(model.transformer.h)
            print(f" - Number of layers: {num_layers}")
        else:
            print(" - Could not find transformer layers in model.")

    print("-----------------------------------\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Transformer Neuron Activations with t-SNE")
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

    # Load model
    model = load_custom_gpt_from_checkpoint(args.checkpoint_path)
    model.to(args.device)

    # Print model structure info
    print_model_structure(model)

    # Load meta.pkl
    stoi, itos = load_meta(meta_path)

    # Your sample text
    text = "14 61 65 14 29 35 43 52 61 65"


    # Get activations
    activations = get_activations(text, model, stoi, args.device)

    # Tokenized text
    tokenized_text = text.split(" ")

    # Visualize
    visualize_activations_heatmap(activations[:len(tokenized_text)], tokenized_text)
