import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from transformers import AutoTokenizer
import argparse
from model import GPT, GPTConfig
import pickle

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



def get_embeddings(text, model, stoi):
    idx = custom_encode(text, stoi)  # This returns a list
    idx = torch.tensor(idx).unsqueeze(0).to(device)  # Convert list to tensor and unsqueeze it
    logits, loss, (tok_emb, pos_emb) = model(idx, return_hidden_states=True)
    embeddings = tok_emb.squeeze(0)  # (seq_len, embed_size)
    return embeddings


def visualize_embeddings_tsne(embeddings, labels):
    n_samples = embeddings.shape[0]
    perplexity = min(30, max(2, n_samples // 3))  # Dynamically adjust perplexity to avoid errors

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings.detach().numpy())

    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c='blue', alpha=0.6)

    for i, label in enumerate(labels):
        plt.annotate(label, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]), fontsize=8, alpha=0.7)

    plt.title("t-SNE Visualization of Transformer Embeddings: Circle")
    plt.savefig("circle_embeddings.png")
    return None


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
    parser = argparse.ArgumentParser(description="Visualize Transformer Embeddings with t-SNE")
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
    ckpt_iter = args.ckpt_iter
    problem = args.problem
    device = args.device
    temperature = args.temperature
    num_nodes = args.num_nodes
    num_of_paths = args.num_of_paths
    config = args.config

    data_path = f'data/{dataset}/{num_nodes}_{problem}'
    meta_path = f'{data_path}/meta.pkl'
    args = parser.parse_args()

    # Load model
    model = load_custom_gpt_from_checkpoint(args.checkpoint_path)

    # Load meta.pkl (assumed in same directory as checkpoint)
    stoi, itos = load_meta(meta_path)

    # Your sample text
    text = "55 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54"
    # Get embeddings
    embeddings = get_embeddings(text, model, stoi)
    # Split text into tokens
    tokenized_text = text.split(" ")
    #print(len(tokenized_text))
    # Visualize
    #print(embeddings[:len(tokenized_text)])
    visualize_embeddings_tsne(embeddings[:len(tokenized_text)], tokenized_text)
