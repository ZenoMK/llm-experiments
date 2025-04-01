import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from transformers import AutoModel, AutoTokenizer


def get_embeddings(text, model, tokenizer):
    tokens = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=50)
    with torch.no_grad():
        outputs = model(**tokens)
    return outputs.last_hidden_state.squeeze(0).numpy()


def visualize_embeddings_tsne(embeddings, labels):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c='blue', alpha=0.5)

    for i, label in enumerate(labels):
        plt.annotate(label, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]), fontsize=8, alpha=0.7)

    plt.title("t-SNE Visualization of Transformer Embeddings")
    plt.show()


if __name__ == "__main__":
    model_name = "bert-base-uncased"  # Change this to any transformer model
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    text = "This is a sample sentence for visualization purposes."
    embeddings = get_embeddings(text, model, tokenizer)

    tokenized_text = tokenizer.tokenize(text)
    visualize_embeddings_tsne(embeddings[:len(tokenized_text)], tokenized_text)
