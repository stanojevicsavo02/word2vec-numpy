import re
import numpy as np
from collections import Counter


# the first step is tokenization
# it can be done in several ways, but I chose word-level for the simplicity


def tokenize(text):
    text = text.lower()
    # keep only letters/numbers and simple apostrophes
    tokens = re.findall(r"\b[a-z0-9']+\b", text)
    return tokens


# now we should build our vocabulary from computed tokens


def build_vocab(tokens, min_count=1):
    counter = Counter(tokens)

    vocab_words = [word for word, count in counter.items() if count >= min_count]
    vocab_words.sort()

    word_to_idx = {word: i for i, word in enumerate(vocab_words)}
    idx_to_word = {i: word for word, i in word_to_idx.items()}

    encoded_tokens = [word_to_idx[word] for word in tokens if word in word_to_idx]
    word_counts = np.array([counter[word] for word in vocab_words], dtype=np.int64)

    return word_to_idx, idx_to_word, encoded_tokens, word_counts


# generating skip-gram token pairs is next


def generate_skipgram_pairs(encoded_tokens, window_size=2):
    pairs = []

    for i, center_word in enumerate(encoded_tokens):
        left = max(0, i - window_size)
        right = min(len(encoded_tokens), i + window_size + 1)

        for j in range(left, right):
            if j == i:
                continue
            context_word = encoded_tokens[j]
            pairs.append((center_word, context_word))

    return pairs


# now we start building the neural network
# firstly, we need to initialize the embeddings


def initialize_embeddings(vocab_size, embed_dim):
    W_in = 0.01 * np.random.randn(vocab_size, embed_dim)
    W_out = 0.01 * np.random.randn(vocab_size, embed_dim)
    return W_in, W_out


def sigmoid(x):
    x = np.clip(x, -10, 10)
    return 1 / (1 + np.exp(-x))


def build_negative_sampling_distribution(word_counts):
    probs = word_counts.astype(np.float64) ** 0.75
    probs /= probs.sum()
    return probs


def sample_negative_indices(probs, num_negatives, forbidden_indices=None):
    if forbidden_indices is None:
        forbidden_indices = set()

    vocab_size = len(probs)
    negatives = []

    while len(negatives) < num_negatives:
        idx = np.random.choice(vocab_size, p=probs)
        if idx not in forbidden_indices:
            negatives.append(idx)

    return np.array(negatives, dtype=np.int64)


# computing the loss


def negative_sampling_loss(center_idx, context_idx, neg_indices, W_in, W_out):
    v_c = W_in[center_idx]
    u_o = W_out[context_idx]
    U_neg = W_out[neg_indices]

    pos_score = np.dot(u_o, v_c)
    neg_scores = U_neg @ v_c

    pos_loss = -np.log(sigmoid(pos_score) + 1e-10)
    neg_loss = -np.sum(np.log(sigmoid(-neg_scores) + 1e-10))

    loss = pos_loss + neg_loss
    return loss


def train_one_example(center_idx, context_idx, neg_indices, W_in, W_out, lr):
    # Copy vectors so gradients are computed using the original values
    v_c = W_in[center_idx].copy()      # shape (d,)
    u_o = W_out[context_idx].copy()    # shape (d,)
    U_neg = W_out[neg_indices].copy()  # shape (K, d)

    # Forward pass
    pos_score = np.dot(u_o, v_c)       # scalar
    neg_scores = U_neg @ v_c           # shape (K,)

    pos_sig = sigmoid(pos_score)       # scalar
    neg_sig = sigmoid(neg_scores)      # shape (K,)

    # Loss
    loss = -np.log(pos_sig + 1e-10) - np.sum(np.log(sigmoid(-neg_scores) + 1e-10))

    # Gradients
    # d/dx[-log(sigmoid(x))] = sigmoid(x) - 1
    grad_pos = pos_sig - 1.0           # scalar

    # d/dx[-log(sigmoid(-x))] = sigmoid(x)
    grad_neg = neg_sig                 # shape (K,)

    # Gradient w.r.t. center embedding
    grad_v = grad_pos * u_o + np.sum(grad_neg[:, None] * U_neg, axis=0)

    # Gradient w.r.t. positive output embedding
    grad_u_o = grad_pos * v_c

    # Gradient w.r.t. negative output embeddings
    grad_U_neg = grad_neg[:, None] * v_c[None, :]

    # SGD updates
    W_in[center_idx] -= lr * grad_v
    W_out[context_idx] -= lr * grad_u_o

    for i, neg_idx in enumerate(neg_indices):
        W_out[neg_idx] -= lr * grad_U_neg[i]

    return loss


def train_word2vec_skipgram(
    encoded_tokens,
    word_counts,
    embed_dim=50,
    window_size=2,
    num_negatives=5,
    lr=0.025,
    epochs=5,
    seed=42
):
    np.random.seed(seed)

    vocab_size = len(word_counts)
    pairs = generate_skipgram_pairs(encoded_tokens, window_size)
    neg_probs = build_negative_sampling_distribution(word_counts)

    W_in, W_out = initialize_embeddings(vocab_size, embed_dim)

    for epoch in range(epochs):
        np.random.shuffle(pairs)
        total_loss = 0.0

        for center_idx, context_idx in pairs:
            neg_indices = sample_negative_indices(
                probs=neg_probs,
                num_negatives=num_negatives,
                forbidden_indices={context_idx}
            )

            loss = train_one_example(
                center_idx=center_idx,
                context_idx=context_idx,
                neg_indices=neg_indices,
                W_in=W_in,
                W_out=W_out,
                lr=lr
            )

            total_loss += loss

        avg_loss = total_loss / len(pairs)
        print(f"Epoch {epoch + 1}/{epochs} - avg loss: {avg_loss:.4f}")

    return W_in, W_out


def cosine_similarity(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10
    return np.dot(a, b) / denom


def nearest_neighbors(word, word_to_idx, idx_to_word, embeddings, top_k=5):
    if word not in word_to_idx:
        raise ValueError(f"Word '{word}' not in vocabulary.")

    query_idx = word_to_idx[word]
    query_vec = embeddings[query_idx]

    sims = []
    for idx in range(len(embeddings)):
        if idx == query_idx:
            continue
        sim = cosine_similarity(query_vec, embeddings[idx])
        sims.append((idx_to_word[idx], sim))

    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:top_k]


def load_text_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


text = load_text_file("examples/demo_text.txt")




tokens = tokenize(text)
word_to_idx, idx_to_word, encoded_tokens, word_counts = build_vocab(tokens, min_count=1)

W_in, W_out = train_word2vec_skipgram(
    encoded_tokens=encoded_tokens,
    word_counts=word_counts,
    embed_dim=20,
    window_size=2,
    num_negatives=3,
    lr=0.01,
    epochs=1000,
    seed=42
)

print("\nNearest neighbors using W_in:")
for test_word in ["learning", "language", "word", "vectors", "transformers"]:
    if test_word in word_to_idx:
        neighbors = nearest_neighbors(test_word, word_to_idx, idx_to_word, W_in, top_k=3)
        print(f"{test_word}: {neighbors}")