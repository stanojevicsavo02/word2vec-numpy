# Word2Vec in Pure NumPy

This repository contains a from-scratch implementation of the core training loop of Word2Vec (skip-gram with negative sampling) using only NumPy.

The goal of the project is educational clarity: all major components of the optimization procedure are implemented manually, including:

- forward pass
- negative sampling loss
- gradient computation
- stochastic gradient descent updates

No deep learning frameworks such as PyTorch or TensorFlow are used.

---

## Method

I implemented the **skip-gram** variant of Word2Vec with **negative sampling**.

For each positive `(center, context)` pair extracted from a corpus, the model samples several negative words and optimizes the objective:

$
\[L = -\log \sigma(u_o^\top v_c) - \sum_{k=1}^{K}\log \sigma(-u_{n_k}^\top v_c)\]
$

where:

- $v_c$ is the input embedding of the center word
- $u_o$ is the output embedding of the true context word
- $u_{n_k}$ are the output embeddings of sampled negative words

The gradients are derived manually and applied with SGD.

---

## Repository structure

```text
word2vec-numpy/
├── README.md
├── requirements.txt
├── .gitignore
├── word2vec_numpy.py
└── examples/
    └── demo_text.txt