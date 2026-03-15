# Word2Vec from Scratch (NumPy)

This repository contains a simple implementation of Word2Vec
(skip-gram with negative sampling) written entirely in NumPy.

The goal of this project was to understand how Word2Vec actually works
under the hood, so everything is implemented manually:
- forward pass 
- loss computation 
- gradient derivation 
- parameter updates

No deep learning frameworks such as PyTorch or TensorFlow are used.

------------------------------------------------------------------------

## Approach

The implementation follows the skip-gram model with negative
sampling.

From a text corpus, the algorithm generates
`(center word, context word)` pairs using a sliding window.\
For every positive pair, several random words are sampled as negative
examples.

The model then learns word embeddings by: 
- increasing similarity between the center word and the true context word 
- decreasing similarity between the center word and randomly sampled words

The loss function optimized during training is:

$$
L = -\log \sigma(u_o^\top v_c) -
\sum_{k=1}^{K} \log \sigma(-u_{n_k}^\top v_c)
$$

where: 
- $v_c$ is the embedding of the center word 
- $u_o$ is the embedding of the true context word 
- $u\_{n_k}$ are embeddings of negative samples

Gradients are computed manually and applied using stochastic gradient
descent.

------------------------------------------------------------------------

## Project structure

    word2vec-numpy/
    │
    ├── word2vec_numpy.py
    ├── README.md
    ├── requirements.txt
    ├── .gitignore
    └── examples/
        └── demo_text.txt

The main implementation is in **`word2vec_numpy.py`**.

------------------------------------------------------------------------

## Running the example

Install dependencies:

    pip install -r requirements.txt

Then run:

    python word2vec_numpy.py

The script trains on a small demo corpus and prints nearest neighbors
for a few words.

Example output:

    Epoch 1/50 - avg loss: 3.41
    Epoch 2/50 - avg loss: 3.08
    Epoch 3/50 - avg loss: 2.77

    Nearest neighbors:
    cat → dog mouse kitten
    dog → cat puppy animal

------------------------------------------------------------------------

## Notes

Possible improvements:

-   vectorized mini-batching
-   subsampling of very frequent words
-   larger training dataset
-   precomputed negative sampling tables
-   implementing the CBOW variant

------------------------------------------------------------------------

## Why two embedding matrices?

Word2Vec learns two representations for each word:

-   W_in - used when the word appears as the center word\
-   W_out - used when the word appears as the context word

During training both matrices are updated, but in practice the final
embeddings are usually taken from W_in.