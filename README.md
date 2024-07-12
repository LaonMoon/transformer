# Transformer Re-implementation

## Requirements

- To be updated

## Introduction

This project is the unofficial implementation of NeurlIPS 2017 paper "Attention is All You Need".

## Data

Use IWSLT17 de-en dataset. The following script downloads datasets.

```$ bash download.sh```

And convert the dataset to JSON files.

```$ python convert_to_json.py```

## Tokenizer

- SentencePiece 
    - de-en dataset
    - vocab size : 37000

## Model

- seq_len = 256
- d_model(embedding_dim) = 512

- Embedding : ```Class Embedding```
- Multi-head attention : ```Class Multi_head_attention```
- Feed-forward network : ```Class Feed_forward_network```

### To-Do

- [ ] Assign different indices to UNK and PAD tokens.
- [ ] Build multi-head by split linear matrices
- [ ] Visualize attention map