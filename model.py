import torch
import torch.nn as nn
import numpy as np
import tokenizer

text = "Hello, how are you? ich bin Max."
tokenizer = tokenizer.tokenizer()
input_indices = tokenizer.encode(text)

def embedding_layer(input_size, input_indices):
    embedding_layer = nn.Embedding(input_size, embedding_dim, padding_idx=1)

    input_indices = torch.LongTensor(input_indices)
    print(len(input_indices))
    embedded_input = embedding_layer(input_indices)

    return embedded_input

def positional_encoding(seq_len, d_model):
    pos_enc = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            pos_enc[pos, i] = np.sin(pos / (10000 ** ((2 * i)/d_model)))
            pos_enc[pos, i+1] = np.cos(pos / (10000 ** ((2 * (i+1))/d_model)))
    return pos_enc

input_size = 37000
seq_length = 128 # max_len에 따름
embedding_dim = 512
pos_encoding = positional_encoding(seq_length, embedding_dim)

print("Positional Encoding Shape:", pos_encoding.shape)

# input embedding과 positional encoding을 더함
pos_encoding = torch.FloatTensor(pos_encoding)
embedded_input = embedding_layer(input_size, input_indices)
pos_encoding = pos_encoding[:embedded_input.shape[0], :]
embedded_input_with_pos = embedded_input + pos_encoding.unsqueeze(0)

print("Input text:", text)
print("Input indices:", input_indices)

print("Embedded input:", embedded_input)
print("Shape of embedded input:", embedded_input.shape)

print("Embedded input with positional encoding:", embedded_input_with_pos)
print("Shape of embedded input with positional encoding:", embedded_input_with_pos.shape)