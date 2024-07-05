import torch
import torch.nn as nn
import numpy as np

class Embedding:
    def __init__(self, input_indices):
        self.input_size = 37000 # vocab size 
        self.seq_length = 256 # set arbitrarily
        self.embedding_dim = 512 # d_model
        self.input_indices = torch.LongTensor(input_indices)

    def embedding_layer(self):
        embedding_layer = nn.Embedding(self.input_size, self.embedding_dim, padding_idx=1)
        embedded_input = embedding_layer(self.input_indices)

        return embedded_input
    
    def positional_encoding(self, seq_len, d_model):
        pos_enc = np.zeros((seq_len, d_model))
        for pos in range(seq_len):
            for i in range(0, d_model, 2):
                pos_enc[pos, i] = np.sin(pos / (10000 ** ((2 * i)/d_model)))
                pos_enc[pos, i+1] = np.cos(pos / (10000 ** ((2 * (i+1))/d_model)))
        return pos_enc
    
    def forward(self):
        pos_encoding = self.positional_encoding(self.seq_length, self.embedding_dim)

        # print("Positional Encoding Shape:", pos_encoding.shape)

        # Add input embedding and positional encoding.
        pos_encoding = torch.FloatTensor(pos_encoding)
        embedded_input = self.embedding_layer()
        pos_encoding = pos_encoding[:embedded_input.shape[0], :]
        embedded_input_with_pos = embedded_input + pos_encoding.unsqueeze(0)
        
        return embedded_input_with_pos