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
        embedding_layer = nn.Embedding(self.input_size, self.embedding_dim, padding_idx=0) # padding_idx : index of pad token
        embedded_input = embedding_layer(self.input_indices)
        return embedded_input

    def positional_encoding(self, seq_length, embedding_dim):
        pos_encoding = np.zeros((seq_length, embedding_dim))
        for pos in range(seq_length):
            for i in range(0, embedding_dim, 2):
                pos_encoding[pos, i] = np.sin(pos / (10000 ** ((2 * i)/embedding_dim)))
                pos_encoding[pos, i+1] = np.cos(pos / (10000 ** ((2 * (i+1))/embedding_dim)))
        return pos_encoding
    
    def forward(self):
        pos_encoding = self.positional_encoding(self.seq_length, self.embedding_dim)
        pos_encoding = torch.FloatTensor(pos_encoding)

        embedded_input = self.embedding_layer()
        pos_encoding = pos_encoding[:embedded_input.shape[0], :] # adjust for input sequence length

        #print(pos_encoding.shape)
        #print(embedded_input.shape)

        # embedded_input_with_pos = embedded_input + pos_encoding.unsqueeze(0)
        embedded_input_with_pos = embedded_input + pos_encoding
        return embedded_input_with_pos
    
# class dot_product_attention:
# class multi_head_attention: