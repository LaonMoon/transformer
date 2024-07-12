import torch
import torch.nn as nn
import numpy as np

class Embedding:
    def __init__(self, input_indices, input_size = 37000, seq_length = 256, embedding_dim = 512):
        self.input_size = input_size # vocab size 
        self.seq_length = seq_length # set arbitrarily
        self.embedding_dim = embedding_dim # d_model
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
    
class Multi_head_attention:

    def __init__(self, input, embedding_dim = 512, h = 8):
        self.input = input
        self.d_model = embedding_dim
        self.h = h
        self.d_k = self.d_model//self.h
        self.d_v = self.d_model//self.h
        # linear transform
        self.W_Q = nn.Linear(self.d_model, self.d_k)
        self.W_K = nn.Linear(self.d_model, self.d_k)
        self.W_V = nn.Linear(self.d_model, self.d_v)
        # Query, Key, Value
        self.Q = self.W_Q(self.input)
        self.K = self.W_K(self.input)
        self.V = self.W_V(self.input)
        # W_O
        self.W_O = nn.Linear(self.d_model, self.d_model)

    def shape_QKV(self):
        # Shape check
        print("Q shape:", self.Q.shape)
        print("K shape:", self.K.shape)
        print("V shape:", self.V.shape)

    def scaled_dot_product_attention(self):
        Q = self.Q
        K = self.K
        V = self.V
        # Q * K^T
        Q_KT = torch.matmul(Q, K.transpose(-2, -1))
        print("Q * K^T shape:", Q_KT.shape)
        # Compute sqrt(d_k)
        sqrt_dk = torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        print("sqrt(d_k):", sqrt_dk)
        # Normalize Q_KT
        Q_KT_normalized = Q_KT / sqrt_dk
        print("Normalized Q * K^T shape:", Q_KT_normalized.shape)
        # Softmax(Q_KT_normalized)
        attention_weights = torch.softmax(Q_KT_normalized, dim=-1)
        print("Attention weights shape:", attention_weights.shape)
        # Weighted V
        output = torch.matmul(attention_weights, V)
        print("Attention output shape:", output.shape) 

        return output
    
    def forward(self):
        # multi-head attention
        attention_outputs = []
        for i in range(self.h):
            single_head = self.scaled_dot_product_attention()
            attention_outputs.append(single_head)
        concatenated_heads = torch.cat(attention_outputs, dim=-1)
        print("Concat heads shape:", concatenated_heads.shape)
        # linear transformation
        output = self.W_O(concatenated_heads)

        return output