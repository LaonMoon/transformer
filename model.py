import torch
import torch.nn as nn
import numpy as np
from tokenizer import Tokenizer

class Embedding:
    def __init__(self, input_indices):
        self.input_size = 37000
        self.seq_length = 128 # max_len에 따름
        self.embedding_dim = 512
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

        # print("Input text:", text)
        # print("Input indices:", input_indices)

        # print("Embedded input:", embedded_input)
        # print("Shape of embedded input:", embedded_input.shape)

        # print("Embedded input with positional encoding:", embedded_input_with_pos)
        # print("Shape of embedded input with positional encoding:", embedded_input_with_pos.shape)
        
        return embedded_input_with_pos

# tokenizer = Tokenizer(train_path="train_de_en.json")
# tokenizer.train()

tokenizer_instance = Tokenizer(train_path="train_de_en.json", model_path="sentencepiece.model")
text = "Hello, This is a test of sentencepiece. Ich komme aus Korea. Schön, dich kennenzulernen."
tokenized_samples = tokenizer_instance.tokenize(text, num_samples=3)

for i, sample in enumerate(tokenized_samples, 1):
    print(f"Sample {i}: {sample}")

print("Vocabulary size:", tokenizer_instance.get_vocab_size())
encoded_pieces = tokenizer_instance.encode_as_pieces(text)
print("Encoded as pieces:", encoded_pieces)

# 인코딩된 텍스트를 임베딩에 전달
# 여기서는 첫 번째 샘플을 사용
input_indices = [tokenizer_instance.tokenizer.piece_to_id(piece) for piece in tokenized_samples[0]]
embedding = Embedding(input_indices)
embedded_input_with_pos = embedding.forward()

print("Embedded input with positional encoding:")
print(embedded_input_with_pos)
print("Shape of embedded input with positional encoding:", embedded_input_with_pos.shape)