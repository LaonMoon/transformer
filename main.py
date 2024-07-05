import torch
import torch.nn as nn
import numpy as np
from tokenizer import Tokenizer
from model import Embedding

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