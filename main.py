import torch
import torch.nn as nn
import numpy as np
from tokenizer import Tokenizer
from model import Embedding, Multi_head_attention, Feed_forward_network, Encoder

# # tokenizer 

# # If you want to train new version of tokenizer then execute this code
# tokenizer = Tokenizer(train_path="train_de_en.json")
# tokenizer.train()

tokenizer = Tokenizer(train_path="train_de_en.json", model_path="sentencepiece.model")
text = "Hello, This is a test of sentencepiece. Ich komme aus Korea. Schön, dich kennenzulernen."
tokenized_samples = tokenizer.tokenize(text, num_samples=1)

# # If num_samples > 1
# for i, sample in enumerate(tokenized_samples, 1):
#     print(f"Sample {i}: {sample}")

print("Vocabulary size:", tokenizer.get_vocab_size())
encoded_pieces = tokenizer.encode_as_pieces(text)
print("Encoded as pieces:", encoded_pieces)
input_indices = [tokenizer.tokenizer.piece_to_id(piece) for piece in tokenized_samples[0]]
print("Input indices:", input_indices)
print("Input indices shape:",len(input_indices))

# # input embedding 

embedding = Embedding(input_indices)
embedded_input_with_pos = embedding.forward()

print("Embedded input with positional encoding:")
print(embedded_input_with_pos)
print("Shape of embedded input with positional encoding:", embedded_input_with_pos.shape)

# # Multi_head_attention

# head = Multi_head_attention(embedded_input_with_pos)
# head.shape_QKV()
# head.scaled_dot_product_attention()

# attention_output = head.forward()
# print("multi_head Attention output shape:", attention_output.shape)

# output_tensor = Feed_forward_network(attention_output)
# print("Output shape:", output_tensor.forward().shape)

# # Encoder

encoder = Encoder(embedded_input_with_pos)
encoder_output = encoder.forward()
print("multi_head Attention output shape:", encoder_output.shape)

# # Decoder

# # Transformer