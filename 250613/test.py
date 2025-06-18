import torch
import torch.nn as nn
import torch.optim as optim
from tokenizer import Tokenizer
from model import preprocess_batch, Embedding, Transformer

# [7] 예제 실행

# 예제 문장들
sentences = [
    "Hello, how are you?",
    "Ich komme aus Korea.",
    "Nice to meet you!",
    "Schön, dich kennenzulernen."
]

# tokenizer는 sentencepiece 기반 tokenizer라고 가정
tokenizer = Tokenizer(train_path="../train_de_en.json", model_path="../sentencepiece.model")

# 텍스트 → 토큰 → 인덱스 → 패딩
padded_batch = preprocess_batch(sentences, tokenizer)  # (batch, seq_len)

# 임베딩 + 위치 인코딩
embedding_layer = Embedding(vocab_size=37000, embedding_dim=512, max_len=padded_batch.size(1))
embedded_inputs = embedding_layer(padded_batch)  # (batch, seq_len, embedding_dim)

print("[1] 임베딩 출력 shape:", embedded_inputs.shape)

# # Multi-Head Attention 테스트
# mha = MultiHeadAttention(d_model=512, h=8)
# attn_output = mha(embedded_inputs)  # (batch, seq_len, d_model)
# print("[2] Multi-Head Attention 출력 shape:", attn_output.shape)

# # Encoder Layer 테스트
# encoder_layer = EncoderLayer(d_model=512, h=8)
# enc_out = encoder_layer(embedded_inputs)  # (batch, seq_len, d_model)
# print("[3] Encoder Layer 출력 shape:", enc_out.shape)

# # Encoder 전체 테스트
# encoder = Encoder(d_model=512, h=8, N=6)
# enc_out = encoder(embedded_inputs)
# print("[4] Encoder 출력 shape:", enc_out.shape)

# # Decoder 테스트
# decoder = Decoder(d_model=512, h=8, N=6)
# # Decoder 입력은 임베딩된 입력과 같다고 가정 (teacher forcing)
# decoder_inputs = embedded_inputs

# 마스크 생성
def generate_subsequent_mask(seq_len):
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    return ~mask  # 하삼각만 True

# batch_size = decoder_inputs.size(0)
# seq_len = decoder_inputs.size(1)
# tgt_mask = generate_subsequent_mask(seq_len).to(decoder_inputs.device)
# tgt_mask = tgt_mask.unsqueeze(0).expand(batch_size, -1, -1)  # (B, L, L)

# dec_out = decoder(decoder_inputs, enc_out, tgt_mask=tgt_mask)  # (batch, seq_len, d_model)
# print("[5] Decoder 출력 shape:", dec_out.shape)

transformer = Transformer(vocab_size=37000)

src_ids = padded_batch  # (B, src_len)
tgt_ids = padded_batch  # teacher forcing 가정

seq_len = tgt_ids.size(1)
batch_size = tgt_ids.size(0)
tgt_mask = generate_subsequent_mask(seq_len).to(tgt_ids.device)
tgt_mask = tgt_mask.unsqueeze(0).expand(batch_size, -1, -1)

logits = transformer(src_ids, tgt_ids, tgt_mask=tgt_mask)  # (B, tgt_len, vocab_size)
print("Transformer 출력:", logits.shape)