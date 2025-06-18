import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import math

# [1] Tokenizer 결과를 숫자로 바꾸고 패딩하는 함수
def preprocess_batch(sentences, tokenizer, pad_token_id=0):
    tokenized_batch = [tokenizer.tokenize(sentence, num_samples=1)[0] for sentence in sentences]
    indexed_batch = [
        torch.tensor([tokenizer.tokenizer.piece_to_id(piece) for piece in tokens], dtype=torch.long)
        for tokens in tokenized_batch
    ]
    padded_batch = pad_sequence(indexed_batch, batch_first=True, padding_value=pad_token_id)
    return padded_batch  # shape: (batch_size, seq_len)

# [2] Embedding 클래스 (토큰 + 위치 인코딩)
class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim=512, max_len=256, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.embedding_dim = embedding_dim
        self.register_buffer("pos_encoding", self._make_pos_encoding(max_len, embedding_dim))

    def _make_pos_encoding(self, max_len, d_model):
        pe = np.zeros((max_len, d_model))
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                angle = pos / (10000 ** (i / d_model))
                pe[pos, i] = np.sin(angle)
                if i + 1 < d_model:
                    pe[pos, i + 1] = np.cos(angle)
        return torch.FloatTensor(pe)  # numpy → torch 텐서로 변환

    # torch 버전
    # def _make_pos_encoding(self, max_len, d_model):
    #     pe = torch.zeros(max_len, d_model)
    #     position = torch.arange(0, max_len).unsqueeze(1)
    #     div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    #     pe[:, 0::2] = torch.sin(position * div_term)
    #     pe[:, 1::2] = torch.cos(position * div_term)
    #     return pe  # shape: (max_len, d_model)

    def forward(self, input_ids):
        """
        input_ids: (batch_size, seq_len)
        """
        token_emb = self.embedding(input_ids)  # (batch, seq_len, embed_dim)
        seq_len = input_ids.size(1)
        pos_emb = self.pos_encoding[:seq_len, :].unsqueeze(0)  # (1, seq_len, embed_dim)
        return token_emb + pos_emb

# [3] scaled dot product attention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, h=8):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0

        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        return output

    def forward(self, x, context=None, mask=None):
        # context: 다른 입력 (encoder output 등). 없으면 self-attention
        """
        x: (B, L, d_model) → query
        context: (B, S, d_model) → key/value. None이면 self-attention
        mask: (B, L, S) → optional attention mask
        """
        if context is None:
            context = x
        
        B, L, _ = x.size()
        S = context.size(1)  # context 길이
        
        Q = self.W_Q(x)       # (B, L, d_model)
        K = self.W_K(context) # (B, S, d_model)
        V = self.W_V(context) # (B, S, d_model)

        # Reshape to multi-head (B, h, L/S, d_k)
        Q = Q.view(B, L, self.h, self.d_k).transpose(1, 2)  # (B, h, L, d_k)
        K = K.view(B, S, self.h, self.d_k).transpose(1, 2)  # (B, h, S, d_k)
        V = V.view(B, S, self.h, self.d_k).transpose(1, 2)  # (B, h, S, d_k)

        if mask is not None:
            # mask: (B, L, S) → reshape to (B, 1, L, S) for broadcasting over heads
            mask = mask.unsqueeze(1)  # (B, 1, L, S)

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask=mask)  # (B, h, L, d_k)

        # (B, h, L, d_k) → (B, L, d_model)
        attn_output = attn_output.transpose(1, 2).reshape(B, L, self.d_model)

        return self.out_proj(attn_output)

# [4] Encoder layer
class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, h=8, ff_hidden=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model=d_model, h=h)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_hidden),
            nn.ReLU(),
            nn.Linear(ff_hidden, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # [1] Multi-Head Attention + Residual + Norm
        attn_output = self.self_attn(x)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        # [2] FeedForward + Residual + Norm
        ff_output = self.ffn(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)
        return x
    
# [5] Encoder class
class Encoder(nn.Module):
    def __init__(self, d_model=512, h=8, N=6):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model=d_model, h=h) for _ in range(N)
        ])
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# [6] Decoder layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model=512, h=8, ff_hidden=2048, dropout=0.1):
        super().__init__()
        # [1] Masked Self-Attention
        self.self_attn = MultiHeadAttention(d_model=d_model, h=h)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # [2] Encoder-Decoder Attention
        self.enc_dec_attn = MultiHeadAttention(d_model=d_model, h=h)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        # [3] Feed Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_hidden),
            nn.ReLU(),
            nn.Linear(ff_hidden, d_model)
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_output, tgt_mask=None):
        # x: (B, L, d_model)
        # enc_output: (B, S, d_model)
        # tgt_mask: (B, L, L)

        # [1] Masked Self-Attention
        _x = self.self_attn(x, mask=tgt_mask)
        x = x + self.dropout1(_x)
        x = self.norm1(x)

        # [2] Encoder-Decoder Attention
        _x = self.enc_dec_attn(x, context=enc_output)
        x = x + self.dropout2(_x)
        x = self.norm2(x)

        # [3] Feed Forward Network
        _x = self.ffn(x)
        x = x + self.dropout3(_x)
        x = self.norm3(x)

        return x

class Decoder(nn.Module):
    def __init__(self, d_model=512, h=8, N=6, ff_hidden=2048, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model=d_model, h=h, ff_hidden=ff_hidden, dropout=dropout)
            for _ in range(N)
        ])

    def forward(self, x, enc_output, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask=tgt_mask)
        return x
    
class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, h=8, N=6, ff_hidden=2048, dropout=0.1, pad_idx=0, max_len=256):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model, max_len=max_len, pad_idx=pad_idx)
        self.encoder = Encoder(d_model=d_model, h=h, N=N)
        self.decoder = Decoder(d_model=d_model, h=h, N=N)
        self.generator = nn.Linear(d_model, vocab_size)  # d_model → vocab_size

    def forward(self, src_input_ids, tgt_input_ids, tgt_mask=None):
        """
        src_input_ids: (B, src_len) - encoder input
        tgt_input_ids: (B, tgt_len) - decoder input
        tgt_mask: (B, tgt_len, tgt_len) - subsequent mask for decoder
        """
        # 1. Embedding
        enc_embed = self.embedding(src_input_ids)  # (B, src_len, d_model)
        dec_embed = self.embedding(tgt_input_ids)  # (B, tgt_len, d_model)
        # 2. Encoder
        enc_output = self.encoder(enc_embed)  # (B, src_len, d_model)
        # 3. Decoder (with encoder output and tgt_mask)
        dec_output = self.decoder(dec_embed, enc_output, tgt_mask=tgt_mask)  # (B, tgt_len, d_model)
        # 4. Output projection to vocab logits
        logits = self.generator(dec_output)  # (B, tgt_len, vocab_size)
        return logits
    
def generate_subsequent_mask(seq_len):
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    return ~mask  # 하삼각만 True