# SentencePiece wrapper
class SentencePieceTokenizer:
    def __init__(self, model_path):
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load(model_path)

    def encode(self, text):
        return self.tokenizer.encode(text, out_type=int)

    def decode(self, ids):
        return self.tokenizer.decode(ids)

    def pad_id(self):
        return self.tokenizer.pad_id()
    
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import sentencepiece as spm
import torch
from torch.nn.utils.rnn import pad_sequence
import os

# Custom dataset
class TranslationDataset(Dataset):
    def __init__(self, data, tokenizer, pad_idx=0):
        self.data = data
        self.tokenizer = tokenizer
        self.pad_idx = pad_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src = self.data[idx]['de']
        tgt = self.data[idx]['en']
        src_ids = torch.tensor(self.tokenizer.encode(src), dtype=torch.long)
        tgt_ids = torch.tensor(self.tokenizer.encode(tgt), dtype=torch.long)
        return src_ids, tgt_ids

def collate_fn(batch, pad_idx=0):
    src_batch, tgt_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_idx)
    return src_padded, tgt_padded

# Load IWSLT dataset (limit to small subset)
dataset = load_dataset("iwslt2017", "iwslt2017-en-de", trust_remote_code=True)
train_data = dataset["train"]["translation"]
#train_data = train_data_full[:3000]  # 메모리 절약을 위해 일부만 사용

# Tokenizer & dataset loader
tokenizer = SentencePieceTokenizer("sentencepiece_model.model")
pad_idx = tokenizer.pad_id()

translation_dataset = TranslationDataset(train_data, tokenizer, pad_idx=pad_idx)
dataloader = DataLoader(translation_dataset, batch_size=16, shuffle=True, collate_fn=lambda x: collate_fn(x, pad_idx=pad_idx))

# 테스트: 첫 배치 확인
for src_batch, tgt_batch in dataloader:
    print("SRC batch shape:", src_batch.shape)
    print("TGT batch shape:", tgt_batch.shape)
    break

import torch
import torch.nn as nn
import torch.optim as optim
from model import Transformer, generate_subsequent_mask
from tokenizer import Tokenizer  # sentencepiece wrapper
from torch.nn.utils.rnn import pad_sequence

# 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 토크나이저 불러오기
tokenizer = Tokenizer(model_path="sentencepiece_model.model")
pad_idx = tokenizer.pad_id()

# 모델, 옵티마이저, 손실함수
model = Transformer(vocab_size=tokenizer.get_vocab_size() + 10, pad_idx=pad_idx).to(device)
model.load_state_dict(torch.load("transformer_model.pth"))
model.train()  # 학습 모드 전환
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

# 학습 파라미터
num_epochs = 7

# 실제 학습 루프
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for src_batch, tgt_batch in dataloader:
        src_batch = src_batch.to(device)
        tgt_batch = tgt_batch.to(device)

        tgt_input = tgt_batch[:, :-1]
        tgt_output = tgt_batch[:, 1:]

        # 마스크 생성
        seq_len = tgt_input.size(1)
        batch_size = tgt_input.size(0)
        tgt_mask = generate_subsequent_mask(seq_len).to(device)
        tgt_mask = tgt_mask.unsqueeze(0).expand(batch_size, -1, -1)

        # 모델 forward
        logits = model(src_batch, tgt_input, tgt_mask=tgt_mask)

        # 손실 계산
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            tgt_output.reshape(-1)
        )

        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}")

# 저장 경로
model_save_path = "transformer_model.pth"

# state_dict 저장
torch.save(model.state_dict(), model_save_path)
print(f"✅ 모델이 저장되었습니다: {model_save_path}")