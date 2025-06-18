import sentencepiece as spm
import os

# dataset = load_dataset("iwslt2017", 'iwslt2017-de-en')
# texts = dataset['train']['translation']
# with open("/content/drive/MyDrive/Colab Notebooks/transformer/dataset.txt", "w", encoding="utf-8") as f:
#     for text in texts:
#         f.write(str(text))
#         f.write("\n")

spm.SentencePieceTrainer.train(
    input="dataset.txt",
    model_prefix="sentencepiece_model",
    vocab_size=37000,
    pad_id=0,                   # ✅ 패딩 토큰 ID 지정
    pad_piece="<pad>",         # ✅ 패딩 토큰 문자 지정
    unk_id=1,
    bos_id=2,
    eos_id=3
)

tokenizer = spm.SentencePieceProcessor()
tokenizer.load("sentencepiece_model.model")

# test
text = "안녕하세요, SentencePiece를 테스트합니다."
tokens = tokenizer.encode_as_pieces(text)
print(tokens)