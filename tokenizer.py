# Use sentencepiece tokenizer

from datasets import load_dataset
import sentencepiece as spm

spm.SentencePieceTrainer.train(input="train_de_en.json", model_prefix="sentencepiece_model", vocab_size=37000)
tokenizer = spm.SentencePieceProcessor()
tokenizer.load("sentencepiece_model.model")

# test
text = "Hello, This is test of sentencepiece. Ich komme aus Korea."
tokens = tokenizer.encode_as_pieces(text)
print(tokens)

# vocab size
vocab_size = tokenizer.get_piece_size()
print("Vocabulary size:", vocab_size)

print("Sample tokens:")
for i in range(10):
    print(tokenizer.id_to_piece(i))