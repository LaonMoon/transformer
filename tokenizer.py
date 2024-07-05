# Use sentencepiece tokenizer
# https://github.com/google/sentencepiece

# from datasets import load_dataset
import sentencepiece as spm

class Tokenizer:
    def __init__(self, train_path="train_de_en.json", model_path="sentencepiece.model"):
        self.train_path = train_path
        self.model_path = model_path
        self.tokenizer = spm.SentencePieceProcessor()
        if self.model_path:
            self.load_model(self.model_path)

    def train(self):
        spm.SentencePieceTrainer.train(
            input=self.train_path, 
            model_prefix="sentencepiece", 
            vocab_size=37000,
            character_coverage=1.0,
            model_type='bpe',
        )
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load("sentencepiece.model")
    
    def load_model(self, model_path):
        self.tokenizer.load(model_path)

    def tokenize(self, text, num_samples=10, alpha=0.1, nbest_size=-1):
        if not self.tokenizer:
            raise ValueError("Tokenizer has not been trained yet. Call train() first.")
        
        results = []
        for _ in range(num_samples):
            result = self.tokenizer.encode(text, out_type=str, enable_sampling=True, alpha=alpha, nbest_size=nbest_size)
            results.append(result)
        return results

    def get_vocab_size(self):
        if not self.tokenizer:
            raise ValueError("Tokenizer has not been trained yet. Call train() first.")
        return self.tokenizer.get_piece_size()

    def encode_as_pieces(self, text):
        if not self.tokenizer:
            raise ValueError("Tokenizer has not been trained yet. Call train() first.")
        return self.tokenizer.encode_as_pieces(text)
