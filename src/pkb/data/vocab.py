from collections import Counter
from typing import Dict, List
import json
from pathlib import Path

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
class Vocabulary:
    def __init__(self, min_freq: int = 1, max_size: int|None = None):
        self.min_freq = min_freq
        self.max_size = max_size

        self.word2idx :Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.fitted = False

    def build(self, tokens: List[str]):
        counter = Counter(tokens)

        #filter by frequency
        words = [word for word, count in counter.items() if count >= self.min_freq]

        #sort by frequency
        words.sort(key=lambda word: counter[word], reverse=True)

        if self.max_size:
            words = words[: self.max_size]

        #add special tokens first
        vocab = [PAD_TOKEN, UNK_TOKEN] + words

        self.word2idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.fitted = True

    def encode(self, tokens: List[str])-> List[int]:
        if not self.fitted:
            raise RuntimeError("Vocabulary not built. Call 'build' method first.")
        return [
            self.word2idx.get(token, self.word2idx[UNK_TOKEN]) 
            for token in tokens
        ]
    def decode(self, indices: List[int]) -> List[str]:
        return [self.idx2word[idx] for idx in indices]
    
    @property
    def size(self) -> int:
        return len(self.word2idx)
    
    def save(self, path: str):
        if not self. fitted:
            raise RuntimeError("Vocabulary not built. Call 'build' method first.")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "min_freq": self.min_freq,
            "max_size": self.max_size,
            "word2idx": self.word2idx
        }

        path.write_text(json.dumps(payload, indent = 2), encoding="utf-8")

    @classmethod
    def load(cls, path:str):
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))

        vocab = cls(min_freq = payload.get("min_freq", 1), max_size = payload.get("max_size"))
        vocab.word2idx = payload["word2idx"]
        vocab.idx2word = {int(i): w for w, i in vocab.word2idx.items()}
        vocab.fitted = True
        return vocab

    