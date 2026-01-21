import torch
from torch.utils.data import Dataset
from typing import List
from pkb.data.vocab import Vocabulary

class NextWordDataset(Dataset):
    def __init__(
            self, 
            tokens: List[str],
            vocab: Vocabulary,
            context_size: int 
    ):
        self. tokens = tokens
        self.vocab = vocab
        self.context_size = context_size  

    def __len__(self):
        return len(self.tokens) - self.context_size
    
    def __getitem__(self, idx):
        context = self. tokens[idx: idx+self.context_size]
        target = self.tokens[idx+self.context_size]

        context_ids = torch.tensor(
            self.vocab.encode(context),
            dtype = torch.long
        )

        target_id = torch.tensor(
            self.vocab.word2idx.get(target, self.vocab.word2idx["<unk>"]),
            dtype=torch.long
        )

        return context_ids, target_id
    
