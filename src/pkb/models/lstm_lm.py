import torch
import torch.nn as nn

class PredictiveKeyboardLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_layers : int = 1,
        dropout: float = 0.2,
        pad_idx: int = 0
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim = embedding_dim,
            padding_idx = pad_idx
        )

        self.lstm = nn.LSTM(
            input_size = embedding_dim,
            hidden_size = hidden_dim,
            num_layers = num_layers,
            dropout = dropout if num_layers > 1 else 0.0,
            batch_first = True
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #x: (batch_size, seq_length)
        emb = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        lstm_out, _ = self.lstm(emb)  # (batch_size, seq_length, hidden_dim)
        last = lstm_out[:, -1, :]  # (batch_size, hidden_dim)
        logits = self.fc(self.dropout(last))
        return logits  # (batch_size, vocab_size)