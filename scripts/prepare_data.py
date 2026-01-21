import os
import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader

from pkb.data.preprocess import load_text, normalize, tokenize
from pkb.data.vocab import Vocabulary
from pkb.data.dataset import NextWordDataset
from pkb.models.lstm_lm import PredictiveKeyboardLSTM
from pkb.train.train import train_one_epoch
from pkb.train.eval import evaluate

def main():
    context_size = 8
    batch_size = 128
    epochs = 10
    lr = 5e-4
    min_freq = 2
    max_size = 30000

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- paths
    raw_path = r"data\\raw\\sherlock-holm.es_stories_plain-text_advs.txt"
    vocab_path = r"data\\processed\\vocab.json"
    ckpt_path = r"artifacts\\checkpoints\\best_model.pt"

    os.makedirs(r"data\\processed", exist_ok=True)
    os.makedirs(r"artifacts\\checkpoints", exist_ok=True)
    # ---- data preparation
    text = normalize(load_text(raw_path))
    tokens = tokenize(text)

    vocab = Vocabulary(min_freq=min_freq, max_size=max_size)
    vocab.build(tokens)
    vocab.save(vocab_path)

    dataset = NextWordDataset(tokens=tokens, vocab=vocab, context_size=context_size)

    # ---- train/valid split
    n_total = len(dataset)
    n_valid = int(0.1 * n_total)
    n_train = n_total - n_valid
    train_ds, valid_ds = random_split(dataset, [n_train, n_valid])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

    # ---- model
    model = PredictiveKeyboardLSTM(
        vocab_size=vocab.size,
        embedding_dim=64,              # IMPORTANT: must match your model class arg name
        hidden_dim=128,
        num_layers=1,
        dropout=0.3,
        pad_idx=vocab.word2idx["<pad>"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ---- train
    best_valid = float("inf")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        valid_loss = evaluate(model, valid_loader, criterion, device)

        if valid_loss < best_valid:
            best_valid = valid_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab_path": vocab_path,
                    "context_size": context_size,
                    "embed_dim": 64,
                    "hidden_dim": 128,
                    "num_layers": 1,
                    "dropout": 0.3,
                },
                ckpt_path,
            )

        print(
            f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | "
            f"valid_loss={valid_loss:.4f} | best_valid={best_valid:.4f}"
        )

    print(f"\nSaved vocab: {vocab_path}")
    print(f"Saved best model: {ckpt_path}")

if __name__ == "__main__":
    main()
