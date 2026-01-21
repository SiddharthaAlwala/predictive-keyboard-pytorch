import torch
from pkb.data.vocab import Vocabulary
from pkb.models.lstm_lm import PredictiveKeyboardLSTM
from pkb.inference.predict import suggest_next_words

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vocab_path = r"data\\processed\\vocab.json"
    ckpt_path = r"artifacts\\checkpoints\\best_model.pt"

    vocab = Vocabulary.load(vocab_path)
    ckpt = torch.load(ckpt_path, map_location=device)

    model = PredictiveKeyboardLSTM(
        vocab_size=vocab.size,
        embedding_dim=ckpt["embed_dim"],
        hidden_dim=ckpt["hidden_dim"],
        num_layers=ckpt["num_layers"],
        dropout=ckpt["dropout"],
        pad_idx=vocab.word2idx["<pad>"],
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    context_size = ckpt["context_size"]

    print("Predictive Keyboard ready ")
    print(f"Using context_size={context_size}, vocab_size={vocab.size}, device={device}")

    while True:
        text_in = input("\nType something (or 'quit'): ").strip()
        if text_in.lower() == "quit":
            break

        suggestions = suggest_next_words(
            model, vocab, text_in,
            context_size=context_size,
            k=3,
            device=device,
            temperature=1.0,
            ban_punct=True,
            ban_repeat_last=True
        )
        print("Suggestions:", suggestions)

if __name__ == "__main__":
    main()
