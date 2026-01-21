# Predictive Keyboard using PyTorch

A PyTorch-based **next-word prediction (predictive keyboard)** model built from scratch.  
The system learns language patterns from text and suggests the **top-3 next words**, similar to a smartphone keyboard.

---

## 📌 What This Project Does

Given a partial sentence like:

i am going to

The model predicts likely next words such as:

be, do, get

This is implemented using a **word-level language model** trained on the Sherlock Holmes corpus.

---

## 🧠 Architecture Overview

- **Tokenization**: word-level (with punctuation handling)
- **Vocabulary**: frequency-based with `<pad>` and `<unk>`
- **Model**: Embedding → LSTM → Linear(vocab_size)

- **Loss**: CrossEntropyLoss
- **Optimizer**: Adam
- **Inference**: Top-K sampling with filters (keyboard-style)

---

## 📁 Project Structure

    ```text
    predictive-keyboard-pytorch/
    ├─ README.md
    ├─ pyproject.toml                  # or requirements.txt (pick one)
    ├─ .gitignore
    ├─ .env.example                    # env vars like WANDB_API_KEY (optional)
    ├─ configs/
    │  ├─ default.yaml                 # hyperparams, paths, model sizes
    │  └─ local.yaml                   # ignored; your machine-specific overrides
    ├─ data/
    │  ├─ raw/
    │  │  └─ sherlock_holmes.txt        # copy your uploaded dataset here
    │  ├─ interim/                     # cleaned text, tokenized files (optional)
    │  └─ processed/
    │     ├─ vocab.json                # stoi/itos, special tokens
    │     ├─ train.pt                  # tensors / indexed sequences
    │     └─ valid.pt
    ├─ notebooks/
    │  └─ 01_explore_data.ipynb         # optional exploration
    ├─ src/
    │  └─ pkb/                         # "predictive keyboard" package
    │     ├─ __init__.py
    │     ├─ utils/
    │     │  ├─ seed.py
    │     │  ├─ logging.py
    │     │  └─ io.py                   # load/save json, torch, text
    │     ├─ data/
    │     │  ├─ preprocess.py           # clean + tokenize
    │     │  ├─ vocab.py                # build vocab + numericalize
    │     │  ├─ dataset.py              # PyTorch Dataset/DataLoader
    │     │  └─ collate.py              # padding + batching
    │     ├─ models/
    │     │  ├─ lstm_lm.py              # Embedding + LSTM + Linear
    │     │  └─ sampling.py             # top-k, temperature, filters
    │     ├─ train/
    │     │  ├─ train.py                # training loop
    │     │  ├─ eval.py                 # perplexity/accuracy
    │     │  └─ checkpoints.py          # save/load checkpoints
    │     └─ inference/
    │        └─ predict.py              # given context -> top-3 suggestions
    ├─ scripts/
    │  ├─ prepare_data.py               # raw -> processed
    │  ├─ train.py                      # calls src/pkb/train/train.py
    │  └─ predict.py                    # CLI for suggestions
    ├─ tests/
    │  ├─ test_vocab.py
    │  ├─ test_dataset.py
    │  └─ test_sampling.py
    ├─ artifacts/
    │  ├─ checkpoints/                  # model.pt, optimizer.pt
    │  └─ runs/                         # logs, metrics
    └─ docs/
    └─ design.md                     # notes: choices, experiments
    ```


## 🚀 How to Run

### 1️⃣ Install dependencies
```bash
pip install torch

2️⃣ Train the model
 python scripts/prepare_data.py 

This will:

build the vocabulary

train the LSTM language model

This will:
- build the vocabulary
- train the LSTM language model
- Save: 
    data/processed/vocab.json
    artifacts/checkpoints/best_model.pt

3️⃣ Run the predictive keyboard
- python scripts/predict.py

Example:
    - Type something: i want to
    - Suggestions: ['go', 'get', 'see']

  - Type quit to exit.



 