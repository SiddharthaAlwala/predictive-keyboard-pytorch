import torch, string
from pkb.data.preprocess import normalize, tokenize, load_text
from pkb.models.sampling import top_k_predictions

PUNCT_TOKENS = set(list(string.punctuation))

def suggest_next_words(model, vocab, text: str, context_size: int = 3, k: int = 3, device: str = "cpu", temperature: float = 1.0, ban_punct: bool = True, ban_repeat_last: bool = True):


    """
    Given an input text, suggest the top_k next words based on the model's predictions.
    """
    model.eval()
    tokens = tokenize(normalize(text))
    if len(tokens) == 0:
        return []
    
    context = tokens[-context_size:]
    if(len(context) < context_size):
        context = ["<pad>"] * (context_size - len(context)) + context

    x = torch.tensor([vocab.encode(context)], dtype = torch.long).to(device)  # (1, context_size)
    with torch.no_grad():
        logits = model(x)  # (1, vocab_size)

    banned = {vocab.word2idx["<pad>"], vocab.word2idx["<unk>"]}
    if ban_punct:
        for t in PUNCT_TOKENS:
            if t in vocab.word2idx:
                banned.add(vocab.word2idx[t])

    if ban_repeat_last:
        last = tokens[-1]
        if last in vocab.word2idx:
            banned.add(vocab.word2idx[last])

    # We may ban many tokens; ask for a bigger pool then filter down
    pool = max(k * 10, 50)
    top = top_k_predictions(logits, k=pool, banned_ids=list(banned), temperature=temperature)

    suggestions = []
    for idx, _ in top:
        w = vocab.idx2word[idx]
        # extra guard: avoid punctuation-like tokens
        if ban_punct and w in PUNCT_TOKENS:
            continue
        suggestions.append(w)
        if len(suggestions) == k:
            break

    return suggestions