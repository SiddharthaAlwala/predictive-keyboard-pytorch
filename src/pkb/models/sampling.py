import torch
from typing import List, Tuple

def top_k_predictions(
        logits: torch.Tensor,
        k: int,
        banned_ids: List[int] | None = None,
        temperature: float = 1.0,
    ) -> List[Tuple[int, float]]:
    """
    logits: (vocab_size,) or (1, vocab_size)
    returns: list of(token_id, prob)
    """
    if logits.dim() == 2:
        logits = logits.squeeze(0)
    
    if temperature <= 0:
        raise ValueError("Temperature must be > 0")
    
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)  # (vocab_size,)
    if banned_ids:
        probs[banned_ids] = 0.0
        probs = probs / probs.sum()  # re-normalize
    
    top_probs, top_ids = torch.topk(probs, k=k)
    return list(zip(top_ids.tolist(), top_probs.tolist())
)