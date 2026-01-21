import torch
from torch.nn.utils import clip_grad_norm_

def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip = 1.0):
    model.train()
    total_loss = 0.0
    total_items = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none = True)
        logits = model(x)
        loss = criterion(logits, y)

        loss.backward()
        clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_items += batch_size

    return total_loss / total_items
