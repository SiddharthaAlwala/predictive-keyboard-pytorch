import torch


@torch.no_grad()
def evaluate(model, loader, criterian, device):
    model.eval()
    total_loss = 0.0
    total_items = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterian(logits, y)

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_items += batch_size

    return total_loss / total_items