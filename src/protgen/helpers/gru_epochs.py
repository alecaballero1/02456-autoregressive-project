import torch
from torch import nn

from protgen.data import IGNORE_INDEX
from protgen.helpers.gru_constants import GRADIENT_CLIP_NORM


def train_epoch(model, loader, loss_function, optimizer, device):
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for tokens, targets, _, _ in loader:
        tokens = tokens.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        logits, _ = model(tokens)

        loss_sum = loss_function(
            logits.reshape(-1, model.vocab_size),
            targets.reshape(-1),
        )

        token_count = (targets != IGNORE_INDEX).sum()
        loss = loss_sum / token_count

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP_NORM)
        optimizer.step()

        total_loss += loss_sum.item()
        total_tokens += token_count.item()

    return total_loss / total_tokens


def evaluate(model, loader, loss_function, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for tokens, targets, _, _ in loader:
            tokens = tokens.to(device)
            targets = targets.to(device)

            logits, _ = model(tokens)

            loss_sum = loss_function(
                logits.reshape(-1, model.vocab_size),
                targets.reshape(-1),
            )

            total_loss += loss_sum.item()
            total_tokens += (targets != IGNORE_INDEX).sum().item()

    return total_loss / total_tokens
