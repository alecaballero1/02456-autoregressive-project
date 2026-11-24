"""
Evaluate GRU loss
"""

import torch

from protgen.data import IGNORE_INDEX


class EvaluateGru:
    def __init__(self, model, loss_function, device):
        """Set the model, loss function, and device used for evaluation"""
        self.model = model
        self.loss_function = loss_function
        self.device = device

    def evaluate(self, loader):
        """Return the mean token loss without updating model weights"""
        # Switch to evaluation mode and reset loss totals.
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        # Evaluate batches without calculating gradients.
        with torch.no_grad():
            for tokens, targets, _, _ in loader:
                # Move inputs and targets to the model's device
                tokens = tokens.to(self.device)
                targets = targets.to(self.device)

                # Predict next-token scores for each position
                logits, _ = self.model(tokens)

                # Flatten batch and sequence dimensions to calculate token loss
                loss_sum = self.loss_function(
                    logits.reshape(-1, self.model.vocab_size),
                    targets.reshape(-1),
                )

                # Sum loss and count non-padding targets
                total_loss += loss_sum.item()
                total_tokens += (targets != IGNORE_INDEX).sum().item()

        # Average loss over all non-padding targets.
        return total_loss / total_tokens
