"""
GRU next-token model 

Takes inspiration from course notebook exercises
Other sources:
- https://docs.pytorch.org/docs/2.15/generated/torch.nn.GRU.html
"""

from torch import nn
import torch.nn.functional as nn_func

from protgen.helpers.gru_constants import NUM_GRU_LAYERS, GRU_SIZE, GRU_DROPOUT

class Gru(nn.Module):
    def __init__(self, vocab_size, hidden_size=GRU_SIZE, num_layers=NUM_GRU_LAYERS):
        """Build the recurrent layer and vocabulary output projection."""
        super().__init__()

        self.vocab_size = vocab_size

        # create hidden layers
        self.rnn_layer = nn.GRU(
            input_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=GRU_DROPOUT,
            batch_first=True,
            bidirectional=False
        )

        # create output layer
        self.output_layer = nn.Linear(in_features=hidden_size, out_features=vocab_size)

    def forward(self, tokens, hidden=None):
        """Return next-token logits and the updated hidden state."""
        # Encode token IDs as one-hot vectors
        inputs = nn_func.one_hot(tokens, num_classes=self.vocab_size).float()

        # run through the hidden layers    
        states, final_hidden = self.rnn_layer(inputs, hidden)

        # run through the output layers and return logits
        logits = self.output_layer(states)

        return logits, final_hidden
