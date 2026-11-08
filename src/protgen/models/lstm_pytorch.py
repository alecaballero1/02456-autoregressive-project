# This LSTMRNN class was taken from Notebook ‘5.1 Recurrent Neural Networks’ (original class name is "MyRecurrentNet"), which was provided 
# as part of the course materials for Week 5. Modifications have been made to complete the implementation.

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..constants.lstm_constants import VOCAB_SIZE

class LSTMRNN(nn.Module):
    def __init__(self, n_in, n_hid,nb_layers,bias,vocab_size):
        super(LSTMRNN, self).__init__()
        
        # Recurrent layer
        self.lstm = nn.LSTM(input_size=n_in, hidden_size=n_hid, num_layers=nb_layers, bias=bias, batch_first=True)
        
        # Output layer
        self.l_out = nn.Linear(in_features=n_hid,
                            out_features=vocab_size,
                            bias=False)
        
    def forward(self, x):
        # RNN returns output and last hidden state
        x, (h, c) = self.lstm(x)
        
        # Flatten output for feed-forward layer
        x = x.view(-1, self.lstm.hidden_size)
        
        # Output layer
        x = self.l_out(x)
        
        return x
