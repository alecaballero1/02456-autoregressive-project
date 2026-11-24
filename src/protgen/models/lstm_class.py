# This LSTMRNN class was taken from Notebook ‘5.1 Recurrent Neural Networks’ (original class name is "MyRecurrentNet"), which was provided 
# as part of the course materials for Week 5. Modifications have been made to complete the implementation.

import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMRNN2(nn.Module):
    def __init__(self, n_in, n_hid,nb_layers,bias,families, context_window, embedding_dim):
        super(LSTMRNN2, self).__init__()

        # Context window
        self.context_window = context_window
        window_emb = self.context_window * embedding_dim
        self.embedding = nn.Embedding(num_embeddings=n_in, embedding_dim=embedding_dim, padding_idx=IGNORE_INDEX)

        # Recurrent layer
        self.lstm = nn.LSTM(input_size=window_emb, hidden_size=n_hid, num_layers=nb_layers, bias=bias, batch_first=True)
        
        # Output layer
        self.l_out = nn.Linear(in_features=n_hid, out_features=families, bias=False)
        
    def forward(self, x):
        # RNN does the embedding
        x = self.embedding(x)

        # RNN returns output and last hidden state
        x, (h, c) = self.lstm(x)
        
        # Flatten output for feed-forward layer
        x = x.view(-1, self.lstm.hidden_size)
        
        # Output layer
        x = self.l_out(x)
        
        return x