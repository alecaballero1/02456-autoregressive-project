# This LSTMRNN class was taken from Notebook ‘5.1 Recurrent Neural Networks’ (original class name is "MyRecurrentNet"), which was provided 
# as part of the course materials for Week 5. Modifications have been made to complete the implementation.

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..constants.lstm_constants import VOCAB_SIZE, BOS_IDX, PAD_IDX, EOS_IDX
import math
from tqdm import tqdm
from ..helpers.one_hot_encoding import one_hot_encode_sequence

class LSTMRNN(nn.Module):
    def __init__(self, n_in, n_hid,nb_layers,bias,vocab_size):
        super(LSTMRNN, self).__init__()

        # Initializing vocab size
        self.vocab_size = vocab_size
        
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

    # This sample() method is strongly inspired by the RNNLM.sample() method in Notebook "6.1 Deep Learning with Transformers.ipynb"
    def sample(
        self,
        batch_size:int=1,
        num_steps:int=200,
        temperature: float=1.0
    ):
        # retrives the device that the parameters are stored in (in this case GPU)
        device = next(self.parameters()).device

        # this variable is used to track which sequences have generated the EOS token
        finished_sequences = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # the course sample () method uses torch.empty(), but we have replaced this with torch.full(), since the model expects an input sequence
        token_ids = torch.full((batch_size, 1),BOS_IDX,device=device,dtype=torch.long)

        for t in tqdm(range(num_steps), desc=f"Sampling {num_steps} steps.."):
            # the course RNNLM model uses an embedding layer, but since our LSTMRNN expects one-hot encoded inputs, the sample() method has
            # been modified accordingly
            inputs_one_hot = torch.stack([one_hot_encode_sequence(sequence, self.vocab_size).squeeze(-1) for sequence in token_ids])
            inputs_one_hot = inputs_one_hot.to(device=device, dtype=torch.float32)

            logits = self.forward(inputs_one_hot)
            # Need to reshape the flattened output
            logits = logits.reshape(batch_size,token_ids.shape[1],self.vocab_size)
            logits_t = logits[:, -1:] / temperature

            # prevent generating PAD and BOS tokens
            logits_t[:, :, PAD_IDX] = -math.inf
            logits_t[:, :, BOS_IDX] = -math.inf

            # the RNNLM.sample() method used prevent_repititions, however we have removed this from LSTMRNN.sample(), since eval
            # measures "is it new, is it the same one again"
            p_wt = torch.distributions.Categorical(logits=logits_t)
            tokens_t = p_wt.sample().squeeze(1)

            # Multiple sequences are generated in parallel, for this reason we insert the PAD token in sequences that have already ended
            # while other sequences are still being generated.
            tokens_t = torch.where(finished_sequences,torch.full_like(tokens_t, PAD_IDX),tokens_t)
            token_ids = torch.cat([token_ids, tokens_t.unsqueeze(1)], dim=1)

            finished_sequences = finished_sequences | (tokens_t == EOS_IDX)
            if finished_sequences.all():
                break
        return token_ids[:, 1:]