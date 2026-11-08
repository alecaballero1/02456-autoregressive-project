# This file contains code snippets from the "5.1 Recurrent Neural Networks".ipynb notebook provided in the course,
# specifically code from section "Introduction to the Long Short-Term Memory (LSTM) Cell" in the notebook.
from .normal_initializer import NormalInitializer
from .var import Var
from typing import Sequence
from math import exp, log, tanh
from .dense_layer import DenseLayer

# The code snippet below originates from section "Exercise h) Make the LSTMLayer class" in the referenced notebook.
# Modifications have been made to complete the implementation.
class LSTMLayer:
    def __init__(self, n_in: int, n_hid: int, act_fn, initializer = NormalInitializer(), initializer_hid = NormalInitializer()):
        n_total = n_in + n_hid
        self.n_in = n_in
        self.n_hid = n_hid
        self.g_layer = DenseLayer(n_total, n_hid, tanh(), initializer)
        self.i_layer = DenseLayer(n_total, n_hid, Var.sigmoid(), initializer)
        self.f_layer = DenseLayer(n_total, n_hid, Var.sigmoid(), initializer)
        self.o_layer = DenseLayer(n_total, n_hid, Var.sigmoid(), initializer)
        self.initial_hid = [Var(0.0) for _ in range(n_hid)]
        self.stored_hid = [Var(0.0) for _ in range(n_hid)]
        self.initial_c = [Var(0.0) for _ in range(n_hid)]
        self.stored_c = [Var(0.0) for _ in range(n_hid)]
        self.act_fn = act_fn
    
    def __repr__(self):    
        return 'Feed-forward: ' + repr(self.in_hid_layer) + ' Candidate: ' + repr(self.g_layer) + ' i gate ' + repr(self.i_layer) + ' f gate ' + repr(self.f_layer) + ' o gate ' + repr(self.o_layer) + ' Initial hidden: ' + repr(self.initial_hid)

    def parameters(self) -> Sequence[Var]:      
      return self.in_hid_layer.parameters() + self.g_layer.parameters() + self.i_layer.parameters() + self.f_layer.parameters() + self.o_layer.parameters() + self.initial_hid

    def forward_step(self, input: Sequence[Var], input_hid: Sequence[Var], input_c: Sequence[Var]) -> Sequence[Var]:
        hids = []
        cs = []
        concatenated_input = []
        for val in input_hid:
          concatenated_input.append(val)
        for val in input:
          concatenated_input.append(val)

        g = self.g_layer.forward(concatenated_input)
        i = self.i_layer.forward(concatenated_input)
        f = self.f_layer.forward(concatenated_input)
        o = self.o_layer.forward(concatenated_input)

        for idx in range(self.n_hid):
           c_t = input_c[idx] * f[idx] + g[idx] * i[idx]
           h_t = self.act_fn(c_t) * o[idx]

           cs.append(c_t)
           hids.append(h_t)

        return hids, cs
    
    def forward_sequence(self, input: Sequence[Sequence[Var]], use_stored_hid = False) -> Sequence[Sequence[Var]]:
        out = []
        if use_stored_hid:
            hid = self.stored_hid
            c = self.stored_c
        else:
            hid = self.initial_hid
            c = self.initial_c
        # Takes a sequence and loops over each character in the sequence. Note that each character has dimenson equal to the embeddng dimenson
        for i in range(len(input)):
            hid, c = self.forward_step(input[i], hid, c)
            out.append(hid)
        self.stored_hid = hid
        self.stored_c = c
        return out