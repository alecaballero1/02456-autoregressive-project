# This DenseLayer class was taken from Notebook ‘5.1 Recurrent Neural Networks’, which was provided as part 
# of the course materials for Week 5.

from typing import Sequence
from .var import Var
from .normal_initializer import NormalInitializer

class DenseLayer:
    def __init__(self, n_in: int, n_out: int, act_fn, initializer = NormalInitializer(), use_bias=True):
        self.weights = initializer.init_weights(n_in, n_out)
        self.use_bias = use_bias
        if use_bias:
          self.bias = initializer.init_bias(n_out)
        self.act_fn = act_fn
    
    def __repr__(self):    
        return 'Weights: ' + repr(self.weights) + (' Biases: ' + repr(self.bias) if self.use_bias else '')

    def parameters(self) -> Sequence[Var]:
      params = []
      for r in self.weights:
        params += r

      if self.use_bias:
        params += self.bias

      return params

    def forward(self, input: Sequence[Var]) -> Sequence[Var]:
        # self.weights is a matrix with dimension n_in x n_out. We check that the dimensionality of the input 
        # to the current layer matches the number of nodes in the current layer
        assert len(self.weights) == len(input), "weights and input must match in first dimension"
        weights = self.weights
        out = []
        # For some given data point single_input, we now want to calculate the resulting value in each node in the current layer
        # We therefore loop over the (number of) nodes in the current layer:
        for j in range(len(weights[0])): 
            # Initialize the node value depending on its corresponding parameters.
            node = self.bias[j] if self.use_bias else Var(0.0)
            # We now finish the linear transformation corresponding to the parameters of the currently considered node.
            for i in range(len(input)):
                node += input[i]*weights[i][j]
            node = self.act_fn(node)
            out.append(node)

        return out
    
    def forward_sequence(self, input: Sequence[Sequence[Var]]) -> Sequence[Sequence[Var]]:
        out = []
        for i in range(len(input)): 
            node = self.forward(input[i])
            out.append(node)

        return out