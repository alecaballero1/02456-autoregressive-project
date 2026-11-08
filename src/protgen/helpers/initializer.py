# This Initializer class was taken from Notebook ‘5.1 Recurrent Neural Networks’, which was provided as part 
# of the course materials for Week 5.
class Initializer:

  def init_weights(self, n_in, n_out):
    raise NotImplementedError

  def init_bias(self, n_out):
    raise NotImplementedError