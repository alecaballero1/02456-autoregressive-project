# These methods were taken from Notebook ‘5.1 Recurrent Neural Networks’, which was provided as part of the course materials for Week 5. 

import numpy as np
import torch
from .sequence_to_dicts import sequences_to_dicts

def one_hot_encode(idx, vocab_size):
    """
    One-hot encodes a single word given its index and the size of the vocabulary.
    
    Args:
     `idx`: the index of the given word
     `vocab_size`: the size of the vocabulary
    
    Returns a 1-D numpy array of length `vocab_size`.
    """
    # Initialize the encoded array
    one_hot = np.array([np.zeros(vocab_size)])
    
    # Set the appropriate element to one
    one_hot[0][idx] = 1.0
    return one_hot

# This method from the referenced Notebook has been updated to align with the vocabulary mapping from ProteinDataset.
def one_hot_encode_sequence(sequence, vocab_size):
    """
    One-hot encodes a sequence of words given a fixed vocabulary size.
    
    Args:
     `sentence`: a list of words to encode
     `vocab_size`: the size of the vocabulary
     
    Returns a 3-D torch.Tensor of shape (num words, vocab size, 1).
    """
    # Encode each word in the sentence
    encoding = np.array([one_hot_encode(int(idx), vocab_size) for idx in sequence])

    # Reshape encoding s.t. it has shape (num words, vocab size, 1)
    encoding = encoding.reshape(encoding.shape[0], encoding.shape[2], 1)
    return torch.tensor(encoding)
