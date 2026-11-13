"""
Constants file for gru. Inline comments are the key for the parameter that is used in naming. 
"""
from pathlib import Path

CHECKPOINT_DIR = Path("results/checkpoints/gru")
SEQUENCE_DIR = Path("results/sequences/gru")

SEED = 1

# Model
NUM_GRU_LAYERS = 1 #l
GRU_SIZE = 50 #h
GRU_DROPOUT = 0.0 #Dr

# Training
EPOCHS = 60 #e
BATCH_SIZE = 64 #b
LEARNING_RATE = 0.001 #r
GRADIENT_CLIP_NORM = 1.0 #c
WEIGHT_DECAY = 0.0 #d

# Generation
TEMPERATURE = 1.0 #t
