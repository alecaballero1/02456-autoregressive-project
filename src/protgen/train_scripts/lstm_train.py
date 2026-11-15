# This code was taken from Notebook ‘5.1 Recurrent Neural Networks’, which was provided as part of the course materials for Week 5. 
# Modifications have been made to complete the implementation. The file also contains code from Notebook '4.1 CNN Introduction'.ipynb,
# which was provided as part of the course materials for Week 4.

from ..models.lstm_pytorch import LSTMRNN
import torch.optim as optim
from torch import nn
from ..data.split_data import train_data, val_data, test_data
from ..helpers.one_hot_encoding import one_hot_encode_sequence
import torch
from ..constants.lstm_constants import N_HID, NUM_EPOCHS, CSV_FILENAME, MODEL_NAME, MODEL_NB, NUM_LAYERS, MAX_EPOCHS_WITHOUT_IMPROVEMENT, MIN_IMPROVEMENT
import matplotlib.pyplot as plt
import numpy as np
from protgen.data import IGNORE_INDEX
import time
import csv
from pathlib import Path

# Asserting identical vocab size across datasets
assert train_data.vocab_size == val_data.vocab_size == test_data.vocab_size

# Ensure script is run on GPU rather than CPU
device = torch.device("cuda")

# parameters for early stopping
best_validation_loss = float("inf")
epochs_without_improvement = 0

# Initialize a new network
net = LSTMRNN(train_data.vocab_size, N_HID, NUM_LAYERS, True,train_data.vocab_size).to(device)

# Define a loss function and optimizer for this problem
criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX) # using `ignore_index` to make the padding free of loss as recommended in the introduction.ipynb notebook
optimizer = optim.Adam(net.parameters(), lr=5e-4, weight_decay=1e-4) # learning rate 0.0005, weight decay (L2 regularization) 0.0001 to penalize model and reduce overfitting

word_to_idx = train_data.vocab
idx_to_word = {idx: token for token, idx in word_to_idx.items()}

# Logic for saving best model
src_path = Path(__file__).resolve().parents[2]
model_path = src_path / "protgen" / "generated" / "lstm" / str(MODEL_NB)
model_path.mkdir(parents=True, exist_ok=True)
model_path = model_path / MODEL_NAME

# Track loss
training_loss, validation_loss = [], []

# For each epoch
for i in range(NUM_EPOCHS):
    epoch_start = time.time()
    
    # Track loss
    epoch_training_loss = 0
    epoch_validation_loss = 0
    
    net.eval()
        
    # For each sentence in validation set
    with torch.no_grad():
        for inputs, targets, labels, lengths in val_data:
            
            # One-hot encode input and target sequence
            inputs_one_hot = one_hot_encode_sequence(inputs, val_data.vocab_size)
            
            # Convert input to tensor
            inputs_one_hot = inputs_one_hot.to(dtype=torch.float32, device=device).permute(2,0,1)
            
            # Convert target to tensor
            targets_ = torch.as_tensor(targets, dtype=torch.long, device=device)
            
            # Forward pass
            outputs = net(inputs_one_hot)
            
            # Compute loss
            loss = criterion(outputs, targets_)
            
            # Update loss
            epoch_validation_loss += loss.item()
    
    net.train()
    
    # For each sentence in training set
    for inputs, targets, labels, lengths in train_data:

        # One-hot encode input and target sequence
        inputs_one_hot = one_hot_encode_sequence(inputs, train_data.vocab_size)
        
        # Convert input to tensor
        inputs_one_hot = inputs_one_hot.to(dtype=torch.float32, device=device).permute(2,0,1)
        
        # Convert target to tensor
        targets_ = torch.as_tensor(targets, dtype=torch.long, device=device)
        
        # Forward pass
        outputs = net(inputs_one_hot)
        
        # Compute loss
        loss = criterion(outputs, targets_)
        
        # Clean up gradients from the model
        optimizer.zero_grad()

        # Compute gradients based on the loss from the current batch (backpropagation).
        loss.backward()
        
        # Take one optimizer step using the gradients computed in the previous step.
        optimizer.step()
        
        # Update loss
        epoch_training_loss += loss.item()

    # Save loss for plot
    average_train_loss = epoch_training_loss/len(train_data)
    average_val_loss = epoch_validation_loss/len(val_data)
    training_loss.append(average_train_loss)
    validation_loss.append(average_val_loss)

    # Check improvement
    if average_val_loss < best_validation_loss - MIN_IMPROVEMENT:
        best_validation_loss = average_val_loss
        epochs_without_improvement = 0
        # Save model
        torch.save({"model": net.state_dict(), "criterion": criterion.state_dict(), "optimizer": optimizer.state_dict(), "train_loss": training_loss, "val_loss": validation_loss, "vocabulary": train_data.vocab, "vocabulary_size": train_data.vocab_size, "hidden_size": N_HID, "num_layers":NUM_LAYERS, "epoch": i+1, "best_validation_loss": best_validation_loss}, model_path)    
        print(f"Epoch {i}, validation loss improved: {best_validation_loss:.4f}. New model has been saved.", flush=True)
    else:
        epochs_without_improvement += 1
    # Print loss every 10 epochs
    if i % 10 == 0:
        epoch_duration = time.time() - epoch_start
        print(f'Epoch {i}, training loss: {training_loss[-1]}, validation loss: {validation_loss[-1]}, duration: {epoch_duration:.2f}s', flush=True)

    # Early stopping
    if epochs_without_improvement >= MAX_EPOCHS_WITHOUT_IMPROVEMENT:
        print(f"Early stopping at epoch {i+1}. The best validation loss was {best_validation_loss:.4f}", flush=True)
        break
            
