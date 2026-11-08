# This code was taken from Notebook ‘5.1 Recurrent Neural Networks’, which was provided as part of the course materials for Week 5. 
# Modifications have been made to complete the implementation. The file also contains code from Notebook '4.1 CNN Introduction'.ipynb,
# which was provided as part of the course materials for Week 4.

from ..models.lstm_pytorch import LSTMRNN
import torch.optim as optim
from torch import nn
from ..data.split_data import train_data, val_data, test_data
from ..helpers.one_hot_encoding import one_hot_encode_sequence
import torch
from ..constants.lstm_constants import VOCAB_SIZE, N_IN, N_HID, NUM_EPOCHS
import matplotlib.pyplot as plt
import numpy as np
from protgen.data import AA, FAMILIES, IGNORE_INDEX, special_tokens

# Asserting identical vocab size across datasets
assert train_data.vocab_size == val_data.vocab_size == test_data.vocab_size

# Ensure script is run on GPU rather than CPU
device = torch.device("cuda")

# Initialize a new network
net = LSTMRNN(train_data.vocab_size, N_HID, 1, True,train_data.vocab_size).to(device)

# Define a loss function and optimizer for this problem
criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX) # using `ignore_index` to make the padding free of loss as recommended in the introduction.ipynb notebook
optimizer = optim.Adam(net.parameters(), lr=1e-3)

word_to_idx = train_data.vocab
idx_to_word = {idx: token for token, idx in word_to_idx.items()}

# Track loss
training_loss, validation_loss = [], []

# For each epoch
for i in range(NUM_EPOCHS):
    
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
            inputs_one_hot = torch.Tensor(inputs_one_hot, device=device).permute(2,0,1)
            
            # Convert target to tensor
            targets_ = torch.LongTensor(targets, device=device)
            
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
        inputs_one_hot = torch.Tensor(inputs_one_hot, device=device).permute(2,0,1)
        
        # Convert target to tensor
        targets_ = torch.LongTensor(targets, device=device)
        
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
    training_loss.append(epoch_training_loss/len(train_data))
    validation_loss.append(epoch_validation_loss/len(val_data))

    # Save model
    torch.save({"model": net.state_dict(), "criterion": criterion.state_dict(), "optimizer": optimizer.state_dict(), "train_loss": training_loss, "val_loss": validation_loss, "vocabulary": train_data.vocab, "vocabulary_size": train_data.vocab_size, "hidden_size": N_HID}, "lstm_train_model.pth")    

    # Print loss every 10 epochs
    if i % 10 == 0:
        print(f'Epoch {i}, training loss: {training_loss[-1]}, validation loss: {validation_loss[-1]}')

        
# Get first sentence in test set
inputs, targets, labels, lengths = test_data[0]

# One-hot encode input and target sequence
inputs_one_hot = one_hot_encode_sequence(inputs, test_data.vocab_size)

# Convert input to tensor
inputs_one_hot = torch.Tensor(inputs_one_hot, device=device).permute(2,0,1)

# Convert target to tensor (not used currently, uncomment if used in the future)
# targets_ = torch.LongTensor(targets)

# Forward pass
net.eval()
with torch.no_grad():
    outputs = net(inputs_one_hot).cpu().numpy()

print('\nInput sequence:')
print(inputs)

print('\nTarget sequence:')
print(targets)

print('\nPredicted sequence:')
print([idx_to_word[np.argmax(output)] for output in outputs])

# Plot training and validation loss
epoch = np.arange(len(training_loss))
plt.figure()
plt.plot(epoch, training_loss, 'r', label='Training loss',)
plt.plot(epoch, validation_loss, 'b', label='Validation loss')
plt.legend()
plt.xlabel('Epoch'), plt.ylabel('NLL')
plt.show()