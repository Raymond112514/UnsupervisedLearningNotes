import torch
import torch.nn as nn
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt

#####################################################################################################################################################################################################################################################   Training loop                                                                                             #########
##################################################################################################################################################################################################################################################

def train(model, train_loader, optimizer, n_epochs, batch_size, resample_every=20, device="cpu"):
    """ 
    Training loop for MADE
    """
    model.train()
    model.to(device)
    for epoch in range(n_epochs):
        train_loss = 0
        for i, (X, y) in enumerate(train_loader):
            if i % resample_every == 0:
                model.update_masks()
            X = X.to(device)
            logits = model(X)
            loss = nn.BCEWithLogitsLoss()(logits, X)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f"Epoch {epoch} with loss {train_loss / (len(train_loader) // batch_size)}")

def evaluate(model, val_loader, n_masks, batch_size, device="cpu"):
    """ 
    Evaluates the loss metric for a MADE model
    """
    model.eval()
    model.to(device)
    test_loss = 0
    with torch.no_grad():
        for i, (X, y) in enumerate(val_loader):
            X = X.to(device)
            logits = torch.zeros(X.shape, device=DEVICE)
            model.mask_idx = 0
            for i in range(n_masks):
                model.update_masks()
                logits += model(X).to(DEVICE)
            logits /= n_masks
            loss = nn.BCEWithLogitsLoss()(logits, X)
            test_loss += loss.item()
    return test_loss / (len(val_loader) // batch_size)

#####################################################################################################################################################################################################################################################   Utility fucntion                                                                                          #########
##################################################################################################################################################################################################################################################
  
def plot_samples(model, n_samples, device="cpu"):
    """  
    Takes in a list of samples and display in a 6X6 grid
    """
    samples = model.sample(n_samples, device=device)
    samples = np.squeeze(samples, axis=1)
    grid_size = int(np.sqrt(n_samples))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(6, 6))
    axes = axes.flatten()
    for i in range(n_samples):
        axes[i].imshow(samples[i], cmap='gray')
        axes[i].axis('off')
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()