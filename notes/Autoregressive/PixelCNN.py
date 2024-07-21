import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import pickle
from torch.utils.data import TensorDataset, DataLoader

#####################################################################################################################################################################################################################################################   Training loop                                                                                             #########
##################################################################################################################################################################################################################################################

def train_epoch(model, train_loader, optimizer, criterion, device="cpu"):
    """ 
    Single epoch of training loop
    """
    model.train()
    train_loss = 0.0
    for x in train_loader:
        x = x[0].to(device).float()
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, x.long())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(train_loader)

def evaluate(model, test_loader, criterion, device="cpu"):
    """ 
    Evaluate test performance
    """
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for x in test_loader:
            x = x[0].to(device).float()
            logits = model(x)
            loss = criterion(logits, x.long())
            test_loss += loss.item()
    return test_loss / len(test_loader)

def plot_training_samples(samples, m=1, n=6, title="Generated samples"):
    """ 
    Plot 6 generated samples during training
    """
    fig, ax = plt.subplots(m, n, figsize=(10, 2))
    ax = ax.flatten()
    for i in range(m*n):
        ax[i].imshow(samples[i], cmap="gray")
        ax[i].axis("off")
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.show()

def train(model, train_loader, test_loader, optimizer, criterion, n_epochs, device="cpu", sample=True):
    """ 
    Training loop
    """
    for epoch in range(n_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss = evaluate(model, test_loader, criterion, device)
        if sample:
            with torch.no_grad():
                samples = model.sample(6, device)
                samples = samples / np.max(samples)
                plot_training_samples(samples, title=f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
                
#####################################################################################################################################################################################################################################################   Utility fucntion                                                                                          #########
##################################################################################################################################################################################################################################################

def plot_samples(model, n_samples, device="cpu"):
    """  
    Takes in a list of samples and display in a 6X6 grid
    """
    samples = model.sample(n_samples, device)
    grid_size = int(np.sqrt(n_samples))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(6, 6))
    axes = axes.flatten()
    for i in range(n_samples):
        axes[i].imshow(samples[i], cmap='gray')
        axes[i].axis('off')
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()