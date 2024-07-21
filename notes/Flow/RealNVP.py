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

def plot_training_samples(samples, m=1, n=6, title="Generated samples"):
    fig, ax = plt.subplots(m, n, figsize=(10, 2))
    ax = ax.flatten()
    for i in range(m*n):
        ax[i].imshow(samples[i], cmap="gray")
        ax[i].axis("off")
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.show()

def train_epoch(model, optimizer, train_loader, epoch_num=1, device=DEVICE, sample=True):
    model.train()
    train_loss = 0.0
    for X, _ in train_loader:
        X = X.to(device)
        z, s_det = model(X)
        loss = loss_fn(z, s_det)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss = round(train_loss / len(train_loader), 5)
    if sample:
        with torch.no_grad():
            samples = model.sample(16).permute(0, 2, 3, 1)
            samples = samples.detach().cpu().numpy()
            plot_training_samples(samples, title=f"Epoch {epoch_num} generated samples with loss: {train_loss}")
    return train_loss

def train(model, optimizer, train_loader, n_epochs, device=DEVICE, sample=True):
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    for epoch in range(1, n_epochs+1):
        train_epoch(model, optimizer, train_loader, epoch_num=epoch, device=device, sample=sample)
        torch.save(model.state_dict(), '/content/drive/MyDrive/realnvp_cifar10.pth')
        scheduler.step()
        
#####################################################################################################################################################################################################################################################   Utility fucntion                                                                                          #########
##################################################################################################################################################################################################################################################

def plot_samples(model, n_samples, device="cpu"):
    """  
    Takes in a list of samples and display in a 6X6 grid
    """
    samples = model.sample(36).permute(0, 2, 3, 1)
    samples = torch.clamp((samples + 1) / 2, 0, 1)
    samples = samples.detach().cpu().numpy()
    fig, ax = plt.subplots(6, 6, figsize=(6, 6))
    ax = ax.flatten()
    for i in range(36):
        ax[i].imshow(samples[i], cmap="gray")
        ax[i].axis("off")
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()