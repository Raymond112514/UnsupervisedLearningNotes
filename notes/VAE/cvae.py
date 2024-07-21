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

def loss_fn(x, x_recon, mu, log_std):
    """ 
    Loss function for VAE
    """
    recon_loss = F.mse_loss(x, x_recon, reduction='none').view(x.shape[0], -1).sum(dim=1).mean()
    kl_loss = -0.5 * torch.sum(1 + 2 * log_std - mu.pow(2) - (2 * log_std).exp(), dim=1).mean()
    loss = recon_loss + kl_loss
    return loss, kl_loss, recon_loss

def plot_train_samples(samples, m=1, n=6, title="Generated samples"):
    """ 
    Used to display samples during training
    """
    fig, ax = plt.subplots(m, n, figsize=(10, 2))
    ax = ax.flatten()
    for i in range(m*n):
        ax[i].imshow(samples[i])
        ax[i].axis("off")
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.show()

def train_epoch(model, optimizer, train_loader, epoch_num=1, device="cpu", sample=True):
    """ 
    Trains the model for one epoch, display 6 samples at the end
    """
    model.train()
    train_loss = 0
    for batch_idx, (X, _) in enumerate(train_loader):
        X = X.to(device)
        optimizer.zero_grad()
        x_recon, mu, log_std = model(X)
        loss, kl_loss, recon_loss = loss_fn(X, x_recon, mu, log_std)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss = round(train_loss / len(train_loader), 5)
    if sample:
        with torch.no_grad():
            samples = model.sample(6)
            plot_train_samples(samples, title=f"Epoch {epoch_num} generated samples with loss: {train_loss}")
    return train_loss

def train(model, optimizer, train_loader, n_epochs, device="cpu", sample=True):
    """ 
    Training loop
    """
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    for epoch in range(1, n_epochs+1):
        train_epoch(model, optimizer, train_loader, epoch_num=epoch, device=device, sample=sample)
        scheduler.step()
        
#####################################################################################################################################################################################################################################################   Utility fucntion                                                                                          #########
##################################################################################################################################################################################################################################################

def plot_samples(model, n_samples, device="cpu"):
    """  
    Generates n_samples samples and display them in 6X6 grid
    """
    samples = model.sample(36)
    fig, ax = plt.subplots(6, 6, figsize=(6, 6))
    ax = ax.flatten()
    for i in range(36):
        ax[i].imshow(samples[i], cmap="gray")
        ax[i].axis("off")
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()
    
def plot_reconstruction(model, test_loader, device="cpu"):
    """ 
    Display reconstructed images
    """
    X = next(iter(test_loader))[0][:36].to(device)
    out, _, _ = model(X)
    out = out.permute(0, 2, 3, 1)
    out = torch.clamp((out + 1) / 2, 0, 1)
    X = torch.clamp((X + 1) / 2, 0, 1)
    X = X.permute(0, 2, 3, 1)
    X = X.detach().cpu().numpy()
    out = out.detach().cpu().numpy()

    fig, ax = plt.subplots(6, 12, figsize=(12, 6))
    for i in range(6):
        for j in range(6):
            ax[i, j * 2].imshow(X[i * 6 + j], cmap="gray")
            ax[i, j * 2].axis('off')
            ax[i, j * 2 + 1].imshow(out[i * 6 + j], cmap="gray")
            ax[i, j * 2 + 1].axis('off')