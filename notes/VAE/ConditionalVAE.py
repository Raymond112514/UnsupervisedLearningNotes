import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch.nn.init as init
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

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
    for batch_idx, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        x_recon, mu, log_std = model(X, y)
        loss, kl_loss, recon_loss = loss_fn(X, x_recon, mu, log_std)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss = round(train_loss / len(train_loader), 5)
    if sample:
        with torch.no_grad():
            label = torch.randint(0, model.n_classes, (6, )).to(device)
            samples = model.sample(6, label)
            plot_train_samples(samples, title=f"Epoch {epoch_num} generated samples with loss: {train_loss}")
    return train_loss

def train(model, optimizer, train_loader, n_epochs, device="cpu", sample=True):
    """ 
    Training loop
    """
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    for epoch in range(1, n_epochs+1):
        train_epoch(model, optimizer, train_loader, epoch_num=epoch, device=device, sample=sample)
        torch.save(model.state_dict(), 'vae_cifar10_without_covariance.pth')
        scheduler.step()
        
def plot_samples(model, device="cpu"):
    """ 
    Plot the generated conditional samples
    """
    fig, ax = plt.subplots(10, 10, figsize=(8, 8))
    for i in range(10):
        samples = model.sample(10, torch.tensor(np.arange(10), device=device))
        for j in range(10):
            ax[i, j].imshow(samples[j], cmap="gray")
            ax[i, j].axis("off")