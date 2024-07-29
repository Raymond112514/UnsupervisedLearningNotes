import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import pickle
from torch.utils.data import TensorDataset, DataLoader

class SimpleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=512):
        super(SimpleBlock, self).__init__()
        self.net = nn.Sequential(nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
                                 nn.ReLU(),
                                 nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
                                 nn.ReLU(),
                                 nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1))
        nn.init.normal_(self.net[0].weight, mean=0, std=0.03)
        nn.init.normal_(self.net[2].weight, mean=0, std=0.03)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        out = self.net(x)
        return out
    
def loss_fn(z, log_det):
    nll = (0.5 * z ** 2).sum(-1) + 512 * np.log(2 * np.pi)
    loss = nll - log_det
    return loss.mean()

#####################################################################################################################################################################################################################################################   Training loop                                                                                             #########
##################################################################################################################################################################################################################################################

def plot_training_samples(samples, m=1, n=6, title="Generated samples"):
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

def train_epoch(model, optimizer, train_loader, epoch_num=1, device=DEVICE, sample=True):
    """ 
    Trains the model for one epoch, display 6 samples at the end
    """
    model.train()
    train_loss = 0.0
    for X, _ in train_loader:
        X = X.to(device)
        z, log_det = model(X)
        loss = loss_fn(z, log_det)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss = round(train_loss / len(train_loader), 5)
    if sample:
        with torch.no_grad():
            samples = model.sample(16).permute(0, 2, 3, 1)
            samples = torch.clamp((samples + 1) / 2, 0, 1)
            samples = samples.detach().cpu().numpy()
            plot_training_samples(samples, title=f"Epoch {epoch_num} generated samples with loss: {train_loss}")
    return train_loss

def train(model, optimizer, train_loader, n_epochs, device=DEVICE, sample=True):
    """ 
    Training loop
    """
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
    for epoch in range(1, n_epochs+1):
        train_epoch(model, optimizer, train_loader, epoch_num=epoch, device=device, sample=sample)
        scheduler.step()
        
#####################################################################################################################################################################################################################################################   Utility fucntion                                                                                          #########
##################################################################################################################################################################################################################################################

def plot_samples(model, n_samples, device="cpu"):
    """  
    Takes in a list of samples and display in a 6X6 grid
    """
    samples = model.sample(n_samples).permute(0, 2, 3, 1)
    samples = torch.clamp((samples + 1) / 2, 0, 1)
    samples = samples.detach().cpu().numpy()
    fig, ax = plt.subplots(6, 6, figsize=(6, 6))
    ax = ax.flatten()
    for i in range(36):
        ax[i].imshow(samples[i], cmap="gray")
        ax[i].axis("off")
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()