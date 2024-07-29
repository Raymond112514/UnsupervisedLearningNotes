import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import pickle
from torch.utils.data import TensorDataset, DataLoader

class WeightNormConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(WeightNormConv2d, self).__init__()
        self.weight_norm = nn.utils.weight_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias))

    def forward(self, x):
        return self.weight_norm(x)

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResnetBlock, self).__init__()
        self.net = nn.Sequential(nn.BatchNorm2d(in_channels),
                                 nn.ReLU(inplace=True),
                                 WeightNormConv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU(inplace=True),
                                 WeightNormConv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False))

    def forward(self, x):
        out = self.net(x)
        return x + out

class Resnet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers):
        super(Resnet, self).__init__()
        self.net = [WeightNormConv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
                    nn.ReLU()]
        self.net += [ResnetBlock(hidden_channels, hidden_channels) for _ in range(n_layers)]
        self.net += [nn.ReLU(),
                     WeightNormConv2d(hidden_channels, out_channels, kernel_size=3, padding=1)]
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        return self.net(x)

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