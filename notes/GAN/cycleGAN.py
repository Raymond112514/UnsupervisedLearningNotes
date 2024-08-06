import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torch.utils.data import Dataset, random_split
import os
import cv2


class GrayscaleAndColoredMNIST(Dataset):
    def __init__(self, grayscale_mnist, color_mnist):
        self.grayscale_mnist = np.squeeze(np.stack((grayscale_mnist,) * 3, axis=1), axis=2)
        self.color_mnist = color_mnist

    def __len__(self):
        return len(self.grayscale_mnist)

    def __getitem__(self, idx):
        grayscale_image = self.grayscale_mnist[idx]
        color_image = self.color_mnist[idx]
        grayscale_image = cv2.resize(grayscale_image.transpose(1, 2, 0), (32, 32)).transpose(2, 0, 1)
        color_image = cv2.resize(color_image.transpose(1, 2, 0), (32, 32)).transpose(2, 0, 1)
        return grayscale_image, color_image

def load_mnist(grayscale_mnist_path, colored_mnist_path):
    grayscale_mnist = np.load(grayscale_mnist_path)
    color_mnist = np.load(colored_mnist_path)
    return grayscale_mnist, color_mnist
  
class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResnetBlock, self).__init__()
        self.net = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                                 nn.BatchNorm2d(in_channels),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU(inplace=True))

    def forward(self, x):
        out = self.net(x)
        return x + out
      
def train_epoch(model_g, model_c, train_loader, test_loader, optimizer_disc, optimizer_gen, criterion, lambda_cycle, lambda_identity, sample=True, device="cpu"):
    model_g.train()
    model_c.train()
    model_g.to(device)
    model_c.to(device)
    train_loss = 0.0
    for i, (x_g, x_c) in enumerate(train_loader):
        x_g, x_c = x_g.float().to(device), x_c.float().to(device)

        batch_size = x_g.size(0)

        ## Train Discriminator 
        optimizer_disc.zero_grad()

        ## Discriminator loss for grayscale
        loss_g_real_disc = criterion(model_g.discriminator(x_g), torch.ones(batch_size, 1).to(device))
        loss_g_fake_disc = criterion(model_g.discriminator(model_g.generator(x_c)), torch.zeros(batch_size, 1).to(device))

        ## Discriminator loss for color
        loss_c_real_disc = criterion(model_c.discriminator(x_c), torch.ones(batch_size, 1).to(device))
        loss_c_fake_disc = criterion(model_c.discriminator(model_c.generator(x_g)), torch.zeros(batch_size, 1).to(device))

        ## Compute discriminator loss
        loss_disc = (loss_g_real_disc + loss_g_fake_disc + loss_c_real_disc + loss_c_fake_disc) / 4
        loss_disc.backward()
        optimizer_disc.step()
  
        ## Train Generator
        optimizer_gen.zero_grad()

        ## Generator loss
        loss_g_fake_gen = criterion(model_g.discriminator(model_g.generator(x_c)), torch.ones(batch_size, 1).to(device))
        loss_c_fake_gen = criterion(model_c.discriminator(model_c.generator(x_g)), torch.ones(batch_size, 1).to(device))

        ## Cycle consistency loss
        loss_cycle_g = F.l1_loss(x_g, model_g.generator(model_c.generator(x_g)))
        loss_cycle_c = F.l1_loss(x_c, model_c.generator(model_g.generator(x_c)))

        ## Identity loss
        loss_identity_g = F.l1_loss(x_g, model_g.generator(x_g))
        loss_identity_c = F.l1_loss(x_c, model_c.generator(x_c))

        loss_gen = (loss_g_fake_gen + loss_c_fake_gen) + lambda_cycle * (loss_cycle_g + loss_cycle_c) + lambda_identity * (loss_identity_g + loss_identity_c)
        loss_gen.backward()
        optimizer_gen.step()
        train_loss += loss_disc.item() + loss_gen.item()
    return train_loss
  
def train(n_epochs, model_g, model_c, train_loader, test_loader, optimizer_disc, optimizer_gen, criterion, lambda_cycle, lambda_identity, device='cpu'):
    for _ in range(n_epochs):
        loss = train_epoch(model_g, model_c, train_loader, test_loader, optimizer_disc, optimizer_gen, criterion, lambda_cycle, lambda_identity, device=device)
        
def plot_grayscale_to_color(model_g, model_c, test_loader, device='cpu'):
    test_iterator = iter(test_loader)
    with torch.no_grad():
        xg, xc = next(test_iterator)
        xg, xc = xg.float().to(device), xc.float().to(device)
        model_g.eval()
        model_c.eval()
        sample_c = model_c.generator(xg[:20])
        sample_c = torch.clamp(sample_c, 0, 1)
        recon_g = model_g.generator(sample_c)
        recon_g = torch.clamp(recon_g, 0, 1)

        fig, ax = plt.subplots(3, 20, figsize=(12, 2))
        for i in range(20):
            ax[0, i].imshow(xg[i].permute(1, 2, 0).cpu().detach().numpy())
            ax[0, i].axis('off')
        for i in range(20):
            ax[1, i].imshow(sample_c[i].permute(1, 2, 0).cpu().detach().numpy())
            ax[1, i].axis('off')
        for i in range(20):
            ax[2, i].imshow(recon_g[i].permute(1, 2, 0).cpu().detach().numpy())
            ax[2, i].axis('off')
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.tight_layout()
        plt.show()
        
def plot_color_to_grayscale(model_g, model_c, test_loader, device='cpu')
    test_iterator = iter(test_loader)
    with torch.no_grad():
        xg, xc = next(test_iterator)
        xg, xc = xg.float().to(device), xc.float().to(device)
        model_g.eval()
        model_c.eval()
        sample_g = model_g.generator(xc[:20])
        sample_g = torch.clamp(sample_g, 0, 1)
        recon_c = model_c.generator(sample_g)
        recon_c = torch.clamp(recon_c, 0, 1)

        fig, ax = plt.subplots(3, 20, figsize=(12, 2))
        for i in range(20):
            ax[0, i].imshow(xc[i].permute(1, 2, 0).cpu().detach().numpy())
            ax[0, i].axis('off')
        for i in range(20):
            ax[1, i].imshow(sample_g[i].permute(1, 2, 0).cpu().detach().numpy())
            ax[1, i].axis('off')
        for i in range(20):
            ax[2, i].imshow(recon_c[i].permute(1, 2, 0).cpu().detach().numpy())
            ax[2, i].axis('off')
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.tight_layout()
        plt.show()