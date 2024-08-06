import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ResidualBlock(nn.Module):
    """
    Implementation of residual block
    """
    def __init__(self, in_channels):
        """
        @param in_channels: int
        """
        super(ResidualBlock, self).__init__()
        self.net = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=1, padding=0),
                                 nn.ReLU(),
                                 nn.Conv2d(in_channels=in_channels//2, out_channels=in_channels//2, kernel_size=7, padding=3),
                                 nn.ReLU(),
                                 nn.Conv2d(in_channels=in_channels//2, out_channels=in_channels, kernel_size=1, padding=0),
                                 nn.ReLU())

    def forward(self, x):
        """
        @param x: torch.tensor [B, C, H, W]
        """
        out = self.net(x)
        return x + out
      
def plot_reconstructions(model, X, title="Reconstruction"):
    model.eval()
    out, _, _ = model(X[:6])
    out = out.permute(0, 2, 3, 1).contiguous()
    X = X.permute(0, 2, 3, 1).contiguous()
    out = out.detach().cpu().numpy()
    X = X.detach().cpu().numpy()
    fig, ax = plt.subplots(2, 6, figsize=(12, 4))
    fig.suptitle(title, fontsize=16)
    for i in range(6):
        ax[0, i].imshow(X[i])
        ax[1, i].imshow(out[i])
        ax[0, i].axis('off')
        ax[1, i].axis('off')
    plt.show()

def train_epoch(model, train_loader, optimizer, device="cpu", beta=0.05):
    model.train()
    train_loss = 0.0
    for X, _ in train_loader:
        X = X.to(device)
        optimizer.zero_grad()
        out, commitment_loss, codebook_loss = model(X)
        reconstruction_loss = F.mse_loss(out, X)
        loss = reconstruction_loss + codebook_loss + beta * commitment_loss
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(train_loader)

def train(model, train_loader, optimizer, n_epochs, device="cpu", beta=0.05):
    for epoch in range(n_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, beta)
        plot_reconstructions(model, next(iter(train_loader))[0].to(device), title=f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}')
        
def plot_reconstruction(test_loader):
    X = next(iter(test_loader))[0][:36].to(DEVICE)
    out, _, _ = model(X)
    X = X.detach().cpu().numpy()
    out = out.detach().cpu().numpy()

    fig, ax = plt.subplots(6, 12, figsize=(12, 6))
    for i in range(6):
        for j in range(6):
            ax[i, j * 2].imshow(X[i * 6 + j].reshape(32, 32), cmap='gray')
            ax[i, j * 2].axis('off')
            ax[i, j * 2 + 1].imshow(out[i * 6 + j].reshape(32, 32), cmap='gray')
            ax[i, j * 2 + 1].axis('off')
            
################################################################################################################
################################################################################################################
################################################################################################################

class MaskedConv2d(nn.Module):
    """
    Implementation of masked convolution layer assuming color channels are dependent
    """
    def __init__(self, in_channels, out_channels, kernel_size, mask_type="A", device=DEVICE):
        """
        @param in_channels, out_channels, kernel_size: int
        @param mask_type
            Accepts type "A" or "B" (see figure 1 and 2)
        @ param device: string
        """
        super(MaskedConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.set_mask(kernel_size, mask_type, device)

    def set_mask(self, kernel_size, mask_type, device):
        """
        PixelCNN mask setup
        """
        out_dim, in_dim = self.conv.weight.shape[:2]
        self.mask = torch.ones(self.conv.weight.shape, device=device, requires_grad=False).float()
        one_third_in_dim, one_third_out_dim = in_dim // 3, out_dim // 3

        self.mask[:, :, 1 + kernel_size // 2:, :] = 0.0
        self.mask[:, :, kernel_size // 2, 1 + kernel_size // 2:] = 0.0
        self.mask[:one_third_out_dim, one_third_in_dim:, kernel_size // 2, kernel_size // 2] = 0.0
        self.mask[one_third_out_dim:2*one_third_out_dim, 2*one_third_in_dim:, kernel_size // 2, kernel_size // 2] = 0.0
        self.mask[2*one_third_out_dim:, 3*one_third_in_dim:, kernel_size // 2, kernel_size // 2] = 0.0

        if mask_type == "A":
            self.mask[:one_third_out_dim, :one_third_in_dim, kernel_size // 2, kernel_size // 2] = 0.0
            self.mask[one_third_out_dim:2*one_third_out_dim, one_third_in_dim:2*one_third_in_dim, kernel_size // 2, kernel_size // 2] = 0.0
            self.mask[2*one_third_out_dim:, 2*one_third_in_dim:3*one_third_in_dim, kernel_size // 2, kernel_size // 2] = 0.0

    def forward(self, x):
        """
        @param x: torch.tensor [B, C, H, W]
        """
        self.conv.weight.data *= self.mask
        return self.conv(x)
      
class ResidualBlock(nn.Module):
    """
    Implementation of residual block with fixed mask type B
    """
    def __init__(self, in_channels):
        """
        @param in_channels: int
        """
        super(ResidualBlock, self).__init__()
        self.net = nn.Sequential(MaskedConv2d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=1, mask_type="B"),
                                 nn.ReLU(),
                                 MaskedConv2d(in_channels=in_channels//2, out_channels=in_channels//2, kernel_size=7, mask_type="B"),
                                 nn.ReLU(),
                                 MaskedConv2d(in_channels=in_channels//2, out_channels=in_channels, kernel_size=1, mask_type="B"),
                                 nn.ReLU())

    def forward(self, x):
        """
        @param x: torch.tensor [B, C, H, W]
        """
        out = self.net(x)
        return x + out

class PixelCNN(nn.Module):
    """
    Implementation of PixelCNN for both grayscale and colored images
    """
    def __init__(self, in_channels, hidden_channels, n_classes, n_layers, height, width, color_independent=True):
        """
        @param in_channels, hidden_channels: int
        @param n_classes: int
            Number of output classes
        @param n_layers: int
            Number of Residual block layers
        @param height, width: int
        @param color_independent: boolean
            True if assume color channels are independent (default True)
        """
        super(PixelCNN, self).__init__()
        self.net = [MaskedConv2d(in_channels, hidden_channels, kernel_size=7, mask_type="A"),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU()]
        for _ in range(n_layers):
            self.net.extend([ResidualBlock(hidden_channels),
                             nn.BatchNorm2d(hidden_channels),
                             nn.ReLU()])
        self.net.append(MaskedConv2d(hidden_channels, in_channels*n_classes, kernel_size=1, mask_type="B"))
        self.net = nn.ModuleList(self.net)
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.height = height
        self.width = width
        self.color_independent = color_independent

    def forward(self, x):
        """
        @param x: torch.tensor [B, C, H, W]
        """
        x = (x.float() / (self.n_classes - 1) - 0.5) / 0.5
        for layer in self.net:
            x = layer(x)
        if self.color_independent:
            x = x.view(x.shape[0], self.n_classes, self.in_channels, x.shape[2], x.shape[3])
        else:
            x = x.view(x.shape[0], self.in_channels, self.n_classes, x.shape[2], x.shape[3]).permute(0, 2, 1, 3, 4)
        return x

    def sample(self, n_samples, device):
        """
        @param n_samples: int
            Number of samples to geneerate
        @param device: string
        """
        self.eval()
        with torch.no_grad():
            sample = torch.zeros(n_samples, self.in_channels, self.height, self.width).to(device).float()
            for i in range(self.height):
                for j in range(self.width):
                    for c in range(self.in_channels):
                        out = self.forward(sample)
                        logits = out[:, :, c, i, j]
                        prob = nn.Softmax(dim=1)(logits)
                        sample[:, c, i, j] = torch.multinomial(prob, 1).squeeze(-1)
            sample = sample.detach().cpu()
        return sample
      
def plot_samples(pixelcnn, vae, n_samples, device="cpu", title="Generated images"):
    vae.eval()
    samples = pixelcnn.sample(n_samples, DEVICE)
    samples = samples.long().view(-1)
    out = torch.index_select(vae.quantizer.dictionary, dim=0, index=samples.long().to(DEVICE))
    out = out.view(n_samples, 8, 8, vae.embed_dim).permute(0, 3, 1, 2)
    out = vae.decoder(out)
    out = out.permute(0, 2, 3, 1).contiguous()
    out = out.detach().cpu().numpy()
    fig, ax = plt.subplots(1, 6, figsize=(12, 2))
    fig.suptitle(title, fontsize=16)
    for i in range(6):
        ax[i].imshow(out[i], cmap="gray")
        ax[i].axis('off')
    plt.show()

def train_epoch(pixelcnn, vae, train_loader, optimizer, device='cpu'):
    pixelcnn.train()
    vae.eval()
    train_loss = 0.0
    for X, _ in train_loader:
        X = X.to(device)
        out = vae.encoder(X)
        out, _, _, target = vae.quantizer(out) #[B, E, 8, 8], need [B, 8, 8] labels
        target = target.unsqueeze(1)
        out = pixelcnn(target)
        optimizer.zero_grad()
        loss = nn.CrossEntropyLoss()(out, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss

def train(pixelcnn, vae, train_loader, optimizer, n_epochs, device='cpu'):
    for epoch in range(n_epochs):
        train_loss = train_epoch(pixelcnn, vae, train_loader, optimizer, device)
        plot_samples(pixelcnn, vae, 6, title=f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}')
        
def plot_samples(n_samples, vae, prior)
    vae.eval()
    samples = prior.sample(n_samples, DEVICE)
    samples = samples.long().view(-1)
    out = torch.index_select(model.quantizer.dictionary, dim=0, index=samples.long().to(DEVICE))
    out = out.view(n_samples, 8, 8, 128).permute(0, 3, 1, 2)
    out = vae.decoder(out)
    out = out.permute(0, 2, 3, 1).contiguous()
    out = out.detach().cpu().numpy()

    grid_size = int(np.sqrt(n_samples))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(6, 6))
    axes = axes.flatten()
    for i in range(n_samples):
        axes[i].imshow(out[i], cmap='gray')
        axes[i].axis('off')
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()
    


