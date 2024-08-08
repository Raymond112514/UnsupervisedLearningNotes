import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import math
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SelfAttention(nn.Module):
    def __init__(self, in_channels, n_heads):
        super(SelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(in_channels, n_heads, batch_first=True)
        self.layernorm = nn.LayerNorm([in_channels])

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).view(B, H*W, C)
        x = self.layernorm(x)
        attn, _ = self.attention(x, x, x)
        x = (x + attn).view(B, H, W, C).permute(0, 3, 1, 2)
        return x

class DoubleConv2d(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, residual=False):
        super(DoubleConv2d, self).__init__()
        self.net = nn.Sequential(nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
                                 nn.GroupNorm(1, hidden_channels),
                                 nn.GELU(),
                                 nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1, bias=False),
                                 nn.GroupNorm(1, out_channels))
        self.residual = residual
    
    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.net(x))
        return self.net(x)

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, embed_dim):
        super(DownBlock, self).__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2),
                                 DoubleConv2d(in_channels, in_channels, in_channels, residual=True),
                                 DoubleConv2d(in_channels, out_channels, out_channels))
        self.resize = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, out_channels))

    def forward(self, x, t):
        x = self.net(x)
        t = self.resize(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + t

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, embed_dim):
        super(UpBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.net = nn.Sequential(DoubleConv2d(in_channels, in_channels, in_channels, residual=True),
                                 DoubleConv2d(in_channels, in_channels // 2, out_channels))
        self.resize = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, out_channels))

    def forward(self, x, x_skip, t):
        x = self.upsample(x)
        x = torch.cat([x_skip, x], dim=1)
        x = self.net(x)
        t = self.resize(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + t

def timestep_embedding(timesteps, dim, max_period=10000):
    half_dim = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half_dim, dtype=torch.float32, device=DEVICE) / half_dim)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
  
class Diffusion(nn.Module):

    def __init__(self, in_channels, embed_dim, n_heads, img_size, timesteps, beta_min, beta_max, device="cpu"):
        super(Diffusion, self).__init__()
        self.unet = UNetConditional(in_channels, embed_dim, n_heads, timesteps, n_classes=10, device=device)
        
        self.beta = torch.linspace(beta_min, beta_max, timesteps).to(device)
        self.alpha = 1 - self.beta
        self.alpha_ = torch.cumprod(self.alpha, dim=0)
        
        self.img_size = img_size
        self.in_channels = in_channels
        self.timesteps = timesteps
        self.device = device

    def forward(self, x, t):
        noise = torch.randn_like(x, device=self.device).to(self.device)
        noised_x = torch.sqrt(self.alpha_[t])[:, None, None, None] * x + torch.sqrt(1 - self.alpha_[t])[:, None, None, None] * noise
        return noised_x, noise

    def sample(self, n_samples, labels, scale=3):
        self.unet.eval()
        with torch.no_grad():
            x = torch.randn(n_samples, self.in_channels, self.img_size, self.img_size).to(self.device)
            for t in reversed(range(1, self.timesteps)):
                time = (t * torch.ones(n_samples, device=self.device)).long()
                cond_pred_noise = self.unet(x, time, labels)
                uncond_pred_noise = self.unet(x, time, None)
                pred_noise = torch.lerp(uncond_pred_noise, cond_pred_noise, scale)
                alpha, alpha_, beta = self.alpha[time], self.alpha_[time], self.beta[time]
                alpha = alpha[:, None, None, None]
                alpha_ = alpha_[:, None, None, None]
                beta = beta[:, None, None, None]
                noise = torch.randn(x.shape) if t > 1 else torch.zeros_like(x)
                noise = noise.to(self.device)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_))) * pred_noise) + torch.sqrt(beta) * noise
        self.unet.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x
      
def plot_samples(samples, title="Generated images"):
    samples = samples.permute(0, 2, 3, 1)
    samples = samples.detach().cpu().numpy()
    fig, ax = plt.subplots(1, 6, figsize=(12, 2))
    fig.suptitle(title, fontsize=16)
    for i in range(6):
        ax[i].imshow(samples[i])
        ax[i].axis('off')
    plt.show()

def train_epoch(model, dataloader, optimizer, device="cpu"):
    model.train()
    model.unet.train()
    train_loss = 0.0
    for i, (X, labels) in enumerate(dataloader):
        t = torch.randint(1, model.timesteps, (X.shape[0], ), device=device)
        X = X.to(device).float()
        labels = labels.to(device).long()
        X_noised, noise = model(X, t)
        time = t
        if np.random.random() < 0.1:
            labels = None
        pred_noise = model.unet(X_noised, time, labels)
        optimizer.zero_grad()
        loss = nn.MSELoss()(noise, pred_noise)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() 
    return train_loss / len(dataloader)  

def train(model, dataloader, optimizer, n_epoch, device="cpu"):
    for epoch in range(n_epoch):
        train_loss = train_epoch(model, dataloader, optimizer, device)
        labels = torch.randint(0, 10, (6, )).to(device)
        samples = model.sample(6, labels)
        plot_samples(samples, f"Epoch {epoch+1} Generated images, train loss: {train_loss}")