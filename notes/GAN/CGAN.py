import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def plot_samples(samples, m=1, n=6, title="Generated samples"):
    fig, ax = plt.subplots(m, n, figsize=(10, 2))
    ax = ax.flatten()
    for i in range(m*n):
        ax[i].imshow(samples[i])
        ax[i].axis("off")
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.show()

def train_epoch(model, optimizerD, optimizerG, train_loader, epoch_num=1, device=DEVICE, sample=True):
    model.train()
    model.to(device)
    discriminator = model.discriminator
    generator = model.generator
    discriminator_loss = 0.0
    generator_loss = 0.0

    for i, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)

        # Train the discriminator
        optimizerD.zero_grad()

        # Fake data
        noise = torch.randn(X.size(0), model.noise_dim).to(device)
        y_fake = torch.randint(0, model.n_classes, (X.size(0),)).to(device)
        X_fake = generator(noise, y_fake)
        D_fake = discriminator(X_fake, y_fake)
        D_fake_loss = nn.BCEWithLogitsLoss()(D_fake, torch.zeros_like(D_fake))

        # Real data
        D_real = discriminator(X, y)
        D_real_loss = nn.BCEWithLogitsLoss()(D_real, torch.ones_like(D_real))

        # Backward and optimize
        D_loss = (D_real_loss + D_fake_loss) / 2
        D_loss.backward()
        optimizerD.step()

        # Generate fake data again for generator update
        optimizerG.zero_grad()
        noise = torch.randn(X.size(0), model.noise_dim).to(device)
        y_fake = torch.randint(0, model.n_classes, (X.size(0),)).to(device)
        X_fake = generator(noise, y_fake)
        D_fake = discriminator(X_fake, y_fake)
        G_loss = nn.BCEWithLogitsLoss()(D_fake, torch.ones_like(D_fake))
        G_loss.backward()
        optimizerG.step()

        discriminator_loss += D_loss.item()
        generator_loss += G_loss.item()

    discriminator_loss /= len(train_loader)
    generator_loss /= len(train_loader)
    discriminator_loss = round(discriminator_loss, 5)
    generator_loss = round(generator_loss, 5)

    if sample:
        labels = torch.tensor([0, 1, 2, 3, 4, 5]).to(device)
        samples = model.sample(6, labels, device=DEVICE)
        plot_samples(samples, title=f"Epoch {epoch_num} generated samples with D loss: {discriminator_loss}, G loss: {generator_loss}")

def train(model, train_loader, optimizerD, optimizerG, epochs):
    for epoch in range(epochs):
        train_epoch(model, optimizerD, optimizerG, train_loader, epoch_num=epoch, sample=True)
        
def plot_samples(n_samples):
    fig, ax = plt.subplots(n_samples, n_samples, figsize=(8, 8))

    for i in range(n_samples):
        samples = cgan.sample(n_samples, torch.tensor(np.arange(10), device=DEVICE), device=DEVICE)
        for j in range(n_samples):
            ax[i, j].imshow(samples[j], cmap='gray')
            ax[i, j].axis("off")
            