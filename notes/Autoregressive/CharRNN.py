import torch
import torch.nn as nn
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt

#####################################################################################################################################################################################################################################################   Training loop                                                                                             #########
##################################################################################################################################################################################################################################################  
def train(model, train_loader, n_epochs, lr=1e-6, device="cpu"):
    """ 
    Training loop for CharRNN
    """
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    for epoch in range(n_epochs):
        average_loss = 0
        n = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(X.float())
            loss = criterion(y_pred.view(-1, 2), y.view(-1))
            loss.backward()
            optimizer.step()
            average_loss += loss.item()
            n += 1
        print(f"Epoch {epoch + 1} average loss: {average_loss / n}")

def train_with_pos(model, train_loader, n_epochs, lr=1e-6, device="cpu"):
    """
    Trains a charrnn model with positional encoding
    We stack the (x, y) coordinate before one hot encoded inputs
    The input takes in the form [x, y, 0, 1] or [x, y, 1, 0]
    """
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    x_coords = torch.arange(28).repeat_interleave(28)
    y_coords = torch.arange(28).repeat(28)
    position = torch.stack((x_coords, y_coords), dim=1).float()[:-1]
    position_ = position.unsqueeze(0).repeat(128, 1, 1).to(device)
    for epoch in range(n_epochs):
        average_loss = 0
        n = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            position_ = position.unsqueeze(0).repeat(X.shape[0], 1, 1).to(device)
            X = torch.cat((position_, X), dim=2)
            optimizer.zero_grad()
            y_pred = model(X.float())
            loss = criterion(y_pred.view(-1, 2), y.view(-1))
            loss.backward()
            optimizer.step()
            average_loss += loss.item()
            n += 1
        print(f"Epoch {epoch + 1} average loss: {average_loss / n}")
        
#####################################################################################################################################################################################################################################################   Utility fucntion                                                                                          #########
##################################################################################################################################################################################################################################################
        
def plot_samples(samples):
    """  
    Takes in a list of samples and display in a 6X6 grid
    """
    grid_size = int(np.sqrt(num_samples))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(6, 6))
    axes = axes.flatten()
    for i in range(num_samples):
        axes[i].imshow(samples[i], cmap='gray')
        axes[i].axis('off')
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()
  
def shiftImage(image, pixel):
    """ 
    Shift a image 
    """
    image = image.flatten()
    image = np.roll(image, pixel)
    image = image.reshape(28, 28)
    return image
  
def sample_with_position(model, n_samples, max_len=28**2, vocab_size=2, device="cpu"):
    """
    Generates samples with positional encoding
    """
    samples = torch.zeros((n_samples, max_len), dtype=torch.long).to(device)
    x_coords = torch.arange(int(max_len**0.5)).repeat_interleave(int(max_len**0.5))
    y_coords = torch.arange(int(max_len**0.5)).repeat(int(max_len**0.5))
    position = torch.stack((x_coords, y_coords), dim=1).float()[:-1]
    position_ = position.unsqueeze(0).repeat(n_samples, 1, 1).to(device)
    for i in range(1, max_len):
        x = samples[:, :i]
        x = F.one_hot(x, num_classes=vocab_size).float().to(device)
        x = torch.cat((position_[:, :i], x), dim=2)
        logits = model(x)
        logits = logits[:, -1, :]
        prob = F.softmax(logits, dim=1)
        epsilon = torch.rand(n_samples, 1).to(device)
        characters = (epsilon >= prob[:, :1]).long()
        samples[:, i] = characters.squeeze(1)
    samples = samples.view(n_samples, int(max_len**0.5), int(max_len**0.5))
    samples = samples.detach().cpu().numpy()
    return samples