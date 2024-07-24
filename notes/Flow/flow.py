import numpy as np
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.distributions import Normal
from torch.distributions.uniform import Uniform
import torch.optim as optim
from scipy.interpolate import interp1d

def generate_data(n_samples):
    mu = [-1, 0.2, 3.2]
    sigma = [1.5, 0.8, 1.1]
    weight = [0.7, 2.1, 1]
    mixture = []
    for i in range(len(mu)):
        mixture.extend(weight[i] * np.random.normal(loc=mu[i], scale=sigma[i], size=(n_samples // 3,)))
    train_data = torch.tensor(mixture)
    train_dataset = TensorDataset(train_data)
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    plt.hist(mixture, bins=50)
    plt.title("Mixture of 3 Gaussian")
    plt.show()
    return train_dataloader
  
  
def train(model, train_data, z_dist, epochs=10, for_every=20, lr=0.01):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    N = len(train_data)
    train_losses = []
    for epoch in range(epochs):
        train_loss = 0
        for X in train_data:
            X = X[0].unsqueeze(-1)
            optimizer.zero_grad()
            z, dz = model(X)
            loss = - (z_dist.log_prob(z) + dz.log()).mean()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= N
        train_losses.append(train_loss)
        print(f"Epoch {epoch} training loss: {train_loss}") if epoch % for_every == 0 else None
    return train_losses