



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
    for i, (X, _) in enumerate(dataloader):
        t = torch.randint(1, model.timesteps, (X.shape[0], ), device=device)
        X = X.to(device).float()
        X_noised, noise = model(X, t)
        time = t
        pred_noise = model.unet(X_noised, time)
        optimizer.zero_grad()
        loss = nn.MSELoss()(noise, pred_noise)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() 
    return train_loss / len(dataloader)  

def train(model, dataloader, optimizer, n_epoch, device="cpu"):
    for epoch in range(n_epoch):
        train_loss = train_epoch(model, dataloader, optimizer, device)
        samples = model.sample(6)
        plot_samples(samples, f"Epoch {epoch+1} Generated images, train loss: {train_loss}")
        
def plot_samples(n_samples):
    model.eval()
    samples = model.sample(n_samples).permute(0, 2, 3, 1)
    samples = samples.detach().cpu().numpy()

    grid_size = int(np.sqrt(n_samples))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(6, 6))
    axes = axes.flatten()
    for i in range(n_samples):
        axes[i].imshow(samples[i], cmap='gray')
        axes[i].axis('off')
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()