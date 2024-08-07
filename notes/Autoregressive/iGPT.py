


def plot_samples(samples, m=1, n=6, title="Generated samples"):
    fig, ax = plt.subplots(m, n, figsize=(10, 2))
    ax = ax.flatten()
    for i in range(m*n):
        ax[i].imshow(samples[i], cmap="gray")
        ax[i].axis("off")
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.show()

def train_epoch(model, dataloader, optimizer, criterion, sample=True, epoch_num=0):
    train_loss = 0.0
    for X, _ in dataloader:
        optimizer.zero_grad()
        X = X.to(DEVICE).long()
        output = model(X[:, :-1])
        loss = criterion(output.reshape(-1, 3), X[:, 1:].reshape(-1))
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    train_loss /= len(dataloader)
    if sample:
        with torch.no_grad():
            sample = torch.fill(torch.zeros(6, 1), 2).long().to(DEVICE)
            for i in range(28**2):
                prob = model(sample)
                prob = F.softmax(prob, dim=-1)
                output = torch.multinomial(prob[:, i], 1)
                sample = torch.cat((sample, output), dim=1)
            samples = sample[:, 1:].view(6, 28, 28)
            samples = samples.detach().cpu().numpy()
            plot_samples(samples, title=f"Epoch {epoch_num} generated samples with loss: {train_loss}")
    return train_loss

def train(model, dataloader, optimizer, criterion, epochs=10):
    for epoch in range(epochs):
        train_epoch(model, dataloader, optimizer, criterion, epoch_num=epoch+1)
        
def plot_mnist_samples(n_samples)
    samples = model.sample(n_samples, 28)

    fig, axes = plt.subplots(int(n_samples**0.5), int(n_samples**0.5), figsize=(6, 6))
    axes = axes.flatten()
    for i in range(n_samples):
        axes[i].imshow(samples[i].reshape(28, 28), cmap='gray')
        axes[i].axis('off')
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()
    
def plot_cifar_samples(n_samples=36)
    model.eval()
    with torch.no_grad():
        sample = torch.full((n_samples,), 512).unsqueeze(1).long().to(DEVICE)
        for i in range(32**2):
            prob = model(sample)
            prob = F.softmax(prob, dim=-1)
            output = torch.multinomial(prob[:, i], 1)
            sample = torch.cat((sample, output), dim=1)
        samples = sample[:, 1:].reshape(36, -1)
        samples = samples.detach().cpu().numpy()
        samples[samples == 512] = 0
        samples = get_image_from_index(samples, mbkmeans, 36, 3, 32, 32).permute(0, 2, 3, 1)
        samples = (samples + 1) / 2

    fig, ax = plt.subplots(6, 6, figsize=(6, 6))
    for i in range(6):
        for j in range(6):
            ax[i][j].imshow(samples[6*i+j])
            ax[i][j].axis('off')
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()