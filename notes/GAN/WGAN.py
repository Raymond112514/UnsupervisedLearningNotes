class Discriminator(nn.Module):
    """
    Input: [B, 3, 64, 64]
    Output: [B, ]
    """
    def __init__(self, in_channels=3, weight_init=False):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
        )
        if weight_init:
            self.apply(self._initialize_weights)

    def forward(self, x):
        x = self.net(x).view(-1, 1).squeeze(1)
        return nn.Sigmoid()(x)

    def _initialize_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)

class Generator(nn.Module):
    """
    Input: [B, 100, 1, 1]
    Output: [B, 3, 64, 54]
    """
    def __init__(self, out_channels=3, weight_init=False):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, out_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        if weight_init:
            self.apply(self._initialize_weights)

    def forward(self, z):
        return self.net(z)

    def _initialize_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)

def plot_samples(samples, m=1, n=6, title="Generated samples"):
    fig, ax = plt.subplots(m, n, figsize=(10, 2))
    ax = ax.flatten()
    for i in range(m*n):
        ax[i].imshow(samples[i])
        ax[i].axis("off")
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.show()

def train_epoch(model, optimizerD, optimizerG, train_loader, epoch_num=1, device=DEVICE, sample=True, n_critic=1):
    model.train()
    discriminator = model.discriminator
    generator = model.generator
    discriminator_loss = 0.0
    generator_loss = 0.0
    for i, (X, _) in enumerate(train_loader):
        X = X.to(DEVICE)
        mean_D_loss = 0.0
        optimizerD.zero_grad()

        for _ in range(n_critic):
            noise = torch.randn(X.size(0), 100, 1, 1).to(DEVICE)
            X_fake = generator(noise)
            D_fake = discriminator(X_fake.detach())
            D_real = discriminator(X)
            W_D_loss = torch.mean(D_fake - D_real)
            W_D_loss.backward(retain_graph=True)
            optimizerD.step()

            for p in discriminator.parameters():
                p.data.clamp_(-0.01, 0.01)

        optimizerG.zero_grad()
        X_fake = generator(noise)
        D_fake = discriminator(X_fake)
        W_G_loss = -torch.mean(D_fake)
        W_G_loss.backward()
        optimizerG.step()

        discriminator_loss += W_D_loss.item()
        generator_loss += W_G_loss.item()
    discriminator_loss /= len(train_loader)
    generator_loss /= len(train_loader)
    discriminator_loss = round(discriminator_loss, 5)
    generator_loss = round(generator_loss, 5)

    if sample:
        samples = model.sample(6, device=DEVICE)
        plot_samples(samples, title=f"Epoch {epoch_num} generated samples with D loss: {discriminator_loss}, G loss: {generator_loss}")

def train(model, train_loader, optimizerD, optimizerG, epochs, n_critic=1):
    for epoch in range(epochs):
        train_epoch(model, optimizerD, optimizerG, train_loader, epoch_num=epoch, sample=True, n_critic=n_critic)



def plot_samples(n_samples):
    model.eval()
    samples = model.sample(36, device=DEVICE)
    fig, ax = plt.subplots(6, 6, figsize=(6, 6))
    ax = ax.flatten()
    for i in range(36):
        ax[i].imshow(samples[i])
        ax[i].axis("off")
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()