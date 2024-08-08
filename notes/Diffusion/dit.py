

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)
    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product
    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def get_2d_sincos_pos_embed(embed_dim, grid_size):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def timestep_embedding(timesteps, dim, max_period=10000, device='cpu'):
    half_dim = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half_dim, dtype=torch.float32, device=device) / half_dim)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
  
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, n_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.n_heads = n_heads
        self.embed_dim = embed_dim

        ## Pre self-attention linear projection
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        ## x: [B, L, E], mask: [L, L]
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        q = q.reshape(q.shape[0], q.shape[1], self.n_heads, self.embed_dim // self.n_heads)    ## [B, H, L, E / H]
        k = k.reshape(k.shape[0], k.shape[1], self.n_heads, self.embed_dim // self.n_heads)    ## [B, H, L, E / H]
        v = v.reshape(v.shape[0], v.shape[1], self.n_heads, self.embed_dim // self.n_heads)    ## [B, H, L, E / H]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  ## [B, H, L, L]
        mask = torch.tril(torch.ones(x.shape[1], x.shape[1], device=x.device, requires_grad=False))
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, L, L]
        scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        out = torch.matmul(weights, v)    ## [B, H, L, E / H]
        out = out.permute(0, 2, 1, 3)     ## [B, L, H, E / H]
        out = out.reshape(out.shape[0], out.shape[1], -1)  ## [B, L, E]
        return out
      
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
    model.dit.train()
    train_loss = 0.0
    for i, (X, labels) in enumerate(dataloader):
        t = torch.randint(1, model.timesteps, (X.shape[0], ), device=device)
        X = X.to(device).float()
        X = encode(X) / STD
        labels = labels.to(device).long()
        X_noised, noise = model(X, t)
        time = t
        if np.random.random() < 0.1:
            labels = None
        pred_noise = model.dit(X_noised, time, labels)
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
        samples = decode(samples * STD)
        samples = torch.clamp((samples + 1) / 2, 0, 1)
        plot_samples(samples, f"Epoch {epoch+1} Generated images, train loss: {train_loss}")
        
class Diffusion(nn.Module):

    def __init__(self, img_size, patch_dim, hidden_dim, n_heads, n_layers, n_classes, timesteps, beta_min, beta_max, device="cpu"):
        super(Diffusion, self).__init__()
        self.height, self.width, self.channels = img_size
        self.dit = DiT(height=self.height,
                       width=self.width,
                       patch_dim=patch_dim,
                       hidden_dim=hidden_dim,
                       out_channels=self.channels,
                       n_heads=n_heads,
                       n_layers=n_layers,
                       n_classes=n_classes, device=device)
        self.beta = torch.linspace(beta_min, beta_max, timesteps).to(device)
        self.alpha = 1 - self.beta
        self.alpha_ = torch.cumprod(self.alpha, dim=0)
        self.timesteps = timesteps
        self.device = device

    def forward(self, x, t):
        noise = torch.randn_like(x, device=self.device).to(self.device)
        noised_x = torch.sqrt(self.alpha_[t])[:, None, None, None] * x + torch.sqrt(1 - self.alpha_[t])[:, None, None, None] * noise
        return noised_x, noise

    def sample(self, n_samples, labels, scale=3):
        self.dit.eval()
        with torch.no_grad():
            x = torch.randn(n_samples, self.channels, self.height, self.width).to(self.device)
            for t in reversed(range(1, self.timesteps)):
                time = (t * torch.ones(n_samples, device=self.device)).long()
                cond_pred_noise = self.dit(x, time, labels)
                uncond_pred_noise = self.dit(x, time, None)
                pred_noise = torch.lerp(uncond_pred_noise, cond_pred_noise, scale)
                alpha, alpha_, beta = self.alpha[time], self.alpha_[time], self.beta[time]
                alpha = alpha[:, None, None, None]
                alpha_ = alpha_[:, None, None, None]
                beta = beta[:, None, None, None]
                noise = torch.randn(x.shape) if t > 1 else torch.zeros_like(x)
                noise = noise.to(self.device)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_))) * pred_noise) + torch.sqrt(beta) * noise
        self.dit.train()
        return x
      
def plot_samples(n_samples):
    model.eval()
    samples = model.sample(n_samples, labels=torch.randint(0, 10, (36, )).to(DEVICE), scale=5)
    samples = decode(samples * STD)
    samples = torch.clamp((samples + 1) / 2, 0, 1).permute(0, 2, 3, 1)
    samples = samples.detach().cpu().numpy()

    grid_size = int(np.sqrt(n_samples))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(6, 6))
    axes = axes.flatten()
    for i in range(n_samples):
        axes[i].imshow(samples[i])
        axes[i].axis('off')
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()
    
def plot_cfg(scale):
    fig, ax = plt.subplots(5, 10, figsize=(8, 4))
    fig.suptitle(f'Generated Samples with scale={scale}', fontsize=12)
    for i in range(5):
        samples = model.sample(10, torch.tensor(np.arange(10), device=DEVICE), scale=scale)
        samples = decode(samples * STD)
        samples = torch.clamp((samples + 1) / 2, 0, 1).permute(0, 2, 3, 1)
        samples = samples.detach().cpu().numpy()
        for j in range(10):
            ax[i, j].imshow(samples[j], cmap="gray")
            ax[i, j].axis("off")
    plt.subplots_adjust(top=0.9)