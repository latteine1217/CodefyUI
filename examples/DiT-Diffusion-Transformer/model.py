"""
Diffusion Transformer (DiT) for Image Generation
=================================================

Architecture based on "Scalable Diffusion Models with Transformers"
(Peebles & Xie, 2023).

DiT replaces the U-Net backbone in diffusion models with a Transformer.
Instead of spatial convolutions and skip connections, DiT processes image
patches through a stack of Transformer blocks with adaptive layer norm
(adaLN-Zero) conditioning on timestep and class label.

The diffusion process:
  - Forward: gradually add Gaussian noise to images over T timesteps
  - Training: given a noisy image at timestep t, predict the added noise
  - Inference: iteratively denoise starting from pure Gaussian noise

Key components:
  - Patch embedding: Conv2d to project non-overlapping patches
  - Sinusoidal timestep embedding + class label embedding
  - DiT blocks with adaLN-Zero: adaptive LayerNorm modulated by conditioning
  - Unpatchify: reshape patch predictions back to image space
  - Simple DDPM noise schedule and training objective (MSE on noise)

Config:
  - patch_size=4, d_model=384, depth=6, heads=6 (small demo)
  - CIFAR-10 (32x32x3), T=1000 diffusion timesteps
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Create sinusoidal timestep embeddings (Vaswani et al.)."""
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
    emb = timesteps[:, None].float() * emb[None, :]
    return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class AdaLNZero(nn.Module):
    """
    Adaptive Layer Norm Zero (adaLN-Zero) conditioning.
    Projects conditioning vector to scale, shift, and gate parameters.
    """

    def __init__(self, d_model: int, cond_dim: int):
        super().__init__()
        self.silu = nn.SiLU()
        # 6 parameters: gamma1, beta1, alpha1, gamma2, beta2, alpha2
        self.linear = nn.Linear(cond_dim, 6 * d_model)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, cond: torch.Tensor):
        params = self.silu(cond)
        params = self.linear(params)
        return params.chunk(6, dim=-1)


class DiTBlock(nn.Module):
    """DiT block with adaLN-Zero conditioning."""

    def __init__(self, d_model: int, n_heads: int, cond_dim: int,
                 mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        mlp_dim = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, d_model),
            nn.Dropout(dropout),
        )
        self.adaln = AdaLNZero(d_model, cond_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.adaln(cond)

        # Modulated attention
        h = self.norm1(x) * (1 + gamma1.unsqueeze(1)) + beta1.unsqueeze(1)
        h = self.attn(h, h, h, need_weights=False)[0]
        x = x + alpha1.unsqueeze(1) * h

        # Modulated MLP
        h = self.norm2(x) * (1 + gamma2.unsqueeze(1)) + beta2.unsqueeze(1)
        x = x + alpha2.unsqueeze(1) * self.mlp(h)
        return x


class DiT(nn.Module):
    """Diffusion Transformer for image generation."""

    def __init__(self, img_size: int = 32, patch_size: int = 4,
                 in_channels: int = 3, d_model: int = 384,
                 depth: int = 6, n_heads: int = 6,
                 num_classes: int = 10, mlp_ratio: float = 4.0):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_patches = (img_size // patch_size) ** 2
        self.d_model = d_model

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, d_model) * 0.02)

        # Conditioning: timestep + class
        cond_dim = d_model
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.class_emb = nn.Embedding(num_classes + 1, cond_dim)  # +1 for uncond

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(d_model, n_heads, cond_dim, mlp_ratio)
            for _ in range(depth)
        ])

        # Final layer
        self.final_norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.final_adaln = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 2 * d_model))
        self.final_proj = nn.Linear(
            d_model, patch_size * patch_size * in_channels)

        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)

    def unpatchify(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Reshape patch predictions back to image shape."""
        p = self.patch_size
        x = x.reshape(-1, h, w, p, p, self.in_channels)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.reshape(-1, self.in_channels, h * p, w * p)

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                y: torch.Tensor) -> torch.Tensor:
        """
        x: noisy images (B, C, H, W)
        t: timesteps (B,)
        y: class labels (B,)
        Returns: predicted noise (B, C, H, W)
        """
        # Patch embed
        h_patches = x.shape[2] // self.patch_size
        w_patches = x.shape[3] // self.patch_size
        x = self.patch_embed(x)                            # (B, d_model, h, w)
        x = x.flatten(2).transpose(1, 2)                  # (B, num_patches, d_model)
        x = x + self.pos_embed

        # Conditioning
        t_emb = sinusoidal_embedding(t, self.d_model)
        cond = self.time_mlp(t_emb) + self.class_emb(y)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, cond)

        # Final projection
        gamma, beta = self.final_adaln(cond).chunk(2, dim=-1)
        x = self.final_norm(x) * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        x = self.final_proj(x)

        return self.unpatchify(x, h_patches, w_patches)


class SimpleDDPM:
    """Simple DDPM noise schedule and sampling utilities."""

    def __init__(self, T: int = 1000, beta_start: float = 1e-4,
                 beta_end: float = 0.02, device: torch.device = None):
        self.T = T
        self.device = device or torch.device("cpu")

        betas = torch.linspace(beta_start, beta_end, T, device=self.device)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alpha_bar = alpha_bar
        self.sqrt_alpha_bar = torch.sqrt(alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar)

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor,
                  noise: torch.Tensor) -> torch.Tensor:
        """Forward diffusion: q(x_t | x_0) = sqrt(alpha_bar_t)*x0 + sqrt(1-alpha_bar_t)*noise"""
        s_ab = self.sqrt_alpha_bar[t][:, None, None, None]
        s_omab = self.sqrt_one_minus_alpha_bar[t][:, None, None, None]
        return s_ab * x0 + s_omab * noise


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


if __name__ == "__main__":
    device = get_device()
    print(f"Using device: {device}")

    # Data
    transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # scale to [-1, 1]
    ])
    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True,
                              num_workers=2, pin_memory=True)

    # Model and diffusion
    T_steps = 1000
    model = DiT(
        img_size=32, patch_size=4, in_channels=3, d_model=384,
        depth=6, n_heads=6, num_classes=10, mlp_ratio=4.0,
    ).to(device)
    ddpm = SimpleDDPM(T=T_steps, device=device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"DiT model parameters: {param_count:,}")

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)

    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        total_loss, num_batches = 0.0, 0

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            B = images.shape[0]

            # Sample random timesteps and noise
            t = torch.randint(0, T_steps, (B,), device=device)
            noise = torch.randn_like(images)
            noisy_images = ddpm.add_noise(images, t, noise)

            # Predict noise
            predicted_noise = model(noisy_images, t, labels)
            loss = F.mse_loss(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if (i + 1) % 100 == 0:
                print(f"  Epoch {epoch+1} | Step {i+1}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f}")

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs} | Avg Loss: {avg_loss:.4f}")

    print("Training complete.")
