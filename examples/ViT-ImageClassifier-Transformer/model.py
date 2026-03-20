"""
Vision Transformer (ViT) for CIFAR-10 Image Classification
===========================================================

Architecture based on "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020).

The Vision Transformer applies the standard Transformer encoder (originally
designed for NLP) directly to sequences of image patches. An image is split
into fixed-size patches, each patch is linearly embedded, position embeddings
are added, and the resulting sequence is fed to a standard Transformer encoder.
A learnable [CLS] token is prepended to the sequence; its final representation
serves as the image representation for classification.

Key components:
  - PatchEmbedding: Conv2d projection of non-overlapping patches + CLS token
  - Positional Embedding: learnable 1D positional encoding
  - Transformer Encoder: 12 layers of multi-head self-attention + MLP
  - Classification Head: MLP on the [CLS] token output

Config (ViT-Base adapted for CIFAR-10 32x32 images with patch_size=4):
  - d_model=768, heads=12, layers=12, mlp_dim=3072
  - patch_size=4 (yields 8x8=64 patches from 32x32 images)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T


class PatchEmbedding(nn.Module):
    """Project image patches into embedding space using a single Conv2d."""

    def __init__(self, in_channels: int = 3, patch_size: int = 4,
                 embed_dim: int = 768, img_size: int = 32):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches + 1, embed_dim) * 0.02
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.proj(x)                          # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)          # (B, num_patches, embed_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)     # (B, num_patches+1, embed_dim)
        x = x + self.pos_embed
        return x


class TransformerBlock(nn.Module):
    """Standard pre-norm Transformer encoder block."""

    def __init__(self, d_model: int = 768, num_heads: int = 12,
                 mlp_dim: int = 3072, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads,
                                          dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    """Vision Transformer for image classification."""

    def __init__(self, img_size: int = 32, patch_size: int = 4,
                 in_channels: int = 3, num_classes: int = 10,
                 d_model: int = 768, depth: int = 12, num_heads: int = 12,
                 mlp_dim: int = 3072, dropout: float = 0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_size,
                                          d_model, img_size)
        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.Sequential(
            *[TransformerBlock(d_model, num_heads, mlp_dim, dropout)
              for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)       # (B, num_patches+1, d_model)
        x = self.dropout(x)
        x = self.encoder(x)
        x = self.norm(x)
        cls_out = x[:, 0]             # [CLS] token
        return self.head(cls_out)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for i, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

        if (i + 1) % 100 == 0:
            print(f"  Epoch {epoch+1} | Step {i+1}/{len(loader)} | "
                  f"Loss: {loss.item():.4f}")

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        running_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


if __name__ == "__main__":
    device = get_device()
    print(f"Using device: {device}")

    # Data augmentation and normalization for CIFAR-10
    train_transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_transform)
    test_set = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False,
                             num_workers=2, pin_memory=True)

    # ViT-Base config adapted for CIFAR-10 (patch_size=4 for 32x32 images)
    model = ViT(
        img_size=32, patch_size=4, in_channels=3, num_classes=10,
        d_model=768, depth=12, num_heads=12, mlp_dim=3072, dropout=0.1,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {param_count:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    num_epochs = 20
    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Test  Loss: {test_loss:.4f} Acc: {test_acc:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")

    print("Training complete.")
