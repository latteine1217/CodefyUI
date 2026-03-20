"""
Swin Transformer for CIFAR-10 Image Classification
===================================================

Architecture based on "Swin Transformer: Hierarchical Vision Transformer
using Shifted Windows" (Liu et al., 2021).

Unlike ViT, which computes global self-attention over all patches, Swin
Transformer computes attention within local windows and shifts those windows
between layers to enable cross-window information flow. This gives linear
computational complexity with respect to image size (vs quadratic for ViT).

Key innovations:
  - Window-based multi-head self-attention (W-MSA): attention within fixed
    non-overlapping windows of size M x M
  - Shifted window attention (SW-MSA): shift the window partition by (M/2, M/2)
    pixels in alternating layers for cross-window connectivity
  - Hierarchical feature maps with patch merging: reduces spatial resolution
    while increasing channels (like a CNN backbone)

Config (adapted for CIFAR-10 32x32):
  - patch_size=2, window_size=4
  - 4 stages with depths [2, 2, 6, 2] and dims [96, 192, 384, 768]
  - Heads per stage: [3, 6, 12, 24]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Partition feature map into non-overlapping windows."""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return windows.view(-1, window_size, window_size, C)


def window_reverse(windows: torch.Tensor, window_size: int,
                   H: int, W: int) -> torch.Tensor:
    """Reverse window partition back to feature map."""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size,
                     window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


class WindowAttention(nn.Module):
    """Window-based multi-head self-attention with relative position bias."""

    def __init__(self, dim: int, window_size: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Compute relative position index
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flat = coords.view(2, -1)
        relative_coords = coords_flat[:, :, None] - coords_flat[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        B_, N, C = x.shape  # B_ = num_windows * B
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2)   # (B_, heads, N, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Add relative position bias
        bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, -1).permute(2, 0, 1).contiguous()
        attn = attn + bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(out)


class SwinBlock(nn.Module):
    """Swin Transformer block: W-MSA or SW-MSA + MLP."""

    def __init__(self, dim: int, num_heads: int, window_size: int = 4,
                 shift_size: int = 0, mlp_ratio: float = 4.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden), nn.GELU(),
            nn.Linear(mlp_hidden, dim),
        )

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, L, C = x.shape
        x_2d = x.view(B, H, W, C)

        # Cyclic shift for SW-MSA
        if self.shift_size > 0:
            shifted = torch.roll(x_2d, shifts=(-self.shift_size, -self.shift_size),
                                 dims=(1, 2))
            attn_mask = self._compute_mask(H, W, x.device)
        else:
            shifted = x_2d
            attn_mask = None

        # Window partition -> attention -> reverse
        windows = window_partition(shifted, self.window_size)
        windows = windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted = window_reverse(attn_windows, self.window_size, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x_2d = torch.roll(shifted, shifts=(self.shift_size, self.shift_size),
                              dims=(1, 2))
        else:
            x_2d = shifted

        # Residual connections
        x = x + x_2d.view(B, L, C)
        x = x + self.mlp(self.norm2(x))
        return x

    def _compute_mask(self, H: int, W: int, device: torch.device):
        """Compute attention mask for shifted window partition."""
        img_mask = torch.zeros(1, H, W, 1, device=device)
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        return attn_mask.masked_fill(attn_mask != 0, -100.0)


class PatchMerging(nn.Module):
    """Merge 2x2 neighboring patches to reduce spatial resolution by 2x."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor, H: int, W: int):
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = x.view(B, -1, 4 * C)
        return self.reduction(self.norm(x)), H // 2, W // 2


class SwinStage(nn.Module):
    """A stage of Swin blocks with optional patch merging."""

    def __init__(self, dim: int, depth: int, num_heads: int,
                 window_size: int, downsample: bool = True):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinBlock(dim, num_heads, window_size,
                      shift_size=0 if (i % 2 == 0) else window_size // 2)
            for i in range(depth)
        ])
        self.downsample = PatchMerging(dim) if downsample else None

    def forward(self, x: torch.Tensor, H: int, W: int):
        for block in self.blocks:
            x = block(x, H, W)
        if self.downsample is not None:
            x, H, W = self.downsample(x, H, W)
        return x, H, W


class SwinTransformer(nn.Module):
    """Swin Transformer for image classification."""

    def __init__(self, img_size: int = 32, patch_size: int = 2,
                 in_channels: int = 3, num_classes: int = 10,
                 embed_dim: int = 96, depths: tuple = (2, 2, 6, 2),
                 num_heads: tuple = (3, 6, 12, 24), window_size: int = 4):
        super().__init__()
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm_embed = nn.LayerNorm(embed_dim)
        self.patches_h = img_size // patch_size
        self.patches_w = img_size // patch_size

        self.stages = nn.ModuleList()
        dim = embed_dim
        for i in range(len(depths)):
            stage = SwinStage(
                dim=dim, depth=depths[i], num_heads=num_heads[i],
                window_size=window_size, downsample=(i < len(depths) - 1),
            )
            self.stages.append(stage)
            if i < len(depths) - 1:
                dim *= 2

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)                    # (B, C, H', W')
        x = x.flatten(2).transpose(1, 2)          # (B, H'*W', C)
        x = self.norm_embed(x)

        H, W = self.patches_h, self.patches_w
        for stage in self.stages:
            x, H, W = stage(x, H, W)

        x = self.norm(x)
        x = x.mean(dim=1)                         # global average pool
        return self.head(x)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


if __name__ == "__main__":
    device = get_device()
    print(f"Using device: {device}")

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

    model = SwinTransformer(
        img_size=32, patch_size=2, in_channels=3, num_classes=10,
        embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
        window_size=4,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Swin Transformer parameters: {param_count:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

            if (i + 1) % 100 == 0:
                print(f"  Epoch {epoch+1} | Step {i+1}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f}")

        scheduler.step()
        train_loss = running_loss / total
        train_acc = correct / total

        # Evaluate
        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                preds = model(images).argmax(1)
                test_correct += (preds == labels).sum().item()
                test_total += labels.size(0)

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Test Acc: {test_correct/test_total:.4f}")

    print("Training complete.")
