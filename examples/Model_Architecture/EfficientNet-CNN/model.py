"""
EfficientNet-B0 for CIFAR-10
=============================
EfficientNet (Tan & Le, "EfficientNet: Rethinking Model Scaling for CNNs", Google 2019)
introduced compound scaling -- uniformly scaling depth, width, and resolution with a
fixed ratio -- to achieve better accuracy/efficiency trade-offs than ad-hoc scaling.

The backbone uses Mobile Inverted Bottleneck Convolution (MBConv) blocks:
    input -> 1x1 expand -> depthwise k x k -> SE attention -> 1x1 project (+residual)

Key innovations:
  - Inverted bottlenecks (expand then project, opposite of standard bottlenecks)
  - Depthwise separable convolutions for parameter efficiency
  - Squeeze-and-Excitation (SE) modules for channel-wise attention
  - Swish/SiLU activation throughout

B0 stage configs: (expand_ratio, channels, num_blocks, stride, kernel_size)
    Stage 1: (1, 16, 1, 1, 3)    Stage 5: (6, 112, 3, 1, 5)
    Stage 2: (6, 24, 2, 2, 3)    Stage 6: (6, 192, 4, 2, 5)
    Stage 3: (6, 40, 2, 2, 5)    Stage 7: (6, 320, 1, 1, 3)
    Stage 4: (6, 80, 3, 2, 3)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import torchvision
import torchvision.transforms as T


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class SqueezeExcitation(nn.Module):
    """Channel attention via global squeeze and per-channel excitation."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(1, channels // reduction)
        self.fc1 = nn.Conv2d(channels, mid, 1)
        self.fc2 = nn.Conv2d(mid, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = x.mean(dim=(2, 3), keepdim=True)   # global avg pool
        scale = F.silu(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale


class MBConvBlock(nn.Module):
    """
    Mobile Inverted Bottleneck Conv block.

    expand (1x1) -> depthwise (kxk) -> SE -> project (1x1) + residual
    Residual is applied only when stride == 1 and in_channels == out_channels.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int,
                 stride: int, expand_ratio: int):
        super().__init__()
        mid_ch = in_ch * expand_ratio
        self.use_residual = (stride == 1 and in_ch == out_ch)

        layers: list[nn.Module] = []

        # Expansion phase (skip if ratio == 1)
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_ch, mid_ch, 1, bias=False),
                nn.BatchNorm2d(mid_ch),
                nn.SiLU(inplace=True),
            ])

        # Depthwise convolution
        pad = (kernel_size - 1) // 2
        layers.extend([
            nn.Conv2d(mid_ch, mid_ch, kernel_size, stride=stride,
                      padding=pad, groups=mid_ch, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.SiLU(inplace=True),
        ])

        # Squeeze-and-Excitation
        layers.append(SqueezeExcitation(mid_ch))

        # Projection phase
        layers.extend([
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ])

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_residual:
            out = out + x
        return out


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

# (expand_ratio, out_channels, num_blocks, stride, kernel_size)
B0_CONFIG = [
    (1,  16,  1, 1, 3),
    (6,  24,  2, 2, 3),
    (6,  40,  2, 2, 5),
    (6,  80,  3, 2, 3),
    (6, 112,  3, 1, 5),
    (6, 192,  4, 2, 5),
    (6, 320,  1, 1, 3),
]


class EfficientNet(nn.Module):
    """
    EfficientNet-B0 adapted for CIFAR-10 (32x32).

    The stem uses stride 1 (instead of stride 2) to preserve spatial resolution
    on small inputs. Stage strides are kept as-is from the original B0 config.
    """

    def __init__(self, num_classes: int = 10, config: list = B0_CONFIG):
        super().__init__()

        # Stem: 3x3 conv, stride 1 for CIFAR-sized inputs
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
        )

        # MBConv stages
        stages: list[nn.Module] = []
        in_ch = 32
        for expand, out_ch, n_blocks, stride, ksize in config:
            for i in range(n_blocks):
                s = stride if i == 0 else 1
                stages.append(MBConvBlock(in_ch, out_ch, ksize, s, expand))
                in_ch = out_ch
        self.stages = nn.Sequential(*stages)

        # Head
        head_ch = 1280
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, head_ch, 1, bias=False),
            nn.BatchNorm2d(head_ch),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(head_ch, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, 0, math.sqrt(2.0 / fan_out))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    device = get_device()
    print(f"Using device: {device}")

    # -- Data ------------------------------------------------------------------
    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    trainset = torchvision.datasets.CIFAR10(root="./data", train=True,
                                            download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root="./data", train=False,
                                           download=True, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                               shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=256,
                                              shuffle=False, num_workers=2)

    # -- Model -----------------------------------------------------------------
    model = EfficientNet(num_classes=10).to(device)
    if device.type == "cuda":
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=(device.type == "cuda"))

    num_epochs = 100
    for epoch in range(1, num_epochs + 1):
        # -- Train -------------------------------------------------------------
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, labels)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_loss = running_loss / total
        train_acc = 100.0 * correct / total

        # -- Evaluate ----------------------------------------------------------
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)
        test_acc = 100.0 * correct / total

        print(f"Epoch [{epoch:>3}/{num_epochs}]  "
              f"Loss: {train_loss:.4f}  "
              f"Train Acc: {train_acc:.2f}%  "
              f"Test Acc: {test_acc:.2f}%  "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")

    print("Training complete.")


if __name__ == "__main__":
    main()
