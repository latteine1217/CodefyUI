"""
ResNet-18 for CIFAR-10
======================
Residual Networks (He et al., "Deep Residual Learning for Image Recognition", 2015)
introduced skip (shortcut) connections that allow gradients to flow directly through
the network, enabling training of substantially deeper architectures.

The core insight is the residual mapping: instead of learning H(x) directly, the
network learns F(x) = H(x) - x, so the output becomes F(x) + x.  This makes it
easier for layers to learn identity mappings when needed, preventing degradation
in very deep networks.

BasicBlock structure:
    input -> Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> (+shortcut) -> ReLU

Architecture: Stem conv -> 4 layer groups [64,128,256,512] x 2 blocks -> AvgPool -> FC
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import torchvision
import torchvision.transforms as T


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class BasicBlock(nn.Module):
    """
    ResNet BasicBlock with skip connection.

    When the spatial size or channel count changes (stride > 1 or in != out),
    a 1x1 convolution shortcut projects the identity to match dimensions.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + identity)
        return out


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class ResNet(nn.Module):
    """
    ResNet-18 adapted for CIFAR-10 (32x32).

    Compared to the ImageNet variant, the stem uses a 3x3 conv with stride 1
    (instead of 7x7 stride 2 + maxpool) to preserve spatial resolution on the
    smaller 32x32 inputs.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.in_channels = 64

        # Stem: single 3x3 conv (no aggressive downsampling for 32x32 images)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Four residual layer groups, each with 2 BasicBlocks
        self.layer1 = self._make_layer(64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(512, num_blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

        self._init_weights()

    def _make_layer(self, out_channels: int, num_blocks: int, stride: int):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, stride=s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


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
    model = ResNet(num_classes=10).to(device)
    if device.type == "cuda":
        model = torch.compile(model)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1,
                                momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[60, 120, 160],
                                                     gamma=0.2)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=(device.type == "cuda"))

    num_epochs = 200
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
