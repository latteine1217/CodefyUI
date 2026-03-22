"""
Train CNN on MNIST — Complete Training Pipeline Example
========================================================
Corresponds to the graph.json that uses:
  - preset:Simple CNN Classifier  →  Conv2d(1→32) → ReLU → MaxPool → Conv2d(32→64) → ReLU → MaxPool
  - preset:Training Pipeline      →  Dataset(MNIST) → DataLoader → CrossEntropyLoss → Adam → TrainingLoop

This script is a standalone, runnable implementation of the exact same pipeline.
Run: python model.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ── Model: matches preset:Simple CNN Classifier ──────────────────

class SimpleCNN(nn.Module):
    """
    Mirrors the 'Simple CNN Classifier' preset:
      Conv2d(1→32, k=3, p=1) → ReLU → MaxPool(2)
      Conv2d(32→64, k=3, p=1) → ReLU → MaxPool(2)
      Flatten → FC(64*7*7 → 128) → ReLU → FC(128 → 10)
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        # Feature extractor (matches preset internal nodes)
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),   # conv1
            nn.ReLU(),                                               # act1
            nn.MaxPool2d(kernel_size=2, stride=2),                   # pool1
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # conv2
            nn.ReLU(),                                               # act2
            nn.MaxPool2d(kernel_size=2, stride=2),                   # pool2
        )
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


# ── Device selection ─────────────────────────────────────────────

def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Training (matches preset:Training Pipeline) ─────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


# ── Main ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = select_device()
    print(f"Device: {device}")

    # --- Data (matches Training Pipeline preset: Dataset → DataLoader) ---
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST("./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # --- Model ---
    model = SimpleCNN(num_classes=10).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # --- Training config (matches Training Pipeline preset params) ---
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 10

    # --- Training loop (matches TrainingLoop node execution) ---
    print(f"\n{'Epoch':>5} {'Loss':>10} {'Train Acc':>10} {'Test Acc':>10}")
    print("-" * 40)

    for epoch in range(1, epochs + 1):
        avg_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        train_acc = evaluate(model, train_loader, device)
        test_acc = evaluate(model, test_loader, device)
        print(f"{epoch:>5d} {avg_loss:>10.4f} {train_acc:>9.1%} {test_acc:>9.1%}")

    print(f"\nFinal Test Accuracy: {test_acc:.1%}")
