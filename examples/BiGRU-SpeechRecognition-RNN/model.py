"""
Bidirectional GRU for Speech Recognition (DeepSpeech-style)
============================================================
Architecture:
    2 stacked Bidirectional GRU layers (input=80 mel-frequency bins,
    hidden=256 per direction, 512 total) followed by a fully connected
    classifier that produces per-frame character logits.

    Input:  (batch, time_frames=100, mel_bins=80)
    Output: (batch, time_frames, num_classes)  -- per-frame character logits

    The bidirectional structure lets each frame attend to both past and future
    acoustic context, which is critical for speech where coarticulation means
    a phoneme's realization depends on its neighbors.

Real-world use cases:
    - Automatic speech recognition (ASR) / speech-to-text
    - Keyword spotting and voice command detection
    - Speaker diarization front-end features
    - Audio event classification

This demo generates synthetic mel-spectrogram data, trains with
CrossEntropyLoss as a CTC proxy (frame-level classification), and reports
per-frame accuracy.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class BiGRUSpeechModel(nn.Module):
    def __init__(
        self,
        mel_bins: int = 80,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_classes: int = 29,  # 26 letters + space + apostrophe + blank
        dropout: float = 0.15,
    ):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(mel_bins)
        self.bigru = nn.GRU(
            input_size=mel_bins,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        gru_out_size = hidden_size * 2  # bidirectional
        self.classifier = nn.Sequential(
            nn.Linear(gru_out_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, mel_bins)
        x = self.batch_norm(x.transpose(1, 2)).transpose(1, 2)  # normalize mel axis
        gru_out, _ = self.bigru(x)              # (batch, time, hidden*2)
        logits = self.classifier(gru_out)       # (batch, time, num_classes)
        return logits


# ---------------------------------------------------------------------------
# Dataset -- synthetic mel spectrograms
# ---------------------------------------------------------------------------
class SyntheticSpeechDataset(Dataset):
    """
    Generates random mel-spectrogram-like tensors and integer class labels
    per frame.  Each class has a distinct spectral pattern so the model can
    learn a non-trivial mapping.
    """

    def __init__(self, num_samples: int, num_frames: int, mel_bins: int,
                 num_classes: int, seed: int = 42):
        super().__init__()
        rng = torch.Generator().manual_seed(seed)

        # Class-specific spectral prototypes
        prototypes = torch.randn(num_classes, mel_bins, generator=rng)

        self.labels = torch.randint(0, num_classes, (num_samples, num_frames), generator=rng)
        self.features = torch.zeros(num_samples, num_frames, mel_bins)

        for i in range(num_samples):
            for t in range(num_frames):
                cls = self.labels[i, t].item()
                noise = 0.4 * torch.randn(mel_bins, generator=rng)
                self.features[i, t] = prototypes[cls] + noise

    def __len__(self) -> int:
        return self.features.size(0)

    def __getitem__(self, idx: int):
        return self.features[idx], self.labels[idx]


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for feats, labels in loader:
        feats, labels = feats.to(device), labels.to(device)
        logits = model(feats)                        # (B, T, C)
        loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * feats.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.numel()

    return total_loss / len(loader.dataset), correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for feats, labels in loader:
        feats, labels = feats.to(device), labels.to(device)
        logits = model(feats)
        loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        total_loss += loss.item() * feats.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.numel()
    return total_loss / len(loader.dataset), correct / total


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    NUM_FRAMES = 100
    MEL_BINS = 80
    NUM_CLASSES = 29
    EPOCHS = 25
    BATCH_SIZE = 32
    LR = 3e-4

    device = select_device()
    print(f"Using device: {device}")

    train_ds = SyntheticSpeechDataset(800, NUM_FRAMES, MEL_BINS, NUM_CLASSES, seed=42)
    val_ds = SyntheticSpeechDataset(200, NUM_FRAMES, MEL_BINS, NUM_CLASSES, seed=99)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = BiGRUSpeechModel(
        mel_bins=MEL_BINS, num_classes=NUM_CLASSES,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    print("-" * 60)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/{EPOCHS}  "
                f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
                f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}"
            )

    print("-" * 60)
    print("Training complete.")
