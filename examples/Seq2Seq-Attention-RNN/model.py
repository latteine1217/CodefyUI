"""
Seq2Seq with Attention (RNN + Transformer Hybrid)
==================================================
Architecture:
    Encoder  -- 2-layer Bidirectional GRU (hidden=256 per direction, output=512)
    Attention -- nn.MultiheadAttention (embed_dim=512, 8 heads)
    Decoder  -- 2-layer LSTM (input=512+512 context, hidden=256)
    Output   -- Linear projection to target vocabulary logits

    At each decoder step the current decoder hidden state queries the full
    encoder output via multi-head attention.  The attended context vector is
    concatenated with the embedded target token and fed into the LSTM.

    Input:  source sequence of integer tokens  (batch, src_len)
    Output: target logits per step             (batch, tgt_len, vocab_size)

Real-world use cases:
    - Machine translation (e.g. English to French)
    - Text summarization (document to summary)
    - Dialogue response generation
    - Code generation from natural language descriptions

This demo generates synthetic integer sequences (reverse-copy task), trains
the model, and reports per-token accuracy.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(
            embed_dim, hidden_size, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=dropout,
        )

    def forward(self, src: torch.Tensor):
        # src: (batch, src_len)
        embedded = self.embedding(src)              # (batch, src_len, embed)
        outputs, hidden = self.gru(embedded)        # outputs: (B, S, H*2)
        # Merge bidirectional hidden: (num_layers*2, B, H) -> (num_layers, B, H*2)
        num_layers = hidden.size(0) // 2
        hidden = hidden.view(num_layers, 2, -1, hidden.size(-1))
        hidden = torch.cat([hidden[:, 0], hidden[:, 1]], dim=-1)  # (layers, B, H*2)
        return outputs, hidden


# ---------------------------------------------------------------------------
# Attention Decoder
# ---------------------------------------------------------------------------
class AttentionDecoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, enc_dim: int,
                 hidden_size: int, num_layers: int = 2, num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.attention = nn.MultiheadAttention(
            embed_dim=enc_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.lstm = nn.LSTM(
            input_size=embed_dim + enc_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc_out = nn.Linear(hidden_size, vocab_size)
        self.hidden_proj = nn.Linear(enc_dim, hidden_size)

    def _init_lstm_state(self, enc_hidden: torch.Tensor):
        """Project encoder GRU hidden (layers, B, enc_dim) to LSTM (h, c)."""
        h = torch.tanh(self.hidden_proj(enc_hidden))  # (layers, B, hidden)
        c = torch.zeros_like(h)
        return h.contiguous(), c.contiguous()

    def forward(self, tgt: torch.Tensor, enc_outputs: torch.Tensor,
                enc_hidden: torch.Tensor):
        # tgt: (batch, tgt_len)  enc_outputs: (batch, src_len, enc_dim)
        embedded = self.embedding(tgt)               # (B, T, embed)
        h, c = self._init_lstm_state(enc_hidden)

        outputs = []
        for t in range(embedded.size(1)):
            token_emb = embedded[:, t : t + 1, :]    # (B, 1, embed)

            # Query = last LSTM hidden state reshaped to (B, 1, hidden)
            query = h[-1].unsqueeze(1)                # (B, 1, hidden)
            # Project query to enc_dim for attention compatibility
            attn_out, _ = self.attention(
                query.expand(-1, -1, enc_outputs.size(-1)),
                enc_outputs, enc_outputs,
            )                                          # (B, 1, enc_dim)

            lstm_in = torch.cat([token_emb, attn_out], dim=-1)  # (B, 1, embed+enc)
            lstm_out, (h, c) = self.lstm(lstm_in, (h, c))       # (B, 1, hidden)
            outputs.append(lstm_out)

        outputs = torch.cat(outputs, dim=1)           # (B, T, hidden)
        return self.fc_out(outputs)                    # (B, T, vocab_size)


# ---------------------------------------------------------------------------
# Full Seq2Seq model
# ---------------------------------------------------------------------------
class Seq2SeqAttention(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128,
                 enc_hidden: int = 256, dec_hidden: int = 256,
                 num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        enc_dim = enc_hidden * 2  # bidirectional
        self.encoder = Encoder(vocab_size, embed_dim, enc_hidden, dropout=dropout)
        self.decoder = AttentionDecoder(
            vocab_size, embed_dim, enc_dim, dec_hidden,
            num_heads=num_heads, dropout=dropout,
        )

    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        enc_outputs, enc_hidden = self.encoder(src)
        logits = self.decoder(tgt, enc_outputs, enc_hidden)
        return logits


# ---------------------------------------------------------------------------
# Dataset -- synthetic reverse-copy task
# ---------------------------------------------------------------------------
class ReverseCopyDataset(Dataset):
    """
    Source: random integer sequences [BOS, t1, t2, ..., EOS]
    Target: the same tokens reversed [BOS, ..., t2, t1, EOS]
    Tokens 0=PAD, 1=BOS, 2=EOS, 3..vocab_size-1 = content tokens.
    """

    BOS, EOS = 1, 2

    def __init__(self, num_samples: int, seq_len: int, vocab_size: int, seed: int = 42):
        super().__init__()
        rng = torch.Generator().manual_seed(seed)
        content = torch.randint(3, vocab_size, (num_samples, seq_len), generator=rng)

        self.src = torch.cat([
            torch.full((num_samples, 1), self.BOS, dtype=torch.long),
            content,
            torch.full((num_samples, 1), self.EOS, dtype=torch.long),
        ], dim=1)

        self.tgt = torch.cat([
            torch.full((num_samples, 1), self.BOS, dtype=torch.long),
            content.flip(dims=[1]),
            torch.full((num_samples, 1), self.EOS, dtype=torch.long),
        ], dim=1)

    def __len__(self) -> int:
        return self.src.size(0)

    def __getitem__(self, idx: int):
        return self.src[idx], self.tgt[idx]


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        tgt_input = tgt[:, :-1]   # teacher forcing: feed all but last
        tgt_labels = tgt[:, 1:]   # predict all but first (BOS)

        logits = model(src, tgt_input)    # (B, T, V)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_labels.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * src.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == tgt_labels).sum().item()
        total += tgt_labels.numel()
    return total_loss / len(loader.dataset), correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        tgt_input = tgt[:, :-1]
        tgt_labels = tgt[:, 1:]
        logits = model(src, tgt_input)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_labels.reshape(-1))
        total_loss += loss.item() * src.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == tgt_labels).sum().item()
        total += tgt_labels.numel()
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
    VOCAB_SIZE = 50
    SEQ_LEN = 12
    EPOCHS = 40
    BATCH_SIZE = 64
    LR = 5e-4

    device = select_device()
    print(f"Using device: {device}")

    train_ds = ReverseCopyDataset(3000, SEQ_LEN, VOCAB_SIZE, seed=42)
    val_ds = ReverseCopyDataset(500, SEQ_LEN, VOCAB_SIZE, seed=99)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = Seq2SeqAttention(
        vocab_size=VOCAB_SIZE, embed_dim=128,
        enc_hidden=256, dec_hidden=256, num_heads=8,
    ).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    print("-" * 65)

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

    print("-" * 65)
    print("Training complete.")
