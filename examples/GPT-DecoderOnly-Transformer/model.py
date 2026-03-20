"""
GPT-2 Small Style Decoder-Only Language Model
==============================================

Architecture based on "Language Models are Unsupervised Multitask Learners"
(Radford et al., 2019) and refined with modern practices.

GPT is a decoder-only Transformer that uses causal (left-to-right) self-
attention to model the probability distribution over the next token given all
preceding tokens. Unlike encoder models (BERT), GPT never attends to future
positions, making it suitable for autoregressive text generation.

Key components:
  - Token Embedding + Learnable Positional Embedding
  - N stacked GPTBlocks: Pre-LayerNorm -> CausalSelfAttention -> Pre-LayerNorm -> MLP(GELU)
  - Causal mask ensures tokens only attend to earlier positions
  - Weight tying between token embedding and output projection (lm_head)

Config (GPT-2 Small):
  - vocab_size=50257, block_size=1024, d_model=768
  - 12 layers, 12 heads, mlp_dim=3072

Includes a simple generate() method with temperature-based sampling.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    d_model: int = 768
    dropout: float = 0.1
    bias: bool = False


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with causal mask."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.d_model % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.d_model // config.n_head
        self.d_model = config.d_model

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)

        # Causal mask: prevent attending to future tokens
        self.register_buffer("causal_mask", torch.tril(
            torch.ones(config.block_size, config.block_size)
        ).view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.head_dim)
        q, k, v = qkv.unbind(dim=2)                       # each (B, T, n_head, head_dim)
        q = q.transpose(1, 2)                              # (B, n_head, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale          # (B, n_head, T, T)
        attn = attn.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.resid_drop(self.out_proj(out))


class MLP(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, 4 * config.d_model, bias=config.bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4 * config.d_model, config.d_model, bias=config.bias)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.act(self.fc1(x))))


class GPTBlock(nn.Module):
    """Pre-norm Transformer decoder block."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """GPT-2 style decoder-only language model."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.block_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([GPTBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying between embedding and output projection
        self.token_emb.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor,
                targets: torch.Tensor | None = None):
        B, T = idx.shape
        assert T <= self.config.block_size, \
            f"Sequence length {T} exceeds block_size {self.config.block_size}"

        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.drop(self.token_emb(idx) + self.pos_emb(pos))

        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                           # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: int | None = None):
        """Autoregressive generation with temperature and optional top-k."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx


class SyntheticTextDataset(Dataset):
    """Generates random token sequences for demonstration."""

    def __init__(self, vocab_size: int, seq_len: int, num_samples: int):
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len + 1))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        return seq[:-1], seq[1:]  # input, target (shifted by 1)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


if __name__ == "__main__":
    device = get_device()
    print(f"Using device: {device}")

    # Use a smaller config for demo (full GPT-2 Small requires significant compute)
    config = GPTConfig(
        vocab_size=50257, block_size=128, n_layer=6, n_head=6,
        d_model=384, dropout=0.1, bias=False,
    )

    model = GPT(config).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"GPT model parameters: {param_count:,}")

    dataset = SyntheticTextDataset(config.vocab_size, config.block_size, 5000)
    loader = DataLoader(dataset, batch_size=16, shuffle=True,
                        num_workers=0, pin_memory=True)

    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1,
                            betas=(0.9, 0.95))

    num_epochs = 3
    for epoch in range(num_epochs):
        model.train()
        total_loss, num_batches = 0.0, 0
        for i, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            _, loss = model(inputs, targets)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if (i + 1) % 50 == 0:
                print(f"  Epoch {epoch+1} | Step {i+1}/{len(loader)} | "
                      f"Loss: {loss.item():.4f}")

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs} | Avg Loss: {avg_loss:.4f}")

    # Demo generation
    print("\n--- Generation Demo ---")
    prompt = torch.zeros(1, 1, dtype=torch.long, device=device)  # BOS token
    generated = model.generate(prompt, max_new_tokens=50,
                               temperature=0.8, top_k=40)
    print(f"Generated token IDs: {generated[0].tolist()[:20]}...")
    print("Training complete.")
