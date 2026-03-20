"""
LLaMA-Style Modern Decoder-Only Language Model
===============================================

Architecture based on "LLaMA: Open and Efficient Foundation Language Models"
(Touvron et al., 2023) incorporating modern innovations over the original GPT.

LLaMA represents the state of the art in open-source LLM architecture, combining
several improvements that have become standard in modern Transformer LLMs:

  - RMSNorm (Root Mean Square Layer Normalization): simpler and faster than
    LayerNorm, removes the mean-centering step.
  - RoPE (Rotary Positional Embedding): encodes position through rotation of
    query/key vectors, providing relative position information with better
    extrapolation to longer sequences than learned embeddings.
  - SwiGLU activation in FFN: gated linear unit with SiLU/Swish activation,
    outperforms standard GELU MLP at equivalent compute.
  - GQA (Grouped-Query Attention): a middle ground between multi-head attention
    (MHA) and multi-query attention (MQA). Multiple query heads share a single
    KV head group, reducing KV cache memory during inference while retaining
    most of MHA quality.
  - Pre-norm architecture with no bias terms.

Config (small demo version):
  - d_model=256, n_layers=4, n_heads=8, n_kv_heads=2
  - vocab_size=32000, max_seq_len=512
  - intermediate_size computed as 2/3 * 4 * d_model, rounded to multiple of 256
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass


@dataclass
class LLaMAConfig:
    vocab_size: int = 32000
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 8         # query heads
    n_kv_heads: int = 2      # key-value heads (GQA)
    max_seq_len: int = 512
    dropout: float = 0.0
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def intermediate_size(self) -> int:
        # SwiGLU convention: 2/3 of 4*d, rounded to multiple of 256
        raw = int(2 * (4 * self.d_model) / 3)
        return ((raw + 255) // 256) * 256


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).type_as(x) * self.weight


def precompute_rope_freqs(dim: int, max_seq_len: int,
                          theta: float = 10000.0,
                          device: torch.device | None = None):
    """Precompute the complex exponentials for Rotary Position Embedding."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rope(xq: torch.Tensor, xk: torch.Tensor,
               freqs_cis: torch.Tensor):
    """Apply rotary embeddings to query and key tensors."""
    # Reshape to complex: (B, T, H, D) -> (B, T, H, D/2) as complex
    xq_c = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_c = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # freqs_cis shape: (T, D/2) -> (1, T, 1, D/2)
    freqs = freqs_cis[None, :xq_c.shape[1], None, :]

    xq_out = torch.view_as_real(xq_c * freqs).flatten(-2)
    xk_out = torch.view_as_real(xk_c * freqs).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class GQAttention(nn.Module):
    """Grouped-Query Attention: multiple Q heads share fewer KV heads."""

    def __init__(self, config: LLaMAConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.n_rep = config.n_heads // config.n_kv_heads  # repetition factor

        self.wq = nn.Linear(config.d_model, config.n_heads * config.head_dim, bias=False)
        self.wk = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.wv = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * config.head_dim, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        # Apply RoPE to queries and keys
        q, k = apply_rope(q, k, freqs_cis)

        # Expand KV heads to match Q heads (GQA)
        if self.n_rep > 1:
            k = k.unsqueeze(3).expand(-1, -1, -1, self.n_rep, -1)
            k = k.reshape(B, T, self.n_heads, self.head_dim)
            v = v.unsqueeze(3).expand(-1, -1, -1, self.n_rep, -1)
            v = v.reshape(B, T, self.n_heads, self.head_dim)

        # Transpose for attention: (B, n_heads, T, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale

        if mask is not None:
            attn = attn + mask[:T, :T]

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)


class SwiGLU(nn.Module):
    """SwiGLU Feed-Forward Network: gate * swish(x) projection."""

    def __init__(self, config: LLaMAConfig):
        super().__init__()
        hidden = config.intermediate_size
        self.w_gate = nn.Linear(config.d_model, hidden, bias=False)
        self.w_up = nn.Linear(config.d_model, hidden, bias=False)
        self.w_down = nn.Linear(hidden, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class LLaMABlock(nn.Module):
    """LLaMA Transformer decoder block."""

    def __init__(self, config: LLaMAConfig):
        super().__init__()
        self.attention_norm = RMSNorm(config.d_model, config.norm_eps)
        self.attention = GQAttention(config)
        self.ffn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.ffn = SwiGLU(config)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x), freqs_cis, mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class LLaMA(nn.Module):
    """LLaMA-style decoder-only language model."""

    def __init__(self, config: LLaMAConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList(
            [LLaMABlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.d_model, config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying
        self.tok_emb.weight = self.lm_head.weight

        # Precompute RoPE frequencies
        self.register_buffer("freqs_cis", precompute_rope_freqs(
            config.head_dim, config.max_seq_len * 2, config.rope_theta))

        # Causal mask
        mask = torch.full((config.max_seq_len, config.max_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("causal_mask", mask)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, tokens: torch.Tensor,
                targets: torch.Tensor | None = None):
        B, T = tokens.shape
        x = self.tok_emb(tokens)
        freqs = self.freqs_cis[:T]

        for layer in self.layers:
            x = layer(x, freqs, self.causal_mask)

        x = self.norm(x)
        logits = self.lm_head(x)

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
            idx_crop = idx[:, -self.config.max_seq_len:]
            logits, _ = self(idx_crop)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_tok], dim=1)
        return idx


class SyntheticDataset(Dataset):
    """Random token dataset for demonstration."""

    def __init__(self, vocab_size: int, seq_len: int, num_samples: int):
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len + 1))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        return seq[:-1], seq[1:]


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


if __name__ == "__main__":
    device = get_device()
    print(f"Using device: {device}")

    config = LLaMAConfig(
        vocab_size=32000, d_model=256, n_layers=4,
        n_heads=8, n_kv_heads=2, max_seq_len=256, dropout=0.0,
    )

    model = LLaMA(config).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LLaMA model parameters: {param_count:,}")
    print(f"  d_model={config.d_model}, layers={config.n_layers}, "
          f"heads={config.n_heads}, kv_heads={config.n_kv_heads}")
    print(f"  intermediate_size={config.intermediate_size}, "
          f"head_dim={config.head_dim}")
    print(f"  GQA repetition factor: {config.n_heads // config.n_kv_heads}")

    dataset = SyntheticDataset(config.vocab_size, config.max_seq_len, 5000)
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
    prompt = torch.randint(0, config.vocab_size, (1, 4), device=device)
    print(f"Prompt tokens: {prompt[0].tolist()}")
    generated = model.generate(prompt, max_new_tokens=32,
                               temperature=0.8, top_k=50)
    print(f"Generated tokens: {generated[0].tolist()}")
    print("Training complete.")
