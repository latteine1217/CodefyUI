"""
BERT-Base Encoder for Masked Language Modeling (MLM)
====================================================

Architecture based on "BERT: Pre-training of Deep Bidirectional Transformers
for Language Understanding" (Devlin et al., 2018).

BERT is a bidirectional Transformer encoder pre-trained with two objectives:
Masked Language Modeling (MLM) and Next Sentence Prediction (NSP). This
implementation focuses on MLM, where 15% of input tokens are randomly masked
and the model learns to predict the original tokens from their context.

Unlike GPT (unidirectional), BERT uses full bidirectional attention, allowing
each token to attend to all other tokens in the sequence. This makes BERT
particularly effective for understanding tasks like classification, NER, and
question answering.

Key components:
  - BERTEmbedding: token embedding + segment embedding + position embedding + LayerNorm
  - TransformerEncoder: 12 layers, d_model=768, 12 heads, ff=3072
  - MLM Head: LayerNorm -> Linear -> GELU -> Linear(vocab_size) for token prediction

Config (BERT-Base):
  - vocab_size=30522, max_seq_len=512, d_model=768
  - 12 layers, 12 heads, intermediate_size=3072
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class BERTEmbedding(nn.Module):
    """BERT input embedding: token + segment + position, followed by LayerNorm."""

    def __init__(self, vocab_size: int = 30522, d_model: int = 768,
                 max_seq_len: int = 512, n_segments: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.segment_emb = nn.Embedding(n_segments, d_model)
        self.position_emb = nn.Embedding(max_seq_len, d_model)
        self.norm = nn.LayerNorm(d_model, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids: torch.Tensor,
                segment_ids: torch.Tensor) -> torch.Tensor:
        B, T = token_ids.shape
        positions = torch.arange(T, device=token_ids.device).unsqueeze(0)
        x = (self.token_emb(token_ids)
             + self.segment_emb(segment_ids)
             + self.position_emb(positions))
        return self.dropout(self.norm(x))


class TransformerBlock(nn.Module):
    """Post-norm Transformer encoder block (original BERT style)."""

    def __init__(self, d_model: int = 768, n_heads: int = 12,
                 ff_dim: int = 3072, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-12)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model, eps=1e-12)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Post-norm: residual -> add -> norm
        attn_out = self.attn(x, x, x, key_padding_mask=attn_mask,
                             need_weights=False)[0]
        x = self.norm1(x + self.drop(attn_out))
        x = self.norm2(x + self.ffn(x))
        return x


class MLMHead(nn.Module):
    """Masked Language Modeling prediction head."""

    def __init__(self, d_model: int = 768, vocab_size: int = 30522):
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(d_model, eps=1e-12)
        self.decoder = nn.Linear(d_model, vocab_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.norm(self.act(self.dense(hidden_states)))
        return self.decoder(x)


class BERT(nn.Module):
    """BERT encoder model with MLM head."""

    def __init__(self, vocab_size: int = 30522, d_model: int = 768,
                 n_layers: int = 12, n_heads: int = 12, ff_dim: int = 3072,
                 max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.embedding = BERTEmbedding(vocab_size, d_model, max_seq_len,
                                       dropout=dropout)
        self.encoder = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ff_dim, dropout)
            for _ in range(n_layers)
        ])
        self.mlm_head = MLMHead(d_model, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
            if module.padding_idx is not None:
                nn.init.zeros_(module.weight[module.padding_idx])
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, token_ids: torch.Tensor, segment_ids: torch.Tensor,
                padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.embedding(token_ids, segment_ids)
        for layer in self.encoder:
            x = layer(x, attn_mask=padding_mask)
        return self.mlm_head(x)   # (B, T, vocab_size)


def mask_tokens(token_ids: torch.Tensor, vocab_size: int,
                mask_token_id: int = 103, mask_prob: float = 0.15):
    """
    Apply BERT-style masking: 15% of tokens are selected.
    Of those: 80% replaced with [MASK], 10% random, 10% unchanged.
    Returns masked inputs and labels (-100 for unmasked positions).
    """
    labels = token_ids.clone()
    probability_matrix = torch.full(token_ids.shape, mask_prob)

    # Do not mask special tokens (0=PAD, 101=CLS, 102=SEP)
    special_mask = (token_ids == 0) | (token_ids == 101) | (token_ids == 102)
    probability_matrix.masked_fill_(special_mask, 0.0)

    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # only compute loss on masked tokens

    # 80% -> [MASK]
    replace_mask = torch.bernoulli(
        torch.full(token_ids.shape, 0.8)).bool() & masked_indices
    token_ids[replace_mask] = mask_token_id

    # 10% -> random token
    random_mask = (torch.bernoulli(
        torch.full(token_ids.shape, 0.5)).bool()
        & masked_indices & ~replace_mask)
    random_tokens = torch.randint(0, vocab_size, token_ids.shape)
    token_ids[random_mask] = random_tokens[random_mask]

    # remaining 10% -> unchanged (already handled)
    return token_ids, labels


class SyntheticMLMDataset(Dataset):
    """Synthetic dataset for MLM demonstration."""

    def __init__(self, vocab_size: int, seq_len: int, num_samples: int):
        self.vocab_size = vocab_size
        # Tokens in range [1000, vocab_size) to avoid special token IDs
        self.data = torch.randint(1000, vocab_size, (num_samples, seq_len))
        # Add CLS (101) at start and SEP (102) at end
        self.data[:, 0] = 101
        self.data[:, -1] = 102

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx].clone()
        segment_ids = torch.zeros_like(tokens)
        return tokens, segment_ids


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


if __name__ == "__main__":
    device = get_device()
    print(f"Using device: {device}")

    # Smaller config for demo (full BERT-Base is 110M params)
    vocab_size = 30522
    model = BERT(
        vocab_size=vocab_size, d_model=384, n_layers=6, n_heads=6,
        ff_dim=1536, max_seq_len=128, dropout=0.1,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"BERT model parameters: {param_count:,}")

    dataset = SyntheticMLMDataset(vocab_size, seq_len=128, num_samples=5000)
    loader = DataLoader(dataset, batch_size=32, shuffle=True,
                        num_workers=0, pin_memory=True)

    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01,
                            betas=(0.9, 0.999), eps=1e-8)
    scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=100)

    num_epochs = 3
    for epoch in range(num_epochs):
        model.train()
        total_loss, num_batches = 0.0, 0

        for i, (token_ids, segment_ids) in enumerate(loader):
            # Apply masking on each batch
            masked_ids, labels = mask_tokens(
                token_ids.clone(), vocab_size, mask_token_id=103)
            masked_ids = masked_ids.to(device)
            segment_ids = segment_ids.to(device)
            labels = labels.to(device)

            logits = model(masked_ids, segment_ids)
            loss = F.cross_entropy(
                logits.view(-1, vocab_size), labels.view(-1), ignore_index=-100)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            num_batches += 1

            if (i + 1) % 50 == 0:
                print(f"  Epoch {epoch+1} | Step {i+1}/{len(loader)} | "
                      f"Loss: {loss.item():.4f}")

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs} | Avg MLM Loss: {avg_loss:.4f}")

    print("Training complete.")
