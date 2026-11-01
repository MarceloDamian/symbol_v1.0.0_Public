"""
A small decoder-only transformer over APL glyph tokens.

Standard GPT block layout (pre-norm, causal self-attention, GELU MLP) with
learned positions and tied input/output embeddings. Deliberately small: the
vocabulary is ~170 glyphs and sequences are < 128 tokens.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int
    max_len: int = 128
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    dropout: float = 0.1

    def to_dict(self) -> dict:
        return asdict(self)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor, cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """With `cache` (past keys, values) the new queries attend to all of it.

        Returns (output, (keys, values)) so decoding can reuse past positions.
        """
        b, t, c = x.shape
        q, k, v = self.qkv(x).split(c, dim=2)
        shape = (b, t, self.n_heads, c // self.n_heads)
        q, k, v = (z.view(shape).transpose(1, 2) for z in (q, k, v))
        if cache is not None:
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)
        # Causal masking is only needed when queries and keys cover the same span;
        # a single decoding query may see every cached position.
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=(cache is None and t > 1),
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.proj(y.transpose(1, 2).contiguous().view(b, t, c)), (k, v)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor, cache=None):
        attn, cache = self.attn(self.ln1(x), cache)
        x = x + attn
        return x + self.mlp(self.ln2(x)), cache


class APLTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok.weight  # weight tying
        self.apply(self._init)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: (batch, time) -> logits (batch, time, vocab)."""
        return self._run(ids)[0]

    def _run(self, ids: torch.Tensor, caches: Optional[list] = None, start: int = 0):
        t = ids.size(1)
        if start + t > self.cfg.max_len:
            raise ValueError("sequence length %d exceeds max_len %d" % (start + t, self.cfg.max_len))
        pos = torch.arange(start, start + t, device=ids.device)
        x = self.drop(self.tok(ids) + self.pos(pos))
        new_caches = []
        for i, block in enumerate(self.blocks):
            x, cache = block(x, caches[i] if caches is not None else None)
            new_caches.append(cache)
        return self.head(self.ln_f(x)), new_caches

    @torch.no_grad()
    def generate(self, prompts: torch.Tensor, stop_id: int, max_new: int = 64,
                 temperature: float = 0.0) -> torch.Tensor:
        """Decode a batch of equal-length prompts: greedy at temperature 0, sampled above it.

        Returns only the generated continuation, (batch, <= max_new). Rows that
        hit `stop_id` keep emitting it so the batch stays rectangular.
        """
        self.eval()
        out = prompts
        done = torch.zeros(prompts.size(0), dtype=torch.bool, device=prompts.device)
        logits, caches = self._run(prompts)
        for _ in range(max_new):
            if out.size(1) >= self.cfg.max_len:
                break
            if temperature > 0:
                probs = torch.softmax(logits[:, -1].float() / temperature, dim=-1)
                nxt = torch.multinomial(probs, 1)[:, 0]
            else:
                nxt = logits[:, -1].argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, stop_id), nxt)
            out = torch.cat([out, nxt[:, None]], dim=1)
            done |= nxt == stop_id
            if bool(done.all()) or out.size(1) >= self.cfg.max_len:
                break
            # Only the new token is fed; past keys/values come from the cache.
            logits, caches = self._run(nxt[:, None], caches, start=out.size(1) - 1)
        return out[:, prompts.size(1) :]
