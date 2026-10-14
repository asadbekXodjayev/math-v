"""
MathGen v2 — Multi-task educational transformer.
Tasks: GENERATE problems | GRADE answers | ASSESS student skill level
Architecture: decoder-only, 8 layers, d=512, RoPE + RMSNorm → ~36.7M params (vocab ~6k).
Trained offline on synthetic data; served CPU-only (see server.py).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from torch.utils.checkpoint import checkpoint


@dataclass
class MathGenConfig:
    vocab_size:  int   = 16384   # larger vocab for math/LaTeX coverage
    context_len: int   = 256     # reduced for 4GB VRAM (was 512)
    n_layers:    int   = 8       # reduced for 4GB VRAM (was 12)
    n_heads:     int   = 8
    d_model:     int   = 512
    d_ff:        int   = 2048    # 4 × d_model
    dropout:     float = 0.1
    bias:        bool  = False
    use_grad_ckpt: bool = True   # gradient checkpointing to save VRAM


# ── Building blocks ───────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.g


class RotaryEmbedding(nn.Module):
    """RoPE — relative position encoding baked into Q,K."""

    def __init__(self, head_dim: int, base: int = 10000):
        super().__init__()
        theta = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("theta", theta)

    def _rot(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=-1)
        return torch.cat([-b, a], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        T = q.size(2)
        freqs = torch.outer(torch.arange(T, device=q.device).float(), self.theta)
        cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)[None, None]
        sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)[None, None]
        return q * cos + self._rot(q) * sin, k * cos + self._rot(k) * sin


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: MathGenConfig):
        super().__init__()
        self.nh = cfg.n_heads
        self.hd = cfg.d_model // cfg.n_heads
        self.qkv  = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)
        self.rope = RotaryEmbedding(self.hd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.nh, self.hd).transpose(1, 2)
        k = k.view(B, T, self.nh, self.hd).transpose(1, 2)
        v = v.view(B, T, self.nh, self.hd).transpose(1, 2)
        q, k = self.rope(q, k)
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.drop.p if self.training else 0.0,
            is_causal=True,
        )
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))


class SwiGLU(nn.Module):
    def __init__(self, cfg: MathGenConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=cfg.bias)
        self.up   = nn.Linear(cfg.d_model, cfg.d_ff, bias=cfg.bias)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


class Block(nn.Module):
    def __init__(self, cfg: MathGenConfig):
        super().__init__()
        self.n1   = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.n2   = RMSNorm(cfg.d_model)
        self.ff   = SwiGLU(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.n1(x))
        x = x + self.ff(self.n2(x))
        return x


# ── Main model ─────────────────────────────────────────────────────────────────

class MathGen(nn.Module):
    """
    Multi-task educational transformer.

    Single model, three tasks distinguished by prompt prefix:
      [GEN]    → generate a math problem as JSON
      [GRADE]  → grade a student answer, return feedback JSON
      [ASSESS] → assess student skill level from answer history, return profile JSON
    """

    def __init__(self, cfg: MathGenConfig):
        super().__init__()
        self.cfg    = cfg
        self.embed  = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop   = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm   = RMSNorm(cfg.d_model)
        self.head   = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.embed.weight   # weight tying
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02 / math.sqrt(2 * self.cfg.n_layers))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.drop(self.embed(idx))
        for block in self.blocks:
            if self.cfg.use_grad_ckpt and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = self.norm(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size),
                targets.view(-1),
                ignore_index=-1,
            )
        return logits, loss

    @torch.inference_mode()
    def generate(
        self,
        prompt: torch.Tensor,
        max_new: int = 400,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.92,
    ) -> torch.Tensor:
        for _ in range(max_new):
            ctx = prompt[:, -self.cfg.context_len:]
            logits, _ = self(ctx)
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            # top-k
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # nucleus (top-p)
            probs = F.softmax(logits, dim=-1)
            sorted_p, sorted_idx = torch.sort(probs, descending=True)
            cum_p = sorted_p.cumsum(dim=-1)
            mask = cum_p - sorted_p > top_p
            sorted_p[mask] = 0.0
            sorted_p /= sorted_p.sum(dim=-1, keepdim=True)
            next_id = sorted_idx.gather(-1, torch.multinomial(sorted_p, 1))

            prompt = torch.cat([prompt, next_id], dim=1)
            if next_id.item() == 0:   # EOS
                break
        return prompt

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(vocab_size: int = 16384, device: str = "cuda") -> tuple["MathGen", MathGenConfig]:
    cfg = MathGenConfig(vocab_size=vocab_size)
    model = MathGen(cfg).to(device)
    n = model.num_params()
    print(f"MathGen v2: {n/1e6:.1f}M params | {cfg.n_layers}L x d{cfg.d_model} | RTX 3050 target")
    return model, cfg
