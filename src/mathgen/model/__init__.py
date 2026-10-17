"""
Decoder-only Transformer arms (RESEARCH_DESIGN.md §4.2).

REF is d=256, 6 layers, 4 heads, pre-norm RMSNorm, RoPE, SwiGLU (d_ff=688), tied embeddings:
about 4.7M non-embedding parameters. The variants change only the position scheme
(learned / none) or the FFN (GELU, d_ff=1032, matched in parameters).

Module and parameter names follow the legacy `src/model.py`, so the legacy 34M checkpoint
loads strictly (`Transformer.from_legacy_state_dict`) and reproduces its logits (plan test 11).
"""

import math
from dataclasses import dataclass

from .. import compat  # noqa: F401  (must precede torch)

import torch
import torch.nn as nn
import torch.nn.functional as F

ARCH_ARMS = ("ref", "learned", "none", "gelu")


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    context_len: int = 256
    n_layers: int = 6
    n_heads: int = 4
    d_model: int = 256
    d_ff: int = 688
    pos: str = "rope"          # rope | learned | none
    ffn: str = "swiglu"        # swiglu | gelu
    dropout: float = 0.0
    norm_eps: float = 1e-8     # legacy value


def arch_config(arm: str, vocab_size: int, context_len: int) -> ModelConfig:
    base = ModelConfig(vocab_size=vocab_size, context_len=context_len)
    variants = {
        "ref": {},
        "learned": {"pos": "learned"},
        "none": {"pos": "none"},
        "gelu": {"ffn": "gelu", "d_ff": 1032},
    }
    if arm not in variants:
        raise ValueError(f"unknown architecture arm {arm!r}")
    return ModelConfig(**{**base.__dict__, **variants[arm]})


class KVCache:
    """Per-layer keys/values for incremental decoding; `length` = positions already cached."""

    def __init__(self):
        self.layers: list[tuple[torch.Tensor, torch.Tensor] | None] = []
        self.length = 0


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()   # normalize in fp32 under bf16 autocast
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps) * self.g).type_as(x)


class RotaryEmbedding(nn.Module):
    """RoPE, rotate-half form, identical to legacy (persistent `theta` buffer)."""

    def __init__(self, head_dim: int, base: int = 10000):
        super().__init__()
        theta = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("theta", theta)

    @staticmethod
    def _rot(x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=-1)
        return torch.cat([-b, a], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, offset: int):
        T = q.size(2)
        freqs = torch.outer(torch.arange(offset, offset + T, device=q.device).float(), self.theta)
        cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)[None, None].to(q.dtype)
        sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)[None, None].to(q.dtype)
        return q * cos + self._rot(q) * sin, k * cos + self._rot(k) * sin


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.nh, self.hd = cfg.n_heads, cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout
        if cfg.pos == "rope":
            self.rope = RotaryEmbedding(self.hd)

    def forward(self, x: torch.Tensor, cache: KVCache | None, layer: int) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = (t.view(B, T, self.nh, self.hd).transpose(1, 2) for t in self.qkv(x).split(C, dim=2))
        offset = cache.length if cache is not None else 0
        if hasattr(self, "rope"):
            q, k = self.rope(q, k, offset)
        if cache is not None:
            if layer < len(cache.layers):
                pk, pv = cache.layers[layer]
                k, v = torch.cat([pk, k], dim=2), torch.cat([pv, v], dim=2)
                cache.layers[layer] = (k, v)
            else:
                cache.layers.append((k, v))
        if offset > 0 and T > 1:
            raise ValueError("cached decoding feeds one token at a time after the prefill")
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0,
                                           is_causal=offset == 0)
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class GELUFFN(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.gelu(self.up(x)))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.n2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ff = SwiGLU(cfg) if cfg.ffn == "swiglu" else GELUFFN(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, cache: KVCache | None, layer: int) -> torch.Tensor:
        x = x + self.drop(self.attn(self.n1(x), cache, layer))
        return x + self.drop(self.ff(self.n2(x)))


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.pos not in ("rope", "learned", "none") or cfg.ffn not in ("swiglu", "gelu"):
            raise ValueError(f"bad config {cfg}")
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        if cfg.pos == "learned":
            self.pos_embed = nn.Embedding(cfg.context_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.embed.weight   # weight tying
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:   # legacy initialization
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02 / math.sqrt(2 * self.cfg.n_layers))
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, idx: torch.Tensor, cache: KVCache | None = None) -> torch.Tensor:
        offset = cache.length if cache is not None else 0
        if offset + idx.size(1) > self.cfg.context_len:
            raise ValueError("sequence longer than the context length")
        x = self.embed(idx)
        if self.cfg.pos == "learned":
            x = x + self.pos_embed(torch.arange(offset, offset + idx.size(1), device=idx.device))
        x = self.drop(x)
        for i, block in enumerate(self.blocks):
            x = block(x, cache, i)
        if cache is not None:
            cache.length += idx.size(1)
        return self.head(self.norm(x))

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def non_embedding_params(self) -> int:
        """Excludes the (tied) token embedding and learned position embeddings."""
        emb = self.embed.weight.numel() + (self.pos_embed.weight.numel() if self.cfg.pos == "learned" else 0)
        return self.num_params() - emb

    @classmethod
    def from_legacy_state_dict(cls, state: dict) -> "Transformer":
        """Load a legacy `src/model.py` checkpoint (strict): infers the config from tensor shapes."""
        vocab, d = state["embed.weight"].shape
        n_layers = 1 + max(int(k.split(".")[1]) for k in state if k.startswith("blocks."))
        d_ff = state["blocks.0.ff.gate.weight"].shape[0]
        head_dim = 2 * state["blocks.0.attn.rope.theta"].shape[0]
        cfg = ModelConfig(vocab_size=vocab, context_len=256, n_layers=n_layers, n_heads=d // head_dim,
                          d_model=d, d_ff=d_ff, pos="rope", ffn="swiglu", dropout=0.0, norm_eps=1e-8)
        model = cls(cfg)
        model.load_state_dict(state, strict=True)
        return model
