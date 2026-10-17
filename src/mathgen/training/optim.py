"""Loss, learning-rate schedule and parameter groups (RESEARCH_DESIGN.md §2, §4.3)."""

import math

import torch
import torch.nn.functional as F


def sequence_loss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Cross-entropy on completion tokens, averaged per sequence, then over the batch."""
    B, T, V = logits.shape
    ce = F.cross_entropy(logits.float().reshape(-1, V), target.reshape(-1), reduction="none").view(B, T)
    per_seq = (ce * mask).sum(1) / mask.sum(1).clamp_min(1.0)
    return per_seq.mean()


def lr_at(step: int, cfg) -> float:
    """Linear warmup over `warmup_frac` of the steps, then cosine decay to `min_lr_frac` × peak."""
    warmup = max(1, round(cfg.warmup_frac * cfg.steps))
    if step < warmup:
        return cfg.lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, cfg.steps - warmup)
    floor = cfg.min_lr_frac * cfg.lr
    return floor + 0.5 * (cfg.lr - floor) * (1 + math.cos(math.pi * progress))


def param_groups(model: torch.nn.Module, weight_decay: float) -> tuple[dict, dict]:
    """Weight decay on ≥2-D weights only: none on norm gains, biases or embeddings."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (decay if p.dim() >= 2 and "embed" not in name else no_decay).append(p)
    return {"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}
