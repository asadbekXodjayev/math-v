"""Batched greedy decoding with a KV cache, grouped by prompt length (so prompts need no padding)."""

from collections import defaultdict

from .. import compat  # noqa: F401  (must precede torch)

import torch

from ..model import KVCache


@torch.no_grad()
def greedy_decode(model, prompts: list[list[int]], eos_id: int, context_len: int, batch_size: int = 256,
                  device: torch.device | str | None = None) -> list[list[int]]:
    """Generated ids per prompt, up to and including EOS, or until the context is full.
    Identical to one-at-a-time full-recompute greedy decoding (tests/test_eval.py)."""
    device = device or next(model.parameters()).device
    results: list[list[int] | None] = [None] * len(prompts)
    by_len: dict[int, list[int]] = defaultdict(list)
    for i, p in enumerate(prompts):
        by_len[len(p)].append(i)
    for length in sorted(by_len):
        idxs = by_len[length]
        for s in range(0, len(idxs), batch_size):
            chunk = idxs[s:s + batch_size]
            cache = KVCache()
            logits = model(torch.tensor([prompts[i] for i in chunk], device=device), cache=cache)[:, -1]
            outs: list[list[int]] = [[] for _ in chunk]
            done = [False] * len(chunk)
            pos = length
            while True:
                nxt = logits.argmax(-1)
                for j, t in enumerate(nxt.tolist()):
                    if not done[j]:
                        outs[j].append(t)
                        done[j] = t == eos_id
                pos += 1
                if all(done) or pos >= context_len:
                    break
                logits = model(nxt[:, None], cache=cache)[:, -1]
            for j, i in enumerate(chunk):
                results[i] = outs[j]
    return results
