"""Example encoding, batching and the seeded, arm-independent data order."""

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from ..data.formats import prompt_text, target_text
from ..tokenizers import Tokenizer

# Tokenizer arm -> (tokenizer file, number mode for the prompt, number mode for the target).
# The decoupled arms (H1c) share T-bpe-num's vocabulary and change only how numbers are split.
TOK_ARMS = {
    "char": ("char", None, None),
    "digit": ("digit", None, None),
    "bpe_num": ("bpe_num", None, None),
    "dec_db": ("bpe_num", "digit", "merged"),   # digit-in / bpe-out
    "dec_bd": ("bpe_num", "merged", "digit"),   # bpe-in / digit-out
}


def load_tokenizer(tok_arm: str, data_dir: str | Path = "data") -> Tokenizer:
    data_dir = Path(data_dir)
    if not data_dir.is_absolute():
        data_dir = Path(__file__).resolve().parents[3] / data_dir
    return Tokenizer.load(data_dir / "tokenizers" / f"{TOK_ARMS[tok_arm][0]}.json")


def encode_example(tok: Tokenizer, tok_arm: str, item: dict, order: str) -> tuple[list[int], int]:
    """Prompt ids + target ids, and the prompt length."""
    _, numbers_prompt, numbers_target = TOK_ARMS[tok_arm]
    p = tok.encode(prompt_text(item["question"]), numbers=numbers_prompt)
    t = tok.encode(target_text(item["hints"], item["answer"], order), numbers=numbers_target)
    return p + t, len(p)


def collate(examples: list[tuple[list[int], int]], pad_id: int) -> dict[str, torch.Tensor]:
    """Right-padded next-token batch. mask[b, t] = 1 where position t predicts a target token."""
    T = max(len(ids) for ids, _ in examples) - 1
    inp = torch.full((len(examples), T), pad_id, dtype=torch.long)
    tgt = torch.full((len(examples), T), pad_id, dtype=torch.long)
    mask = torch.zeros((len(examples), T), dtype=torch.float32)
    for b, (ids, n_prompt) in enumerate(examples):
        n = len(ids) - 1
        inp[b, :n] = torch.tensor(ids[:-1])
        tgt[b, :n] = torch.tensor(ids[1:])
        mask[b, max(n_prompt - 1, 0):n] = 1.0
    return {"input": inp, "target": tgt, "mask": mask}


@lru_cache(maxsize=8)
def _epoch_permutation(seed: int, n: int, epoch: int) -> np.ndarray:
    return np.random.default_rng([seed, epoch]).permutation(n)


def batch_indices(seed: int, n: int, step: int, batch_size: int) -> list[int]:
    """Indices for `step`: consecutive slices of per-epoch permutations seeded by (seed, epoch).
    It depends only on (seed, n, step, batch size), so every arm sees the same order."""
    out = []
    for p in range(step * batch_size, (step + 1) * batch_size):
        out.append(int(_epoch_permutation(seed, n, p // n)[p % n]))
    return out
