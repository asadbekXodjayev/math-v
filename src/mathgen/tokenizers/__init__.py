"""
Tokenizer arms (RESEARCH_DESIGN.md §4.1), all on one shared pre-tokenizer:

    special tokens | ASCII digit runs [0-9]+ | other non-space runs | whitespace runs

- **char:** no merges; one token per character.
- **digit (T-digit, REF):** BPE merges learned only inside non-digit chunks; every number is
  split into single ASCII digits.
- **bpe_num (T-bpe-num):** T-digit's merges plus BPE merges learned inside digit runs.
  T-digit's vocabulary is a prefix of this one, so non-number text tokenizes identically.

The decoupled arms (H1c) use the bpe_num vocabulary with `encode(text, numbers="digit")` on
one segment and `numbers="merged"` on the other.

Merge budgets (fixed in M5, logged in research/decisions.md): up to 2,000 text merges and
1,000 digit merges, each stopping early when the best pair occurs fewer than 2 times.
Ties are broken lexicographically, so fitting is deterministic.
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ARMS = ("char", "digit", "bpe_num")
SPECIALS = ("<PAD>", "<UNK>", "[SOLVE]", "<SEP>", "<JSON>", "<EOS>")
TEXT_MERGES, DIGIT_MERGES, MIN_FREQ = 2000, 1000, 2

_SPECIAL_RE = re.compile("(" + "|".join(re.escape(s) for s in SPECIALS) + ")")
_CHUNK_RE = re.compile(r"[0-9]+|[^0-9\s]+|\s+")


def pretokenize(text: str) -> list[str]:
    out: list[str] = []
    for part in _SPECIAL_RE.split(text):
        if part in SPECIALS:
            out.append(part)
        elif part:
            out.extend(_CHUNK_RE.findall(part))
    return out


def _is_digits(chunk: str) -> bool:
    return chunk.isascii() and chunk.isdigit()


def _learn_merges(chunk_counts: Counter, max_merges: int) -> list[tuple[str, str]]:
    """Byte-pair merges over chunks, with incremental pair counts; ties broken lexicographically."""
    words = [list(c) for c in chunk_counts]
    freqs = list(chunk_counts.values())
    pair_counts: Counter = Counter()
    where: dict[tuple[str, str], set[int]] = defaultdict(set)
    for i, w in enumerate(words):
        for pair in zip(w, w[1:]):
            pair_counts[pair] += freqs[i]
            where[pair].add(i)
    merges = []
    while len(merges) < max_merges and pair_counts:
        best = min(pair_counts, key=lambda p: (-pair_counts[p], p))
        if pair_counts[best] < MIN_FREQ:
            break
        merges.append(best)
        a, b = best
        for i in sorted(where.pop(best, ())):
            w = words[i]
            for pair in zip(w, w[1:]):          # remove this word's old pairs
                pair_counts[pair] -= freqs[i]
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]
            merged, j = [], 0
            while j < len(w):
                if j + 1 < len(w) and w[j] == a and w[j + 1] == b:
                    merged.append(a + b)
                    j += 2
                else:
                    merged.append(w[j])
                    j += 1
            words[i] = merged
            for pair in zip(merged, merged[1:]):  # add its new pairs
                pair_counts[pair] += freqs[i]
                where[pair].add(i)
        pair_counts.pop(best, None)
    return merges


def _apply_merges(chunk: str, ranks: dict[tuple[str, str], int]) -> list[str]:
    symbols = list(chunk)
    while len(symbols) > 1:
        ranked = [(ranks[p], i) for i, p in enumerate(zip(symbols, symbols[1:])) if p in ranks]
        if not ranked:
            break
        pair = (symbols[min(ranked)[1]], symbols[min(ranked)[1] + 1])
        merged, j = [], 0
        while j < len(symbols):
            if j + 1 < len(symbols) and (symbols[j], symbols[j + 1]) == pair:
                merged.append(pair[0] + pair[1])
                j += 2
            else:
                merged.append(symbols[j])
                j += 1
        symbols = merged
    return symbols


class Tokenizer:
    def __init__(self, arm: str, base_chars: list[str], text_merges: list[tuple[str, str]],
                 digit_merges: list[tuple[str, str]]):
        self.arm = arm
        self.specials = SPECIALS
        self.base_chars = base_chars
        self.text_merges, self.digit_merges = text_merges, digit_merges
        # Two different merges can produce the same string (('ab','c') and ('a','bc')): one token each.
        self.vocab = list(dict.fromkeys(list(SPECIALS) + base_chars + [a + b for a, b in text_merges]
                                        + [a + b for a, b in digit_merges]))
        self.token_to_id = {s: i for i, s in enumerate(self.vocab)}
        self.pad_id, self.unk_id = self.token_to_id["<PAD>"], self.token_to_id["<UNK>"]
        self.eos_id = self.token_to_id["<EOS>"]
        self._text_ranks = {p: i for i, p in enumerate(text_merges)}
        self._digit_ranks = {p: i for i, p in enumerate(digit_merges)}
        self._cache: dict[tuple[str, str], list[int]] = {}

    def _encode_chunk(self, chunk: str, numbers: str) -> list[int]:
        cached = self._cache.get((chunk, numbers))
        if cached is not None:
            return cached
        if chunk in SPECIALS:
            pieces = [chunk]
        elif _is_digits(chunk):
            pieces = _apply_merges(chunk, self._digit_ranks) if numbers == "merged" else list(chunk)
        else:
            pieces = _apply_merges(chunk, self._text_ranks)
        ids = [self.token_to_id.get(p, self.unk_id) for p in pieces]
        self._cache[(chunk, numbers)] = ids
        return ids

    def encode(self, text: str, numbers: str | None = None) -> list[int]:
        """`numbers`: 'digit' splits numbers into digits, 'merged' applies digit merges.
        The default is the arm's own behaviour."""
        numbers = numbers or ("merged" if self.digit_merges else "digit")
        return [i for chunk in pretokenize(text) for i in self._encode_chunk(chunk, numbers)]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.vocab[i] for i in ids if i != self.pad_id)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            "arm": self.arm, "base_chars": self.base_chars,
            "text_merges": self.text_merges, "digit_merges": self.digit_merges,
        }, ensure_ascii=False), encoding="utf-8", newline="\n")

    @classmethod
    def load(cls, path: str | Path) -> "Tokenizer":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(d["arm"], d["base_chars"], [tuple(m) for m in d["text_merges"]],
                   [tuple(m) for m in d["digit_merges"]])


def fit_tokenizer(arm: str, texts: list[str]) -> Tokenizer:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    chunks = Counter(c for t in texts for c in pretokenize(t) if c not in SPECIALS)
    base_chars = sorted({ch for c in chunks for ch in c} | set("0123456789"))
    if arm == "char":
        return Tokenizer(arm, base_chars, [], [])
    text_merges = _learn_merges(Counter({c: n for c, n in chunks.items() if not _is_digits(c)}), TEXT_MERGES)
    if arm == "digit":
        return Tokenizer(arm, base_chars, text_merges, [])
    digit_merges = _learn_merges(Counter({c: n for c, n in chunks.items() if _is_digits(c)}), DIGIT_MERGES)
    return Tokenizer(arm, base_chars, text_merges, digit_merges)
