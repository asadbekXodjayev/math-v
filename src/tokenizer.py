"""
Custom BPE tokenizer for MathGen.
Vocab 16K — covers LaTeX, JSON structure, math notation, multilingual (EN/RU/UZ).
"""

import json
import os
import re
from collections import Counter
from pathlib import Path

SPECIAL_TOKENS: dict[str, int] = {
    "<EOS>":    0,
    "<PAD>":    1,
    "<UNK>":    2,
    # Task tokens
    "[GEN]":    3,   # generate a problem
    "[GRADE]":  4,   # grade a student answer
    "[ASSESS]": 5,   # assess skill level
    # Structural
    "<SEP>":    6,   # separates prompt from completion
    "<JSON>":   7,   # marks JSON output start
}

MATH_TOKENS = [
    # LaTeX
    "\\frac", "\\sqrt", "\\sum", "\\int", "\\lim", "\\infty", "\\partial",
    "\\alpha", "\\beta", "\\gamma", "\\delta", "\\theta", "\\lambda",
    "\\sigma", "\\pi", "\\mu", "\\epsilon", "\\nabla", "\\Delta",
    "\\mathbf", "\\vec", "\\hat", "\\cdot", "\\times", "\\otimes", "\\oplus",
    "\\leq", "\\geq", "\\neq", "\\approx", "\\equiv", "\\in", "\\subset",
    "\\forall", "\\exists", "\\rightarrow", "\\Rightarrow", "\\iff",
    "\\begin", "\\end", "\\matrix", "\\pmatrix", "\\bmatrix",
    # JSON structure
    '"questionText":', '"hints":', '"answerKey":', '"parameters":',
    '"correct":', '"score":', '"explanation":', '"hint":',
    '"skill_level":', '"next_difficulty":', '"message":', '"weaknesses":',
    # Math domain words
    "matrix", "vector", "eigenvalue", "eigenvector", "determinant",
    "derivative", "integral", "gradient", "divergence", "curl",
    "linear", "algebra", "arithmetic", "calculus", "function",
    "equation", "inequality", "polynomial", "coefficient", "variable",
    "magnitude", "dot product", "cross product", "projection", "span",
    "basis", "orthogonal", "perpendicular", "transpose", "inverse",
    "differentiate", "integrate", "evaluate", "simplify", "solve",
    # Difficulty / task tokens
    "difficulty:easy", "difficulty:medium", "difficulty:hard",
    "topic:basic-math", "topic:linear-algebra",
    "correct:true", "correct:false",
    "skill_level:", "next_difficulty:",
]


class BPETokenizer:
    def __init__(self):
        self.vocab:     dict[str, int]         = {}
        self.inv_vocab: dict[int, str]         = {}
        self.merges:    list[tuple[str, str]]  = []
        self._encode_cache: dict[str, list[int]] = {}
        self._matchers: list[str] | None = None

    def _get_pairs(self, freq: dict[tuple, int]) -> Counter:
        pairs: Counter = Counter()
        for word, cnt in freq.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i+1])] += cnt
        return pairs

    def _merge_vocab(self, freq: dict[tuple, int], pair: tuple[str, str]) -> dict[tuple, int]:
        merged = "".join(pair)
        out = {}
        for word, cnt in freq.items():
            w: list[str] = []
            i = 0
            while i < len(word):
                if i < len(word)-1 and (word[i], word[i+1]) == pair:
                    w.append(merged); i += 2
                else:
                    w.append(word[i]); i += 1
            out[tuple(w)] = cnt
        return out

    def train(self, texts: list[str], vocab_size: int = 16384) -> None:
        print(f"Training BPE on {len(texts)} texts -> vocab {vocab_size}...")

        # Seed vocab: specials + math tokens + ASCII
        self.vocab = dict(SPECIAL_TOKENS)
        for tok in MATH_TOKENS:
            if tok not in self.vocab:
                self.vocab[tok] = len(self.vocab)
        for i in range(256):
            c = chr(i)
            if c not in self.vocab:
                self.vocab[c] = len(self.vocab)

        # Seed every character that appears in the corpus so non-ASCII math symbols
        # (∫ → ∞ · × … and RU/UZ letters) always have a token and round-trip cleanly.
        for text in texts:
            for ch in text:
                if ch not in self.vocab:
                    self.vocab[ch] = len(self.vocab)

        # Build word frequency
        word_cnt: Counter = Counter()
        for text in texts:
            for w in text.split():
                word_cnt[w] += 1

        word_freq: dict[tuple, int] = {
            tuple(w): c for w, c in word_cnt.items() if c >= 2
        }

        # BPE merge loop
        while len(self.vocab) < vocab_size:
            pairs = self._get_pairs(word_freq)
            if not pairs:
                break
            # DIGIT-LEVEL TOKENIZATION: never merge a pair that would produce a token
            # containing a digit, so every number stays a sequence of single-digit
            # tokens (e.g. 127 -> '1','2','7'). Multi-digit BPE tokens make arithmetic
            # essentially unlearnable for the model — this is the #1 fix for accuracy.
            best = None
            for pair, _cnt in pairs.most_common():
                if not any(c.isdigit() for c in pair[0] + pair[1]):
                    best = pair
                    break
            if best is None:
                break
            self.merges.append(best)
            merged = "".join(best)
            # Guard: never overwrite a reserved special/math token (or any existing
            # entry) — doing so reassigns its ID and corrupts the vocab bijection.
            if merged not in self.vocab:
                self.vocab[merged] = len(self.vocab)
            word_freq = self._merge_vocab(word_freq, best)
            if len(self.vocab) % 1000 == 0:
                print(f"  vocab: {len(self.vocab)}")

        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self._matchers = None
        print(f"Tokenizer ready: {len(self.vocab)} tokens")

    def _encode_word(self, word: str) -> list[int]:
        if word in self._encode_cache:
            return self._encode_cache[word]
        chars = list(word)
        for pair in self.merges:
            merged = "".join(pair)
            i, new = 0, []
            while i < len(chars):
                if i < len(chars)-1 and (chars[i], chars[i+1]) == pair:
                    new.append(merged); i += 2
                else:
                    new.append(chars[i]); i += 1
            chars = new
        result = [self.vocab.get(c, SPECIAL_TOKENS["<UNK>"]) for c in chars]
        self._encode_cache[word] = result
        return result

    def encode(self, text: str) -> list[int]:
        # Single left-to-right pass: greedily match special/math tokens (longest first),
        # emit a space token for every space, BPE-encode the remaining word characters.
        # Guarantees decode(encode(x)) == x (spaces and ordering preserved).
        if not hasattr(self, "_matchers") or self._matchers is None:
            self._matchers = sorted(
                [t for t in list(SPECIAL_TOKENS.keys()) + MATH_TOKENS if t in self.vocab],
                key=len, reverse=True,
            )
        space_id = self.vocab.get(" ", SPECIAL_TOKENS["<UNK>"])
        ids: list[int] = []
        word = ""
        i, n = 0, len(text)

        def flush() -> None:
            nonlocal word
            if word:
                ids.extend(self._encode_word(word))
                word = ""

        while i < n:
            matched = None
            for tok in self._matchers:
                if text.startswith(tok, i):
                    matched = tok
                    break
            if matched is not None:
                flush()
                ids.append(self.vocab[matched])
                i += len(matched)
            elif text[i] == " ":
                flush()
                ids.append(space_id)
                i += 1
            else:
                word += text[i]
                i += 1
        flush()
        return ids

    def decode(self, ids: list[int]) -> str:
        return "".join(self.inv_vocab.get(i, "<UNK>") for i in ids)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"vocab": self.vocab, "merges": self.merges}, f, ensure_ascii=False)

    def load(self, path: str) -> None:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        self.vocab = d["vocab"]
        self.merges = [tuple(m) for m in d["merges"]]
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self._matchers = None
        self._encode_cache = {}

    def __len__(self) -> int:
        return len(self.vocab)

    @property
    def eos_id(self) -> int:
        return SPECIAL_TOKENS["<EOS>"]

    @property
    def pad_id(self) -> int:
        return SPECIAL_TOKENS["<PAD>"]
