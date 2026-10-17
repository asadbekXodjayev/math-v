"""
Instance-level split assignment (RESEARCH_DESIGN.md §3.2–3.3).

Each instance key hashes to a bucket (`sha256(key)` prefix mod `modulus`) that fixes its pool
(train / val / test) for good. Block regions are removed from every pool and exist only in the
block split. M-set templates are not hashed: their whole space is in the train pool.
"""

import hashlib

# Pre-registered block regions: id -> (template, predicate on canonical operands).
BLOCKS = {
    "B1": ("arith_integer", lambda o: o["op"] == "*" and 20 <= o["a"] <= 30 and 20 <= o["b"] <= 30),
    "B2": ("algebra_linear", lambda o: o["x"] == 7),
    "B3": ("func_evaluate", lambda o: o["x"] == -3),
    "B4": ("mat_det", lambda o: min(o["a11"], o["a12"], o["a21"], o["a22"]) >= 7),
    "B5": ("vec_dot", lambda o: 10 <= o["u0"] <= 12 and 10 <= o["v0"] <= 12),
    "B6": ("arith_fractions", lambda o: o["d1"] == 7),
    "B7": ("deriv_polynomial", lambda o: o["n"] == 4 and o["m"] == 2),
}


def bucket(key: str, modulus: int) -> int:
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:16], 16) % modulus


def in_block(template: str, ops: dict) -> str | None:
    for block_id, (name, pred) in BLOCKS.items():
        if name == template and pred(ops):
            return block_id
    return None


def pool(template: str, key: str, ops: dict, config: dict, m_set: set[str]) -> str:
    """'train' | 'val' | 'test' | 'block' for an in-distribution instance."""
    if in_block(template, ops) is not None:   # before the M-set shortcut: blocks never reach training
        return "block"
    if template in m_set:
        return "train"
    b, p = bucket(key, config["pools"]["modulus"]), config["pools"]
    return "train" if b < p["train_below"] else "val" if b < p["val_below"] else "test"
