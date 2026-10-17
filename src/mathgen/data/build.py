"""
Builds data v1 (RESEARCH_DESIGN.md §3) into an output directory:

    eval/{test_id_unseen,val,block,ood_same,ood_longer}.jsonl
    fit/tokenizer_fit.jsonl
    train/seed{s}.jsonl                  # first N lines = the size-N training set (nested)
    manifests/build.json                 # sha256 of every file, subsets, sizes, shortfalls, context length

Every item is checked against the independent oracle as it is written; any disagreement
aborts the build. The output is byte-deterministic for a given config and code (no
timestamps).
"""

import hashlib
import json
import random
import re
from pathlib import Path

from ..eval.oracle import solve
from .formats import prompt_text, target_text, tchar_len
from .ranges import ENUMERATE_LIMIT, domains_for, enumerate_space, product_size, sample_operands, space_size
from .splits import BLOCKS, bucket, in_block, pool
from .templates import TEMPLATES

_CODE_FILES = ["data/templates.py", "data/ranges.py", "data/splits.py", "data/formats.py",
               "data/build.py", "eval/oracle.py"]


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def normalized_sha256(data: bytes) -> str:
    """sha256 with CRLF normalized to LF, so hashes don't depend on the checkout's line endings."""
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def _code_sha256() -> str:
    root = Path(__file__).resolve().parents[1]
    return normalized_sha256(b"".join(rel.encode() + b"\0" + (root / rel).read_bytes() for rel in _CODE_FILES))


class _Writer:
    """Writes JSONL, checks each item against the oracle, tracks the longest T-char sequence."""

    def __init__(self, out_dir: Path):
        self.out_dir, self.files, self.checked = out_dir, {}, 0
        self.max_len, self.max_len_item = 0, None

    def write(self, rel: str, items: list[dict]) -> list[bytes]:
        lines = []
        for it in items:
            if solve(it["question"]) != (it["answer"], it["atype"]):
                raise RuntimeError(f"oracle disagreement in {rel}: {it['question']!r} -> {it['answer']!r}")
            self.checked += 1
            n = tchar_len(prompt_text(it["question"]) + target_text(it["hints"], it["answer"]))
            if n > self.max_len:
                self.max_len, self.max_len_item = n, {"file": rel, "key": it["key"]}
            lines.append((json.dumps(it, ensure_ascii=False) + "\n").encode("utf-8"))
        path = self.out_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        data = b"".join(lines)
        path.write_bytes(data)
        per_template: dict[str, int] = {}
        for it in items:
            per_template[it["template"]] = per_template.get(it["template"], 0) + 1
        self.files[rel] = {"sha256": hashlib.sha256(data).hexdigest(), "n_items": len(items),
                           "per_template": dict(sorted(per_template.items()))}
        return lines


def _train_record(name: str, ops: dict) -> dict:
    q, hints, answer = TEMPLATES[name].render(ops)
    return {"template": name, "key": TEMPLATES[name].key(ops), "operands": ops,
            "question": q, "hints": hints, "answer": answer, "atype": TEMPLATES[name].atype}


def _eval_record(name: str, ops: dict, split: str, config: dict, m_set: set[str],
                 ood_kind: str | None = None) -> dict:
    tpl = TEMPLATES[name]
    q, hints, answer = tpl.render(ops)
    key = tpl.key(ops)
    return {
        "split": split, "template": name, "family": config["templates"][name]["family"],
        "set": "M" if name in m_set else "G", "key": key,
        "bucket": bucket(key, config["pools"]["modulus"]),
        "block_id": in_block(name, ops) if ood_kind is None else None, "ood_kind": ood_kind,
        "operands": ops, "question": q, "hints": hints, "answer": answer, "atype": tpl.atype,
        "n_hints": len(hints), "n_comp_steps": sum(bool(re.search(r"=\s*[-\[]?\d", h)) for h in hints),
        "magnitude": max(abs(v) for v in ops.values() if isinstance(v, int)),
        "tchar_len": tchar_len(prompt_text(q) + target_text(hints, answer)),
    }


def _pick(candidates: list[dict], k: int, rng: random.Random) -> list[dict]:
    return candidates if len(candidates) <= k else rng.sample(candidates, k)


def _distinct_samples(name: str, domains: dict, k: int, rng: random.Random, accept,
                      enumerated: list[dict] | None = None) -> list[dict]:
    """k distinct instances accepted by `accept` (fewer if fewer exist): small spaces are
    enumerated (pass `enumerated` to reuse a cached enumeration), large ones rejection-sampled."""
    tpl = TEMPLATES[name]
    if product_size(tpl, domains) <= ENUMERATE_LIMIT:
        space = enumerated if enumerated is not None else enumerate_space(tpl, domains)
        return _pick([o for o in space if accept(o)], k, rng)
    seen: dict[str, dict] = {}
    attempts = 0
    while len(seen) < k:
        attempts += 1
        if attempts > 1000 * k:
            raise RuntimeError(f"{name}: could not draw {k} distinct instances")
        ops = sample_operands(tpl, domains, rng)
        if accept(ops):
            seen.setdefault(tpl.key(ops), ops)
    return list(seen.values())


def _stream(config: dict, label: str, seed: int, n: int, m_set: set[str]) -> list[dict]:
    """Template uniform, then an instance uniform over the template's train pool (rejection)."""
    rng = random.Random(f"{label}|{seed}")
    names = sorted(TEMPLATES)
    domains = {name: domains_for(config, name, "id") for name in names}
    out = []
    while len(out) < n:
        name = rng.choice(names)
        while True:
            ops = sample_operands(TEMPLATES[name], domains[name], rng)
            if pool(name, TEMPLATES[name].key(ops), ops, config, m_set) == "train":
                break
        out.append(_train_record(name, ops))
    return out


def build(config: dict, out_dir: str | Path, config_sha256: str) -> dict:
    out_dir = Path(out_dir)
    names = sorted(TEMPLATES)
    sizes = {n: space_size(TEMPLATES[n], domains_for(config, n, "id")) for n in names}
    m_set = {n for n, s in sizes.items() if s < config["gset_min_space"]}
    g_set = [n for n in names if n not in m_set]
    if bad := sorted({BLOCKS[b][0] for b in config["blocks"]} & m_set):
        raise ValueError(f"block templates must be in the G-set, but {bad} fell into the M-set")
    k = config["split_sizes"]
    w = _Writer(out_dir)
    shortfalls: dict[str, dict] = {}
    id_space: dict[str, list[dict]] = {}

    def enumerated(name: str) -> list[dict] | None:
        dom = domains_for(config, name, "id")
        if product_size(TEMPLATES[name], dom) > ENUMERATE_LIMIT:
            return None
        if name not in id_space:
            id_space[name] = enumerate_space(TEMPLATES[name], dom)
        return id_space[name]

    def note(rel: str, cell: str, requested: int, got: int) -> None:
        if got < requested:
            shortfalls.setdefault(rel, {})[cell] = {"requested": requested, "got": got}

    for split, want in (("test_id_unseen", "test"), ("val", "val")):
        items = []
        for name in g_set:
            rng = random.Random(f"{split}|{name}")
            dom = domains_for(config, name, "id")
            accept = (lambda o, nm=name: pool(nm, TEMPLATES[nm].key(o), o, config, m_set) == want)
            found = _distinct_samples(name, dom, k[split], rng, accept, enumerated(name))
            note(f"eval/{split}.jsonl", name, k[split], len(found))
            items += [_eval_record(name, o, split, config, m_set) for o in found]
        w.write(f"eval/{split}.jsonl", items)

    items = []
    for block_id in config["blocks"]:
        name = BLOCKS[block_id][0]
        rng = random.Random(f"block|{block_id}")
        accept = (lambda o, nm=name, b=block_id: in_block(nm, o) == b)
        found = _distinct_samples(name, domains_for(config, name, "id"), k["block"], rng, accept, enumerated(name))
        note("eval/block.jsonl", block_id, k["block"], len(found))
        items += [_eval_record(name, o, "block", config, m_set) for o in found]
    w.write("eval/block.jsonl", items)

    for split in ("ood_same", "ood_longer"):
        items = []
        for name in names:
            dom = domains_for(config, name, split)
            if dom is None:
                continue
            rng = random.Random(f"{split}|{name}")
            found = _distinct_samples(name, dom, k[split], rng, lambda o: True)
            note(f"eval/{split}.jsonl", name, k[split], len(found))
            items += [_eval_record(name, o, split, config, m_set, ood_kind=split) for o in found]
        w.write(f"eval/{split}.jsonl", items)

    fit = config["tokenizer_fit"]
    w.write("fit/tokenizer_fit.jsonl", _stream(config, "fit", fit["stream_seed"], fit["n"], m_set))

    subsets = {}
    for seed in config["train"]["seeds"]:
        lines = w.write(f"train/seed{seed}.jsonl", _stream(config, "train", seed, config["train"]["n_max"], m_set))
        subsets[str(seed)] = {str(n): hashlib.sha256(b"".join(lines[:n])).hexdigest()
                              for n in config["train"]["n_subsets"]}

    context_length = 256
    while context_length < w.max_len:
        context_length *= 2
    summary = {
        "config_sha256": config_sha256,
        "code_sha256": _code_sha256(),
        "space_sizes": sizes,
        "g_set": g_set,
        "m_set": sorted(m_set),
        "files": w.files,
        "shortfalls": shortfalls,
        "train_subsets": subsets,
        "oracle_agreement": {"checked": w.checked, "disagreements": 0},
        "context": {"max_tchar_len": w.max_len, "argmax": w.max_len_item, "context_length": context_length,
                    "rule": "smallest power of two >= longest T-char sequence over all built files, min 256"},
    }
    (out_dir / "manifests").mkdir(parents=True, exist_ok=True)
    (out_dir / "manifests" / "build.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                                                      encoding="utf-8", newline="\n")  # LF on every OS
    return summary
