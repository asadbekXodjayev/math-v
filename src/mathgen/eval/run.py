"""
Evaluation of a trained run on every split (RESEARCH_DESIGN.md §5).

For each item it records:
- math-correct and exact match (failures count as wrong);
- JSON validity and truncation;
- copy consistency and an error class;
- the gold completion's log-likelihood and bits per character;
- whether digit and bpe-num tokenize the item's prompt / target numbers differently
  (`div_in` / `div_out`, the H1 negative-control covariate).

Output in runs/<exp_id>/:
    eval/<split>.jsonl     per-item records
    eval.json              per-split summaries; primary endpoint = G-set test-ID-unseen accuracy
"""

import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

from .. import compat  # noqa: F401  (must precede torch)

import torch

from ..data.build import read_jsonl
from ..data.config import load_config
from ..data.formats import prompt_text, target_text, tchar_len
from ..model import ModelConfig, Transformer
from ..training.data import TOK_ARMS, collate, encode_example, load_tokenizer
from ..training.loop import deterministic_algorithms
from .decode import greedy_decode
from .oracle import exact_match, math_correct, parses

EVAL_SPLITS = ("test_id_unseen", "val", "block", "ood_same", "ood_longer")
ERRORS = ("correct", "truncated", "format", "unparseable", "copy_error", "scratchpad_error", "wrong")


def parse_completion(text: str) -> dict:
    truncated = not text.endswith("<EOS>")
    body = text if truncated else text[:-len("<EOS>")]
    try:
        obj = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        obj = None
    ok = (isinstance(obj, dict) and isinstance(obj.get("answerKey"), str) and isinstance(obj.get("hints"), list)
          and all(isinstance(h, str) for h in obj["hints"]))
    return {"valid_json": ok, "truncated": truncated,
            "answer": obj["answerKey"] if ok else None, "hints": obj["hints"] if ok else None}


def _final_value(hint: str) -> str:
    """The text after the last '=' or '≈' in a hint ('Then multiply: 4 × 8 = 32' -> '32')."""
    return re.split(r"[=≈]", hint)[-1].strip()


def score_item(gold: dict, text: str) -> dict:
    p = parse_completion(text)
    rec = {"valid_json": p["valid_json"], "truncated": p["truncated"], "pred_answer": p["answer"],
           "correct": False, "exact": False, "copy_consistent": None}
    # Copy consistency applies when the gold hints end by stating the gold answer.
    applicable = bool(gold["hints"]) and _final_value(gold["hints"][-1]) == gold["answer"]
    if p["truncated"]:
        return {**rec, "error": "truncated"}
    if not p["valid_json"]:
        return {**rec, "error": "format"}
    answer = p["answer"]
    rec["correct"] = math_correct(answer, gold["answer"], gold["atype"])
    rec["exact"] = exact_match(answer, gold["answer"])
    pred_final = _final_value(p["hints"][-1]) if p["hints"] else None
    if applicable:
        rec["copy_consistent"] = pred_final == answer.strip()
    if rec["correct"]:
        error = "correct"
    elif not parses(answer, gold["atype"]):
        error = "unparseable"
    elif applicable and pred_final is not None and math_correct(pred_final, gold["answer"], gold["atype"]):
        error = "copy_error"         # the hints reached the right value, the answer didn't
    elif applicable and pred_final == answer.strip():
        error = "scratchpad_error"   # wrong value in the hints, copied faithfully
    else:
        error = "wrong"
    return {**rec, "error": error}


def train_recall_items(train_items: list[dict], per_template: int, seed: int) -> list[dict]:
    """Up to `per_template` distinct instances per template from a run's own training set, with
    the number of times each was seen (`exposure`). Deterministic in (seed, training set)."""
    exposure = Counter(i["key"] for i in train_items)
    first: dict[str, dict[str, dict]] = defaultdict(dict)
    for it in train_items:
        first[it["template"]].setdefault(it["key"], it)
    out = []
    for template in sorted(first):
        keys = sorted(first[template])
        pick = keys if len(keys) <= per_template else random.Random(f"recall|{seed}|{template}").sample(keys, per_template)
        out += [{**first[template][k], "split": "train_recall", "exposure": exposure[k]} for k in sorted(pick)]
    return out


@torch.no_grad()
def _gold_nll(model, examples: list[tuple[list[int], int]], pad_id: int, device, batch_size: int) -> list[float]:
    """Summed negative log-likelihood (nats) of each gold completion, teacher-forced."""
    out = []
    for s in range(0, len(examples), batch_size):
        b = collate(examples[s:s + batch_size], pad_id)
        logits = model(b["input"].to(device)).float()
        ce = torch.nn.functional.cross_entropy(logits.transpose(1, 2), b["target"].to(device), reduction="none")
        out += (ce * b["mask"].to(device)).sum(1).tolist()
    return out


def evaluate_items(model, tok, tok_arm: str, order: str, items: list[dict], context_len: int,
                   device, batch_size: int = 256, divergence=None) -> list[dict]:
    model.eval()
    numbers_prompt = TOK_ARMS[tok_arm][1]
    prompts = [tok.encode(prompt_text(it["question"]), numbers=numbers_prompt) for it in items]
    gens = greedy_decode(model, prompts, tok.eos_id, context_len, batch_size, device)
    examples = [encode_example(tok, tok_arm, it, order) for it in items]
    nll = _gold_nll(model, examples, tok.pad_id, device, batch_size)
    records = []
    for it, gen, ex, n in zip(items, gens, examples, nll):
        target = target_text(it["hints"], it["answer"], order)
        rec = {k: it.get(k) for k in ("key", "template", "family", "set", "split", "block_id", "ood_kind",
                                      "n_comp_steps", "magnitude", "exposure")}
        rec.update(score_item(it, tok.decode(gen)))
        rec.update({"gen_tokens": len(gen), "target_tokens": len(ex[0]) - ex[1], "gold_nll": n,
                    "gold_bpc": n / math.log(2) / tchar_len(target)})
        if divergence is not None:
            rec.update(divergence(it, order))
        records.append(rec)
    return records


def make_divergence(data_dir: str | Path = "data"):
    """Covariate: do T-digit and T-bpe-num split this item's prompt / target numbers differently?"""
    digit, bpe = load_tokenizer("digit", data_dir), load_tokenizer("bpe_num", data_dir)

    def divergence(item: dict, order: str) -> dict:
        p, t = prompt_text(item["question"]), target_text(item["hints"], item["answer"], order)
        return {"div_in": len(digit.encode(p)) != len(bpe.encode(p)),
                "div_out": len(digit.encode(t)) != len(bpe.encode(t))}
    return divergence


def summarize(records: list[dict]) -> dict:
    def acc(rs):
        return sum(r["correct"] for r in rs) / len(rs) if rs else None

    by = lambda key: {k: {"n": len(v), "acc": acc(v)} for k, v in sorted(_group(records, key).items())}
    return {
        "n": len(records), "acc": acc(records),
        "acc_gset": acc([r for r in records if r["set"] == "G"]),
        "exact": sum(r["exact"] for r in records) / len(records),
        "valid_json": sum(r["valid_json"] for r in records) / len(records),
        "errors": {e: sum(r["error"] == e for r in records) for e in ERRORS},
        "mean_gold_bpc": sum(r["gold_bpc"] for r in records) / len(records),
        "by_template": by("template"), "by_family": by("family"),
    }


def _group(records: list[dict], key: str) -> dict:
    g = defaultdict(list)
    for r in records:
        g[r[key]].append(r)
    return g


def _with_meta(items: list[dict], config: dict, m_set: set[str]) -> list[dict]:
    return [{**it, "family": config["templates"][it["template"]]["family"],
             "set": "M" if it["template"] in m_set else "G"} for it in items]


def eval_run(run_dir: str | Path, data_dir: str | Path | None = None, device: str = "cuda",
             splits: tuple[str, ...] = EVAL_SPLITS + ("train_recall",), batch_size: int = 256,
             recall_per_template: int = 200, limit: int | None = None) -> dict:
    """`data_dir` defaults to the run's own data directory. `limit`: evaluate only the first
    `limit` items of each split (smoke tests). Decoding runs in torch's deterministic mode, as
    training does, whether or not it follows training in the same process."""
    with deterministic_algorithms():
        return _eval_run(run_dir, data_dir, device, splits, batch_size, recall_per_template, limit)


def _eval_run(run_dir, data_dir, device, splits, batch_size, recall_per_template, limit) -> dict:
    start = time.perf_counter()
    run_dir = Path(run_dir)
    repo = Path(__file__).resolve().parents[3]
    final = torch.load(run_dir / "final.pt", map_location=device, weights_only=False)
    run, mcfg = final["run"], ModelConfig(**final["model_config"])
    data_dir = Path(data_dir or run["data_dir"])
    data_dir = data_dir if data_dir.is_absolute() else repo / data_dir
    model = Transformer(mcfg).to(device)
    model.load_state_dict(final["model"])
    tok = load_tokenizer(run["tok_arm"], data_dir)
    config = load_config(repo / "configs" / "base.yaml")
    m_set = set(json.loads((data_dir / "manifests" / "build.json").read_text(encoding="utf-8"))["m_set"])
    divergence = make_divergence(data_dir)

    (run_dir / "eval").mkdir(exist_ok=True)
    summary = {"exp_id": run["exp_id"], "splits": {}}
    for split in splits:
        if split == "train_recall":
            train = read_jsonl(data_dir / "train" / f"seed{run['seed']}.jsonl")[:run["n_train"]]
            items = _with_meta(train_recall_items(train, recall_per_template, run["seed"]), config, m_set)
            del train   # ~200 MB; only the sampled recall items are needed
        else:
            items = read_jsonl(data_dir / "eval" / f"{split}.jsonl")
        items = items[:limit] if limit else items
        records = evaluate_items(model, tok, run["tok_arm"], run["order"], items, mcfg.context_len, device,
                                 batch_size, divergence)
        with open(run_dir / "eval" / f"{split}.jsonl", "w", encoding="utf-8", newline="\n") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
        summary["splits"][split] = summarize(records)
    summary["primary_endpoint"] = summary["splits"].get("test_id_unseen", {}).get("acc_gset")
    summary["eval_wall_s"] = time.perf_counter() - start   # model + data loading included (G3 cost projection)
    (run_dir / "eval.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    return summary


def val_callback(cfg, limit: int | None = None, batch_size: int = 256):
    """For training's `on_eval`: G-set val math-correct accuracy and mean gold bits/char (pilot P2)."""
    repo = Path(__file__).resolve().parents[3]
    data_dir = Path(cfg.data_dir) if Path(cfg.data_dir).is_absolute() else repo / cfg.data_dir
    items = read_jsonl(data_dir / "eval" / "val.jsonl")
    items = items[:limit] if limit else items

    def on_eval(model, tok, step: int) -> dict:
        recs = evaluate_items(model, tok, cfg.tok_arm, cfg.order, items, cfg.context_len,
                              next(model.parameters()).device, batch_size)
        s = summarize(recs)
        return {"val_acc": s["acc"], "val_bpc": s["mean_gold_bpc"], "val_n": s["n"]}
    return on_eval
