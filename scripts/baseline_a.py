"""
Baseline A, provenance only (RESEARCH_DESIGN.md §2): score a legacy [GEN] model on data-v1
items through a forced prefix.

The legacy model was trained to *generate* problems. Its prompt is
`[GEN] topic:<t> difficulty:<d> <SEP><JSON>`. We force the question it should solve,
`{"questionText": <q>, "hints": [`, then let it complete the hints and answerKey greedily, and
score the answerKey with the data-v1 oracle. These numbers are NOT comparable to the study's
arms: the legacy pipeline saw most ID instances during training, under a different task and
data distribution (including the F3 bug).

Usage (from mathvis-model/):
    .venv/Scripts/python.exe scripts/baseline_a.py --run runs/p1-legacy-repro-s0 [--splits test_id_unseen ood_longer]
"""

import argparse
import importlib
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mathgen.compat  # noqa: E402,F401
import torch  # noqa: E402

from mathgen.data.build import read_jsonl  # noqa: E402
from mathgen.eval.decode import greedy_decode  # noqa: E402
from mathgen.eval.oracle import math_correct  # noqa: E402
from mathgen.model import Transformer  # noqa: E402

LEGACY_EOS = 0   # src/tokenizer.py SPECIAL_TOKENS["<EOS>"]


def topic_for_template() -> dict[str, tuple[str, str]]:
    """First (topic, difficulty) whose legacy generator list contains each template."""
    legacy = importlib.import_module("dataset")   # legacy src/dataset.py
    out = {}
    for topic, by_diff in legacy.TOPIC_GENERATORS.items():
        for diff in ("easy", "medium", "hard"):
            for fn in by_diff.get(diff, []):
                out.setdefault(fn.__name__, (topic, diff))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="legacy run dir with best.pt and tokenizer.json")
    ap.add_argument("--splits", nargs="+", default=["test_id_unseen", "ood_longer"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    run = ROOT / args.run
    legacy_tok = importlib.import_module("tokenizer").BPETokenizer()
    legacy_tok.load(str(run / "tokenizer.json"))
    state = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    model = Transformer.from_legacy_state_dict(state["model"]).to(args.device).eval()
    topics = topic_for_template()

    report = {"run": args.run, "checkpoint": "best.pt", "val_loss": state.get("val_loss"),
              "label": "provenance only: legacy [GEN] model, forced prefix; saw most ID instances in training",
              "splits": {}}
    for split in args.splits:
        items = read_jsonl(ROOT / "data" / "eval" / f"{split}.jsonl")
        heads, prompts = [], []
        for it in items:
            topic, diff = topics[it["template"]]
            head = '{"questionText": ' + json.dumps(it["question"]) + ', "hints": ['   # legacy ensure_ascii=True
            heads.append(head)
            prompts.append(legacy_tok.encode(f"[GEN] topic:{topic} difficulty:{diff} <SEP><JSON>" + head))
        gens = greedy_decode(model, prompts, eos_id=LEGACY_EOS, context_len=model.cfg.context_len,
                             batch_size=128, device=args.device)
        by_t = defaultdict(list)
        for it, head, gen in zip(items, heads, gens):
            ok = False
            if gen and gen[-1] == LEGACY_EOS:
                try:
                    obj = json.loads(head + legacy_tok.decode(gen[:-1]))
                    ok = isinstance(obj.get("answerKey"), str) and math_correct(obj["answerKey"], it["answer"], it["atype"])
                except (json.JSONDecodeError, AttributeError):
                    ok = False
            by_t[it["template"]].append(ok)
        flat = [x for v in by_t.values() for x in v]
        report["splits"][split] = {"n": len(flat), "acc": sum(flat) / len(flat),
                                   "by_template": {t: sum(v) / len(v) for t, v in sorted(by_t.items())}}
        print(f"{split}: {sum(flat)}/{len(flat)} = {sum(flat) / len(flat):.4f}")
    out = ROOT / "research" / "artifacts" / f"baseline_a_{run.name}.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
