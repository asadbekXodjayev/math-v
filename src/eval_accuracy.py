"""
Answer-correctness eval for MathGen.

The old eval_model.py only checked whether [GEN] output is *valid JSON* — never whether
the math is *correct*. This script fills that gap: it runs the [GEN] inference path, parses
the generated questionText, INDEPENDENTLY re-derives the correct answer via src/verifier.py
(never trusting the model's answerKey), and reports exact-match accuracy per topic.

Sources:
  generator : deterministic templates in dataset.py  (control; must be ~100% — it validates verifier.py)
  model     : the neural net's [GEN] output           (under test)

Usage (use the project venv python):
  .venv/Scripts/python.exe src/eval_accuracy.py --source generator --n 40
  .venv/Scripts/python.exe src/eval_accuracy.py --source model --n 8
"""

import sys, io, re, json, argparse
from pathlib import Path

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
except Exception:
    pass

SRC = Path(__file__).parent
sys.path.insert(0, str(SRC))

# torch/WMI import workaround (same as server.py)
import platform as _platform
try:
    _uc = _platform.uname_result("Windows", "localhost", "10", "10.0", "AMD64")
    _uc.__dict__["processor"] = ""
    _platform._uname_cache = _uc
except Exception:
    pass
_platform.win32_ver = lambda *a, **k: ("10", "10.0", "", "")
_platform._wmi_query = lambda *a, **k: None

from dataset import TOPIC_GENERATORS, _gen_problem
from verifier import compute_true_answer, answer_correct


# ── sources ──────────────────────────────────────────────────────────────────────

def gen_from_generator(topic, diff):
    p = _gen_problem(topic, diff)
    return (p["questionText"], p["answerKey"]) if p else (None, None)


def make_model_source(device_str, max_new, ck_dir):
    import torch
    from model import MathGen, MathGenConfig
    from tokenizer import BPETokenizer
    tok = BPETokenizer(); tok.load(str(ck_dir / "tokenizer.json"))
    cfg = MathGenConfig(vocab_size=len(tok), use_grad_ckpt=False)
    dev = torch.device(device_str)
    m = MathGen(cfg).to(dev)
    ck = torch.load(ck_dir / "best.pt", map_location=dev)
    m.load_state_dict(ck["model"]); m.eval()
    print(f"Loaded best.pt | val_loss={ck.get('val_loss', 0):.4f} | device={dev}\n")

    def gen(topic, diff):
        prompt = f"[GEN] topic:{topic} difficulty:{diff} <SEP><JSON>"
        ids = tok.encode(prompt)
        inp = torch.tensor([ids], dtype=torch.long, device=dev)
        with torch.inference_mode():
            out = m.generate(inp, max_new=max_new, temperature=0.7)
        text = tok.decode(out[0, len(ids):].tolist())
        mt = re.search(r"\{.*\}", text, re.DOTALL)
        if not mt:
            return ("__BADJSON__", None)
        try:
            data = json.loads(mt.group())
        except Exception:
            return ("__BADJSON__", None)
        return (data.get("questionText", ""), data.get("answerKey", None))
    return gen


# ── run / report ─────────────────────────────────────────────────────────────────

def run_source(gen_fn, n, diffs):
    per = {}
    for t in list(TOPIC_GENERATORS.keys()):
        c = dict(total=0, badjson=0, unparse=0, correct=0, wrong=0)
        wrong_ex = []
        for d in diffs:
            for _ in range(n):
                c["total"] += 1
                q, model_ans = gen_fn(t, d)
                if q == "__BADJSON__" or q is None:
                    c["badjson"] += 1; continue
                truth = compute_true_answer(q)
                if truth is None:
                    c["unparse"] += 1; continue
                if answer_correct(model_ans, truth[0], truth[1]):
                    c["correct"] += 1
                else:
                    c["wrong"] += 1
                    if len(wrong_ex) < 2:
                        wrong_ex.append((q[:70], model_ans, truth[0]))
        per[t] = (c, wrong_ex)
    return per


def report(name, per):
    print("=" * 78)
    print(f"ANSWER ACCURACY  —  source = {name}")
    print("=" * 78)
    print(f"{'topic':40s} {'acc':>7s}  {'ok/seen':>9s}  {'badJSON':>7s} {'unparse':>7s}")
    tot = dict(total=0, badjson=0, unparse=0, correct=0, wrong=0)
    for t, (c, _) in per.items():
        for k in tot:
            tot[k] += c[k]
        seen = c["correct"] + c["wrong"]
        acc = (100 * c["correct"] / seen) if seen else 0.0
        print(f"{t:40s} {acc:6.1f}% {c['correct']:3d}/{seen:<3d}   "
              f"{c['badjson']:5d}   {c['unparse']:5d}  {'#' * round(acc / 5)}")
    seen = tot["correct"] + tot["wrong"]
    acc = (100 * tot["correct"] / seen) if seen else 0.0
    print("-" * 78)
    print(f"{'OVERALL':40s} {acc:6.1f}% {tot['correct']:3d}/{seen:<3d}   "
          f"{tot['badjson']:5d}   {tot['unparse']:5d}")
    print(f"\n  total {tot['total']} | badJSON {tot['badjson']} | unverifiable {tot['unparse']} "
          f"| verified {seen} -> {tot['correct']} correct, {tot['wrong']} wrong")
    shown = 0
    print("\n  sample WRONG (model vs true):")
    for t, (_, ex) in per.items():
        for q, ma, tr in ex:
            print(f"    [{t.split('/')[-1]}] {q}\n        model: {ma!r:>14}   true: {tr!r}")
            shown += 1
            if shown >= 8:
                break
        if shown >= 8:
            break
    if shown == 0:
        print("    (none)")
    print()
    return acc, tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["generator", "model", "both"], default="both")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--ckpt-dir", default=str(SRC.parent / "checkpoints"), help="legacy-repro: run directory")
    ap.add_argument("--seed", type=int, default=0, help="legacy-repro: seeds sampling and generators")
    args = ap.parse_args()
    import random, torch
    random.seed(args.seed); torch.manual_seed(args.seed)
    ck_dir = Path(args.ckpt_dir)
    diffs = ["easy", "medium", "hard"]

    results = {}
    if args.source in ("generator", "both"):
        results["generator"] = report("generator (control)", run_source(gen_from_generator, args.n, diffs))

    if args.source in ("model", "both"):
        dev = "cuda" if args.cuda else "cpu"
        print(f"\nBuilding model source (device={dev}, max_new={args.max_new})...")
        gen_fn = make_model_source(dev, args.max_new, ck_dir)
        results["model"] = report("model (best.pt, under test)", run_source(gen_fn, args.n, diffs))

    out = ck_dir / "accuracy_report.json"
    out.write_text(json.dumps({k: {"overall_acc": v[0], **v[1]} for k, v in results.items()}, indent=2),
                   encoding="utf-8")
    print(f"Saved summary -> {out}")
    if "generator" in results and results["generator"][0] < 99:
        print("\n[!] generator control < 99% -> verifier.py has gaps; treat model numbers as a lower bound.")


if __name__ == "__main__":
    main()
