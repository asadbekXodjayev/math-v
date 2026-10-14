"""
M1 audit probe — re-measures every repository-audit finding (F1–F9) directly from the
legacy code at tag `legacy-v0`. Read-only with respect to src/ and checkpoints/; the only
output is research/artifacts/audit_results.json (+ stdout).

Usage (from mathvis-model/):
    .venv/Scripts/python.exe research/audit_scripts/audit_probe.py [--skip-model]
"""

import argparse
import hashlib
import io
import json
import math
import random
import re
import sys
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
CKPT = ROOT / "checkpoints"
OUT = ROOT / "research" / "artifacts" / "audit_results.json"
sys.path.insert(0, str(SRC))

# Same torch/WMI import workaround the legacy server uses on this machine.
import platform as _platform
try:
    _uc = _platform.uname_result("Windows", "localhost", "10", "10.0", "AMD64")
    _uc.__dict__["processor"] = ""
    _platform._uname_cache = _uc
except Exception:
    pass
_platform.win32_ver = lambda *a, **k: ("10", "10.0", "", "")
_platform._wmi_query = lambda *a, **k: None

import torch  # noqa: E402

import dataset as legacy_ds  # noqa: E402
from dataset import TopicGenerators, build_all_samples, format_generate  # noqa: E402
from model import MathGen, MathGenConfig  # noqa: E402
from tokenizer import BPETokenizer  # noqa: E402
from verifier import compute_true_answer, smart_equal  # noqa: E402

RESULTS: dict = {}


def section(name: str) -> None:
    print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


def two_prop_p(k1: int, n1: int, k2: int, n2: int) -> float:
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (k1 / n1 - k2 / n2) / se
    return math.erfc(abs(z) / math.sqrt(2))


TEMPLATES = [n for n in dir(TopicGenerators) if not n.startswith("_")]


# ── F1 — architecture / parameter count vs README claims ────────────────────────────

def f1_params() -> None:
    section("F1  parameter counts (README claims ~5M params, d=256, vocab 8192)")
    out = {}
    for name, ck_path, tok_path in [
        ("current_best", CKPT / "best.pt", CKPT / "tokenizer.json"),
        ("baseline_v2_best", CKPT / "baseline_v2" / "best.pt", CKPT / "baseline_v2" / "tokenizer.json"),
    ]:
        tok = BPETokenizer()
        tok.load(str(tok_path))
        ck = torch.load(ck_path, map_location="cpu", weights_only=True)
        saved_cfg = ck.get("cfg", {})
        cfg = MathGenConfig(vocab_size=len(tok), use_grad_ckpt=False)
        m = MathGen(cfg)
        m.load_state_dict(ck["model"], strict=True)
        total = m.num_params()
        emb = cfg.vocab_size * cfg.d_model
        out[name] = {
            "vocab_size": len(tok),
            "n_merges": len(tok.merges),
            "d_model": cfg.d_model, "d_ff": cfg.d_ff, "n_layers": cfg.n_layers,
            "n_heads": cfg.n_heads, "context_len": cfg.context_len,
            "total_params": total,
            "embedding_params_tied": emb,
            "non_embedding_params": total - emb,
            "saved_cfg": saved_cfg,
            "epoch": ck.get("epoch"), "val_loss": ck.get("val_loss"),
            "global_step": ck.get("global_step"),
        }
        print(f"{name:18s} vocab={len(tok):5d}  total={total/1e6:6.2f}M  "
              f"non-emb={(total-emb)/1e6:6.2f}M  epoch={ck.get('epoch')}  "
              f"val_loss={ck.get('val_loss'):.4f}  step={ck.get('global_step')}")
    RESULTS["F1_params"] = out


# ── F2 — train-loss logging bug ──────────────────────────────────────────────────────

def f2_loss_logging() -> None:
    section("F2  train loss = epoch_loss / global_step (cumulative)  ->  understated ~epoch x")
    lines = [json.loads(l) for l in (CKPT / "train_log.jsonl").read_text().splitlines() if l.strip()]
    # The last contiguous run 1..10 is digit_retrain5 (fresh run, 901 optimizer steps/epoch).
    last_run = lines[-10:]
    assert [r["epoch"] for r in last_run] == list(range(1, 11)), "unexpected log layout"
    rows = []
    for r in last_run:
        corrected = r["train_loss"] * r["epoch"]   # epoch_loss/S = reported * (e*S)/S
        rows.append({"epoch": r["epoch"], "reported_train": r["train_loss"],
                     "corrected_train": corrected, "val": r["val_loss"]})
        print(f"epoch {r['epoch']:2d}  reported train={r['train_loss']:.4f}  "
              f"corrected train={corrected:.4f}  val={r['val_loss']:.4f}")

    log = (CKPT / "digit_retrain5.log").read_text(encoding="utf-8", errors="replace")
    tps = [(int(e), int(s), float(t)) for e, s, t in
           re.findall(r"E(\d+) step (\d+) \| loss=[\d.]+ \| lr=[\d.e+-]+ \| ([\d.]+)k tok/s", log)]
    e1_last = max((t for t in tps if t[0] == 1), key=lambda t: t[1])
    e10_last = max((t for t in tps if t[0] == 10), key=lambda t: t[1])
    print(f"logged tok/s: E1 last={e1_last[2]}k  E10 last={e10_last[2]}k "
          f"(cumulative tokens / per-epoch time -> inflates ~epoch x; also counts padding)")
    RESULTS["F2_loss_logging"] = {"digit_retrain5": rows,
                                  "logged_tok_per_s_k": {"E1_last": e1_last[2], "E10_last": e10_last[2]}}


# ── F3 — subtraction generator bug ──────────────────────────────────────────────────

def f3_subtraction() -> None:
    section("F3  arith_integer subtraction: a=max(a,b); b=min(a,b)  ->  'b - b = 0'")
    random.seed(0)
    n_sub = n_zero = 0
    for _ in range(100_000):
        q, _, ans = TopicGenerators.arith_integer()
        m = re.match(r"Compute (\d+) - (\d+)\.", q)
        if m:
            n_sub += 1
            n_zero += int(m.group(1) == m.group(2))
    expected_correct = 29 / 841     # P(a == b) for independent a, b ~ U{2..30}
    print(f"subtraction items: {n_sub}  with a == b: {n_zero} ({100*n_zero/n_sub:.1f}%)  "
          f"| a correct generator gives {100*expected_correct:.1f}%")
    RESULTS["F3_subtraction"] = {"n_subtraction": n_sub, "n_a_eq_b": n_zero,
                                 "frac_a_eq_b": n_zero / n_sub,
                                 "expected_frac_if_correct": expected_correct}


# ── F4 — verifier gaps ──────────────────────────────────────────────────────────────

def f4_verifier() -> None:
    section("F4  verifier: generator-control agreement + off-distribution probes")
    random.seed(1)
    control = {}
    for name in TEMPLATES:
        fn = getattr(TopicGenerators, name)
        ok = 0
        n = 300
        for _ in range(n):
            q, _, ans = fn()
            truth = compute_true_answer(q)
            ok += int(truth is not None and smart_equal(ans, truth[0], truth[1]))
        control[name] = ok / n
    bad = {k: v for k, v in control.items() if v < 1.0}
    print(f"generator control: {sum(v == 1.0 for v in control.values())}/{len(control)} templates "
          f"at 100%  {('below 100%: ' + str(bad)) if bad else ''}")

    # Probes the model can emit but the generator never does; truth computed independently.
    probes = [
        ("Solve for x: 7x + 11 = 72", str(Fraction(72 - 11, 7))),
        ("Solve for x: 4x + 3 = 12", str(Fraction(12 - 3, 4))),
        ("Solve for x: -2x + 3 = 7", "-2"),
        ("Compute -3 + 5.", "2"),
        ("Find ||v|| for v = [-3, 4].", "5"),
        ("Compute u·v for u = [-1, 2] and v = [3, 4].", "5"),
    ]
    probe_rows = []
    for q, true_ans in probes:
        got = compute_true_answer(q)
        status = ("UNVERIFIABLE (None)" if got is None
                  else "OK" if smart_equal(got[0], true_ans, got[1]) else f"WRONG TRUTH {got[0]!r}")
        probe_rows.append({"question": q, "true": true_ans, "verifier": got and got[0], "status": status})
        print(f"  {q:48s} true={true_ans:>5s}  verifier -> {status}")
    RESULTS["F4_verifier"] = {"generator_control_acc": control, "probes": probe_rows}


# ── F5 — instance spaces + legacy split leakage ─────────────────────────────────────

def f5_leakage() -> None:
    section("F5  instance-space sizes and legacy 92/8 row-level split leakage")
    random.seed(2)
    spaces = {}
    for name in TEMPLATES:
        fn = getattr(TopicGenerators, name)
        seen: set[str] = set()
        at_half = 0
        for i in range(60_000):
            seen.add(fn()[0])
            if i == 29_999:
                at_half = len(seen)
        spaces[name] = {"unique_questions_60k_draws": len(seen), "saturated": len(seen) == at_half}
    for name, s in sorted(spaces.items(), key=lambda kv: kv[1]["unique_questions_60k_draws"]):
        flag = "exhaustive" if s["saturated"] else ">= (not saturated)"
        print(f"  {name:22s} {s['unique_questions_60k_draws']:7d}  {flag}")

    # Re-create the digit_retrain5 data recipe (n_per_topic=1000 -> 47,000 samples) and its split.
    random.seed(3)
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        samples = build_all_samples(n_synthetic_per_topic=1000)
    split = int(len(samples) * 0.92)
    train, val = samples[:split], samples[split:]
    train_set = set(train)

    def qtext(s: str) -> str | None:
        m = re.search(r'"questionText": "(.*?)", "hints"', s)
        return m.group(1) if m else None

    train_q = {qtext(s) for s in train if s.startswith("[GEN]")}
    val_gen = [s for s in val if s.startswith("[GEN]")]
    exact_dup = sum(s in train_set for s in val_gen)
    q_dup = sum(qtext(s) in train_q for s in val_gen)
    print(f"legacy recipe: {len(samples)} samples | val [GEN]={len(val_gen)} | "
          f"val sample identical to a train sample: {100*exact_dup/len(val_gen):.1f}% | "
          f"val questionText seen in train: {100*q_dup/len(val_gen):.1f}%")
    RESULTS["F5_leakage"] = {
        "instance_spaces": spaces,
        "legacy_recipe_n_samples": len(samples),
        "val_gen_n": len(val_gen),
        "val_gen_exact_duplicate_of_train_frac": exact_dup / len(val_gen),
        "val_gen_question_seen_in_train_frac": q_dup / len(val_gen),
    }


# ── F6/F7 — legacy accuracy numbers are within noise ────────────────────────────────

def f7_stats() -> None:
    section("F6/F7  legacy reported accuracies: uncertainty and comparison")
    a = (238, 300)   # digit tokenizer run (digit_model_eval2.log)
    b = (147, 180)   # baseline_v2 BPE run (baseline_accuracy.log)
    ca, cb = wilson(*a), wilson(*b)
    p = two_prop_p(*a, *b)
    print(f"digit : {a[0]}/{a[1]} = {100*a[0]/a[1]:.1f}%  Wilson95 [{100*ca[0]:.1f}, {100*ca[1]:.1f}]")
    print(f"bpe   : {b[0]}/{b[1]} = {100*b[0]/b[1]:.1f}%  Wilson95 [{100*cb[0]:.1f}, {100*cb[1]:.1f}]")
    print(f"two-proportion z-test p = {p:.3f}  (and the runs also differ in data size, epochs, eval n)")
    RESULTS["F7_stats"] = {"digit": {"k": a[0], "n": a[1], "wilson95": ca},
                           "bpe": {"k": b[0], "n": b[1], "wilson95": cb},
                           "two_prop_p": p}


# ── F8 — tokenizer / format confounds ───────────────────────────────────────────────

def f8_tokenizer() -> None:
    section("F8  JSON ascii-escaping, unicode .isdigit(), sequence lengths vs context 256")
    random.seed(4)
    # Real legacy [GEN] samples: every (topic, difficulty) slot, exactly as build_all_samples does.
    gen_samples = []
    for topic, by_diff in legacy_ds.TOPIC_GENERATORS.items():
        for diff in by_diff:
            for _ in range(70):
                gen_samples.append(format_generate(topic, diff, legacy_ds._gen_problem(topic, diff)))
    q, h, a = TopicGenerators.mat_mul()
    per_template_text = {"mat_mul": format_generate("linear-algebra/matrix-multiplication", "easy",
                                                    {"questionText": q, "hints": h, "answerKey": a})}
    esc = [len(re.findall(r"\\u[0-9a-f]{4}", s)) for s in gen_samples]
    print(f"[GEN] samples with \\uXXXX escapes: {100*sum(e > 0 for e in esc)/len(esc):.1f}%  "
          f"(mean {sum(esc)/len(esc):.1f} escapes/sample)")
    print("example:", per_template_text["mat_mul"][:170])
    uni = {c: c.isdigit() for c in "²³₁₂"}
    print(f"str.isdigit on unicode: {uni}  -> the 'digit-level' rule also blocks merges on these")

    tok_rows = {}
    for tname, tpath in [("digit_1299", CKPT / "tokenizer.json"),
                         ("bpe_6031", CKPT / "baseline_v2" / "tokenizer.json")]:
        tok = BPETokenizer()
        tok.load(str(tpath))
        lens = [len(tok.encode(s)) for s in gen_samples]
        lens_sorted = sorted(lens)
        tok_rows[tname] = {"mean": sum(lens) / len(lens), "p95": lens_sorted[int(0.95 * len(lens))],
                           "max": max(lens), "frac_gt_257": sum(l > 257 for l in lens) / len(lens)}
        r = tok_rows[tname]
        print(f"  {tname:10s} tokens/sample mean={r['mean']:.0f} p95={r['p95']} max={r['max']} "
              f"truncated(>257)={100*r['frac_gt_257']:.1f}%")
    chars = sorted(len(s) for s in gen_samples)
    tok_rows["char_level"] = {"mean": sum(chars) / len(chars), "p95": chars[int(0.95 * len(chars))],
                              "max": chars[-1], "frac_gt_257": sum(c > 257 for c in chars) / len(chars)}
    r = tok_rows["char_level"]
    print(f"  {'char':10s} chars/sample  mean={r['mean']:.0f} p95={r['p95']} max={r['max']} "
          f"truncated(>257)={100*r['frac_gt_257']:.1f}%  (MathDataset truncates silently)")
    RESULTS["F8_tokenizer"] = {"escape_frac": sum(e > 0 for e in esc) / len(esc),
                               "isdigit_unicode": uni, "lengths": tok_rows}


# ── F9 — nondeterminism of the data pipeline ────────────────────────────────────────

def f9_determinism() -> None:
    section("F9  data pipeline determinism (legacy has no seed anywhere)")
    import contextlib
    digests = []
    for _ in range(2):
        random.seed()   # what an unseeded legacy run effectively does
        with contextlib.redirect_stdout(io.StringIO()):
            s = build_all_samples(n_synthetic_per_topic=50)
        digests.append(hashlib.sha256("\n".join(s).encode()).hexdigest()[:16])
    print(f"two unseeded builds -> sha256 {digests[0]} vs {digests[1]}  identical={digests[0] == digests[1]}")
    src_txt = "\n".join(p.read_text(encoding="utf-8") for p in SRC.glob("*.py"))
    n_seed_calls = len(re.findall(r"(manual_seed|random\.seed|np\.random\.seed)", src_txt))
    print(f"seed calls in src/*.py: {n_seed_calls}")
    RESULTS["F9_determinism"] = {"unseeded_build_digests": digests, "seed_calls_in_src": n_seed_calls}


# ── Smoke test — legacy model actually loads and generates ──────────────────────────

def smoke_model() -> None:
    section("SMOKE  legacy best.pt loads and generates on CPU (legacy sampling, seeded)")
    tok = BPETokenizer()
    tok.load(str(CKPT / "tokenizer.json"))
    cfg = MathGenConfig(vocab_size=len(tok), use_grad_ckpt=False)
    m = MathGen(cfg)
    m.load_state_dict(torch.load(CKPT / "best.pt", map_location="cpu", weights_only=True)["model"])
    m.eval()
    torch.manual_seed(0)
    rows = []
    for topic in ["basic-math/arithmetic", "linear-algebra/matrix-multiplication", "calculus/derivatives"]:
        prompt = f"[GEN] topic:{topic} difficulty:medium <SEP><JSON>"
        ids = tok.encode(prompt)
        t0 = time.perf_counter()
        out = m.generate(torch.tensor([ids]), max_new=256, temperature=0.7)
        dt = time.perf_counter() - t0
        text = tok.decode(out[0, len(ids):].tolist())
        n_new = out.shape[1] - len(ids)
        mt = re.search(r"\{.*\}", text, re.DOTALL)
        data = None
        try:
            data = json.loads(mt.group()) if mt else None
        except Exception:
            pass
        q = data.get("questionText") if data else None
        truth = compute_true_answer(q) if q else None
        ok = bool(truth and smart_equal(data.get("answerKey"), truth[0], truth[1]))
        rows.append({"topic": topic, "question": q, "answerKey": data and data.get("answerKey"),
                     "verified_truth": truth and truth[0], "correct": ok,
                     "new_tokens": n_new, "seconds": dt})
        print(f"  {topic:38s} {n_new:3d} tok in {dt:5.2f}s  correct={ok}  Q={q!r}")
    RESULTS["smoke_model"] = rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-model", action="store_true")
    args = ap.parse_args()
    RESULTS["env"] = {"python": sys.version.split()[0], "torch": torch.__version__,
                      "cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available()}
    f1_params()
    f2_loss_logging()
    f3_subtraction()
    f4_verifier()
    f5_leakage()
    f7_stats()
    f8_tokenizer()
    f9_determinism()
    if not args.skip_model:
        smoke_model()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(RESULTS, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
