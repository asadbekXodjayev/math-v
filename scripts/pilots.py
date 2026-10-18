"""
Pilots P1-P4 (RESEARCH_DESIGN.md §8). Each stage applies the pre-registered rule and writes its
result to research/artifacts/. Every run resumes from its checkpoint, so a stage can be
re-launched after an interruption.

    p1  legacy 34M reproduction (Baseline A) from the legacy-repro worktree, then its legacy
        self-consistency eval and forced-prefix scoring on data v1
    p2  REF for 30K steps, val accuracy every 2.5K -> step budget S (§4.3)
    p3  LR sweep {5e-4, 1e-3, 2e-3} x 6 arms at 0.3*S, seed 0 -> per-arm LR (§4.3)
    p4  REF x 5 seeds at S with the chosen LR, full eval -> seed SD, SESOI, P4 gate (§7)

Usage (from mathvis-model/):
    .venv/Scripts/python.exe scripts/pilots.py p2 [p3 p4 ...]
"""

import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

ART = ROOT / "research" / "artifacts"
PY = sys.executable
LR_GRID = (5e-4, 1e-3, 2e-3)
SWEEP_ARMS = ("ref", "char", "bpe_num", "learned", "none", "gelu")
P2_STEPS, P2_EVAL_EVERY = 30_000, 2_500


def sh(cmd: list[str], cwd: Path = ROOT) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def write(name: str, obj: dict) -> None:
    (ART / name).write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(obj, indent=2), flush=True)


def read_metrics(run: Path) -> list[dict]:
    return [json.loads(l) for l in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]


def train_wall_h(run: Path) -> float:
    return sum(m.get("elapsed_s", 0) for m in read_metrics(run)) / 3600


def p1() -> None:
    worktree, run_dir = ROOT / ".worktrees" / "legacy-repro", ROOT / "runs" / "p1-legacy-repro-s0"
    start = time.perf_counter()
    if not (run_dir / "best.pt").exists() or not (run_dir / "train_log.jsonl").exists() or \
            len((run_dir / "train_log.jsonl").read_text().splitlines()) < 10:
        sh([PY, "-B", "src/train.py", "--fresh", "--epochs", "10", "--batch-size", "12", "--grad-accum", "4",
            "--n-per-topic", "1000", "--seed", "0", "--run-dir", str(run_dir)], cwd=worktree)
    train_h = (time.perf_counter() - start) / 3600
    sh([PY, "-B", "src/eval_accuracy.py", "--source", "both", "--n", "10", "--cuda", "--seed", "0",
        "--ckpt-dir", str(run_dir)], cwd=worktree)
    sh([PY, "-B", "scripts/baseline_a.py", "--run", "runs/p1-legacy-repro-s0"])
    log = [json.loads(l) for l in (run_dir / "train_log.jsonl").read_text().splitlines()]
    write("p1_legacy_repro.json", {
        "command": "legacy-repro worktree: src/train.py --fresh --epochs 10 --batch-size 12 --grad-accum 4 "
                   "--n-per-topic 1000 --seed 0",
        "train_log": log, "wall_h_this_session": train_h,
        "self_consistency": json.loads((run_dir / "accuracy_report.json").read_text()),
        "original_final_run": {"val_loss": 0.1331, "corrected_train_loss": 0.1328, "self_consistency": "238/300 = 79.3%"},
    })


def p2() -> int:
    run = ROOT / "runs" / f"p2-ref-s0-st{P2_STEPS}"
    sh([PY, "-B", "scripts/train.py", "--config", "configs/arms/ref.yaml", "--seed", "0", "--steps", str(P2_STEPS),
        "--eval-every", str(P2_EVAL_EVERY), "--exp-id", run.name])
    curve = [(m["step"], m["val_acc"]) for m in read_metrics(run) if "val_acc" in m]
    best = max(a for _, a in curve)
    first = min(s for s, a in curve if a >= 0.98 * best)
    S = min(30_000, max(10_000, math.ceil(first / 2_500) * 2_500))
    write("p2_budget.json", {"rule": "smallest checkpoint with val acc >= 98% of best, rounded up to 2.5K, "
                                     "clamped to [10K, 30K]", "val_curve": curve, "best_val_acc": best,
                             "first_step_at_98pct": first, "S": S, "train_wall_h": train_wall_h(run)})
    return S


def budget() -> int:
    return json.loads((ART / "p2_budget.json").read_text())["S"]


def p3_val(exp: str) -> None:
    """Val completion loss and accuracy of one finished P3 run, written to runs/<exp>/p3_val.json.
    Runs in its own process so the orchestrator never initialises CUDA (~0.6 GB of host RAM)."""
    import mathgen.compat  # noqa: F401  (must precede torch)
    import torch
    from mathgen.data.build import read_jsonl
    from mathgen.eval.run import evaluate_items
    from mathgen.model import ModelConfig, Transformer
    from mathgen.training import load_tokenizer
    from mathgen.training.loop import deterministic_algorithms

    run = ROOT / "runs" / exp
    final = torch.load(run / "final.pt", map_location="cuda", weights_only=False)
    model = Transformer(ModelConfig(**final["model_config"])).cuda()
    model.load_state_dict(final["model"])
    tok_arm = final["run"]["tok_arm"]
    with deterministic_algorithms():
        recs = evaluate_items(model, load_tokenizer(tok_arm), tok_arm, final["run"]["order"],
                              read_jsonl(ROOT / "data" / "eval" / "val.jsonl"), model.cfg.context_len, "cuda")
    out = {"val_loss": statistics.fmean(r["gold_nll"] / r["target_tokens"] for r in recs),
           "val_acc": statistics.fmean(r["correct"] for r in recs), "val_n": len(recs)}
    tmp = run / "p3_val.json.tmp"   # atomic: an interrupted write never leaves a half file that resume trusts
    tmp.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")
    tmp.replace(run / "p3_val.json")


def p3() -> dict:
    steps = round(0.3 * budget() / 100) * 100
    results, chosen = {}, {}
    for arm in SWEEP_ARMS:
        for lr in LR_GRID:
            exp = f"p3-{arm}-s0-st{steps}-lr{lr:g}"
            sh([PY, "-B", "scripts/train.py", "--config", f"configs/arms/{arm}.yaml", "--seed", "0",
                "--steps", str(steps), "--lr", str(lr), "--exp-id", exp])
            run = ROOT / "runs" / exp
            if not (run / "p3_val.json").exists():
                sh([PY, "-B", "scripts/pilots.py", "p3val", exp])
            val = json.loads((run / "p3_val.json").read_text(encoding="utf-8"))
            results.setdefault(arm, {})[f"{lr:g}"] = {"val_loss": val["val_loss"], "val_acc": val["val_acc"],
                                                      "train_wall_h": train_wall_h(run)}
        by_lr = results[arm]
        best = min(v["val_loss"] for v in by_lr.values())
        chosen[arm] = min(float(lr) for lr, v in by_lr.items() if v["val_loss"] <= best * 1.01)
    for arm, lr in chosen.items():                      # write the chosen LR into the arm configs
        path = ROOT / "configs" / "arms" / f"{arm}.yaml"
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        lines = [l for l in path.read_text(encoding="utf-8").splitlines() if not l.startswith("lr:")]
        path.write_text("\n".join(lines) + f"\nlr: {lr:g}   # chosen by pilot P3\n", encoding="utf-8", newline="\n")
    for child, parent in (("answer_first", "ref"), ("dec_db", "ref"), ("dec_bd", "ref")):
        path = ROOT / "configs" / "arms" / f"{child}.yaml"
        lines = [l for l in path.read_text(encoding="utf-8").splitlines() if not l.startswith("lr:")]
        path.write_text("\n".join(lines) + f"\nlr: {chosen[parent]:g}   # parent arm {parent} (P3)\n",
                        encoding="utf-8", newline="\n")
    write("p3_lr_sweep.json", {"rule": "lowest final val completion loss (per-sequence normalized); "
                                        "ties within 1% -> lower LR", "steps": steps, "results": results,
                               "chosen": chosen})
    return chosen


def p4() -> dict:
    S = budget()
    lr = yaml.safe_load((ROOT / "configs" / "arms" / "ref.yaml").read_text())["lr"]
    per_seed = {}
    for seed in range(5):
        exp = f"ref-s{seed}-n128000-st{S}"
        sh([PY, "-B", "scripts/train.py", "--config", "configs/arms/ref.yaml", "--seed", str(seed),
            "--steps", str(S), "--exp-id", exp, "--then-eval"])
        run = ROOT / "runs" / exp
        ev = json.loads((run / "eval.json").read_text())
        per_seed[seed] = {"primary": ev["primary_endpoint"],
                          **{s: v["acc"] for s, v in ev["splits"].items()},
                          "train_wall_h": train_wall_h(run)}
    sd = lambda key: statistics.stdev(v[key] for v in per_seed.values())
    sd_primary, sd_ood = sd("primary"), sd("ood_longer")
    write("p4_seed_variance.json", {
        "S": S, "lr": lr, "per_seed": per_seed,
        "mean_primary": statistics.fmean(v["primary"] for v in per_seed.values()),
        "sd_primary": sd_primary, "sd_ood_longer": sd_ood,
        "sesoi": max(0.02, 2 * sd_primary),
        "p4_gate_ood_sd_gt_5pp": sd_ood > 0.05,
    })
    return per_seed


STAGES = {"p1": p1, "p2": p2, "p3": p3, "p4": p4}

if __name__ == "__main__":
    if sys.argv[1:2] == ["p3val"]:
        p3_val(sys.argv[2])
        sys.exit()
    for stage in sys.argv[1:] or ["p2", "p3", "p4"]:
        print(f"\n===== pilot {stage} =====", flush=True)
        STAGES[stage]()
