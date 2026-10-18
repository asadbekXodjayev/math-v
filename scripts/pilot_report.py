"""
Markdown tables for research/PILOT_REPORT.md, computed from the pilot artifacts and run
directories, so no number in the report is typed in by hand.

    P2  val curve and the step budget S
    P3  LR sweep (val completion loss / accuracy) and the chosen LRs
    P4  REF x 5 seeds, seed SD, SESOI and the P4 gate
    costs  measured s/step per arm, per-run overhead and eval time, and the projected cost of
           the core grid (G3)

Usage (from mathvis-model/):
    .venv/Scripts/python.exe scripts/pilot_report.py      # writes research/artifacts/pilot_tables.md
"""

import json
import statistics
from pathlib import Path

import pilots

ROOT = Path(__file__).resolve().parents[1]
ART, RUNS = ROOT / "research" / "artifacts", ROOT / "runs"
# Core grid after G3 (RESEARCH_DESIGN.md §8), in run order: (stage, [(arm, runs)]).
GRID = [("C-tok", [("bpe_num", 3), ("char", 3)]),
        ("C-dec", [("dec_db", 3), ("dec_bd", 3)]),
        ("C-arch", [("learned", 3), ("none", 3), ("gelu", 3)]),
        ("C-af", [("answer_first", 3)]),
        ("C-data", [("ref", 9)]),      # N in {2K, 8K, 32K} x 3; same S steps and batch size as REF
        ("C-dtok", [("bpe_num", 3)])]  # T-bpe-num at N = 8K
# Arms without a P3 measurement of their own borrow one (estimates, named in the table).
BORROWED = {"answer_first": ("ref",), "dec_db": ("ref", "bpe_num"), "dec_bd": ("ref", "bpe_num")}


def load(name: str) -> dict | None:
    path = ART / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def read(run: Path, name: str) -> dict:
    return json.loads((run / name).read_text(encoding="utf-8"))


def s_per_step(run: Path) -> float:
    return read(run, "train_stats.json")["train_wall_s"] / read(run, "record.json")["config"]["steps"]


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def p2_section(p2: dict) -> list[str]:
    val_n = next(m["val_n"] for m in pilots.read_metrics(RUNS / pilots.p2_exp()) if "val_n" in m)
    out = ["## P2: step budget", "", f"| step | val math-correct (n = {val_n}) |", "|---:|---:|"]
    out += [f"| {s:,} | {pct(a)} |" for s, a in p2["val_curve"]]
    out += ["", f"Best {pct(p2['best_val_acc'])}; first checkpoint at ≥ 98% of best: step {p2['first_step_at_98pct']:,} "
                f"→ **S = {p2['S']:,}** (rule: {p2['rule']}). Training wall time {60 * p2['train_wall_h']:.0f} min.", ""]
    return out


def p3_section(p3: dict) -> list[str]:
    lrs = [f"{lr:g}" for lr in pilots.LR_GRID]
    out = [f"## P3: LR sweep ({p3['steps']:,} steps = 0.3·S rounded to 100, seed 0)", "",
           "| arm | " + " | ".join(f"LR {lr}: val loss / acc" for lr in lrs) + " | chosen |",
           "|---|" + "---:|" * (len(lrs) + 1)]
    for arm in pilots.SWEEP_ARMS:
        cells = [f"{v['val_loss']:.4f} / {pct(v['val_acc'])}" for v in (p3["results"][arm][lr] for lr in lrs)]
        out.append(f"| {arm} | " + " | ".join(cells) + f" | **{p3['chosen'][arm]:g}** |")
    out += ["", f"Rule: {p3['rule']}.", ""]
    return out


def p4_section(p4: dict) -> list[str]:
    splits = [k for k in next(iter(p4["per_seed"].values())) if k not in ("primary", "train_wall_h")]
    out = [f"## P4: REF × 5 seeds (S = {p4['S']:,}, LR {p4['lr']:g})", "",
           "| seed | primary: test_id_unseen, G-set | " + " | ".join(f"{s} (all)" for s in splits) + " |",
           "|---:|---:|" + "---:|" * len(splits)]
    for seed, v in p4["per_seed"].items():
        out.append(f"| {seed} | {pct(v['primary'])} | " + " | ".join(pct(v[s]) for s in splits) + " |")
    out += ["", f"- Mean primary {pct(p4['mean_primary'])}; seed SD {100 * p4['sd_primary']:.2f} pp.",
            f"- **SESOI = max(2 pp, 2 × SD) = {100 * p4['sesoi']:.2f} pp.**",
            f"- OOD-longer seed SD {100 * p4['sd_ood_longer']:.2f} pp → P4 gate (> 5 pp cuts arms): "
            f"**{'TRIGGERED' if p4['p4_gate_ood_sd_gt_5pp'] else 'not triggered'}**.", ""]
    return out


def cost_section(p0: dict, p3: dict, p4: dict | None) -> list[str]:
    p3_runs = {arm: [RUNS / pilots.p3_exp(arm, p3["steps"], lr) for lr in pilots.LR_GRID] for arm in pilots.SWEEP_ARMS}
    speed = {arm: statistics.fmean(s_per_step(r) for r in runs) for arm, runs in p3_runs.items()}
    for arm, sources in BORROWED.items():
        speed[arm] = statistics.fmean(speed[s] for s in sources)
    fresh = [read(r, "train_stats.json") for runs in p3_runs.values() for r in runs]
    fresh = [s for s in fresh if s["resumed_from_step"] == 0]
    overhead_s = statistics.fmean(s["last_session_wall_s"] - s["train_wall_s"] for s in fresh)
    deterministic = all(read(r, "record.json")["config"]["deterministic"] for runs in p3_runs.values() for r in runs)

    S = p4["S"] if p4 else load("p2_budget.json")["S"]
    p4_runs = [RUNS / pilots.p4_exp(seed, S) for seed in range(5)] if p4 else []
    if p4_runs:
        ref_at_S = statistics.fmean(s_per_step(r) for r in p4_runs)
        evals = [read(r, "eval.json") for r in p4_runs]
        eval_ref, eval_n = statistics.fmean(e["eval_wall_s"] for e in evals), sum(v["n"] for v in evals[0]["splits"].values())
        eval_src = "mean of the 5 P4 runs at S, recorded by eval_run"
    else:
        ref_at_S, eval_ref, eval_n = speed["ref"], p0["eval"]["wall_s"], p0["eval"]["n_items"]
        eval_src = "P0, a 1K-step model; P4 replaces it"
    scale = ref_at_S / speed["ref"]   # one condition for every arm: REF's speed at S over its P3 speed

    out = [f"## Measured throughput ({'deterministic mode, ' if deterministic else ''}P3 runs, mean of 3 LRs)", "",
           "| arm | s/step |", "|---|---:|"]
    out += [f"| {arm} | {speed[arm]:.4f} |" for arm in pilots.SWEEP_ARMS]
    if p4_runs:
        out += [f"| ref, P4 runs at S = {S:,} | {ref_at_S:.4f} |"]
    out += ["", f"- Every arm's s/step is scaled by {scale:.3f} (REF at S over REF in P3) for the projection.",
            f"- Per-run overhead outside the step loop (data encoding, provenance, final save): {overhead_s:.0f} s, "
            "mean of the P3 runs that did not resume; interpreter and CUDA start-up are not included.",
            f"- Full eval of a REF run ({eval_n:,} items incl. train-recall): {eval_ref:.0f} s ({eval_src}). "
            "Other arms' eval time is scaled by their s/step relative to REF (estimate). C-data and C-dtok "
            "runs at N < 128K have fewer train-recall items, so their eval time is overstated (conservative).",
            "- Arms without their own P3 runs borrow a speed: "
            + "; ".join(f"{a} = mean of {', '.join(s)}" for a, s in BORROWED.items()) + ".", "",
            f"## Projected core grid (G3), S = {S:,}", "",
            "| stage | arm | runs | train min/run | eval min/run | GPU-h |", "|---|---|---:|---:|---:|---:|"]
    total_runs = total_h = 0.0
    for stage, arms in GRID:
        for arm, n in arms:
            train_min = (S * speed[arm] * scale + overhead_s) / 60
            eval_min = eval_ref * speed[arm] / speed["ref"] / 60
            hours = n * (train_min + eval_min) / 60
            total_runs, total_h = total_runs + n, total_h + hours
            note = " (borrowed speed)" if arm in BORROWED else ""
            out.append(f"| {stage} | {arm}{note} | {n} | {train_min:.1f} | {eval_min:.1f} | {hours:.2f} |")
    out += [f"| **total** | | **{int(total_runs)}** | | | **{total_h:.1f}** |", ""]
    return out


def main() -> None:
    p0, p2 = load("p0_report.json"), load("p2_budget.json")
    p3, p4 = load("p3_lr_sweep.json"), load("p4_seed_variance.json")
    lines = ["<!-- generated by scripts/pilot_report.py; do not edit by hand -->", ""]
    lines += p2_section(p2) if p2 else ["## P2: pending", ""]
    lines += p3_section(p3) if p3 else ["## P3: pending", ""]
    lines += p4_section(p4) if p4 else ["## P4: pending", ""]
    if p2 and p3:
        lines += cost_section(p0, p3, p4)
    out = ART / "pilot_tables.md"   # a file, not stdout: the Windows console can't encode "≥"
    out.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
