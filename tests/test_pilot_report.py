"""scripts/pilot_report.py: the tables of research/PILOT_REPORT.md are computed from the pilot artifacts."""

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def report(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("pilot_report", ROOT / "scripts" / "pilot_report.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "ART", tmp_path / "artifacts")
    monkeypatch.setattr(mod, "RUNS", tmp_path / "runs")
    mod.ART.mkdir()
    yield mod
    sys.modules.pop("pilots", None)


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def fake_run(runs: Path, name: str, steps: int, s_per_step: float, overhead_s: float = 60.0,
             eval_s: float | None = None) -> None:
    run = runs / name
    write_json(run / "record.json", {"config": {"steps": steps, "deterministic": True}})
    write_json(run / "train_stats.json", {"train_wall_s": steps * s_per_step, "resumed_from_step": 0,
                                          "last_session_wall_s": steps * s_per_step + overhead_s})
    if eval_s is not None:
        (run / "final.pt").write_bytes(b"")
        write_json(run / "eval.json", {"eval_wall_s": eval_s,
                                       "splits": {"val": {"n": 644}, "train_recall": {"n": 4600}}})
        os.utime(run / "final.pt", (1_000.0, 1_000.0))
        os.utime(run / "eval.json", (1_000.0 + 5 * 3600, 1_000.0 + 5 * 3600))   # re-evaluated hours later


def pilot_artifacts(mod, S=10_000, steps3=8_200, s_per_step=0.06, arm_speed=None) -> None:
    """P0, P2 and P3 artifacts; every P3 run trains at `s_per_step` (or arm_speed[arm]) plus 60 s overhead."""
    write_json(mod.ART / "p0_report.json", {"eval": {"wall_s": 70.0, "n_items": 11_195}})
    write_json(mod.ART / "p2_budget.json", {"rule": "r", "val_curve": [[2500, 0.5], [27500, 0.92]], "best_val_acc": 0.93,
                                            "first_step_at_98pct": 27500, "S": S, "train_wall_h": 0.5})
    (mod.RUNS / mod.pilots.p2_exp()).mkdir(parents=True)
    (mod.RUNS / mod.pilots.p2_exp() / "metrics.jsonl").write_text(
        json.dumps({"step": 2500, "val_acc": 0.5, "val_n": 644}) + "\n", encoding="utf-8")
    results = {arm: {f"{lr:g}": {"val_loss": 0.01, "val_acc": 0.8} for lr in mod.pilots.LR_GRID}
               for arm in mod.pilots.SWEEP_ARMS}
    write_json(mod.ART / "p3_lr_sweep.json", {"rule": "r", "steps": steps3, "results": results,
                                              "chosen": {arm: 5e-4 for arm in mod.pilots.SWEEP_ARMS}})
    for arm in mod.pilots.SWEEP_ARMS:
        for lr in mod.pilots.LR_GRID:
            fake_run(mod.RUNS, mod.pilots.p3_exp(arm, steps3, lr), steps3, (arm_speed or {}).get(arm, s_per_step))


def p4_artifacts(mod, S=10_000, s_per_step=0.09, eval_s=60.0) -> None:
    per_seed = {str(s): {"primary": 0.5, "test_id_unseen": 0.6, "ood_longer": 0.4, "train_wall_h": 0.2}
                for s in range(5)}
    write_json(mod.ART / "p4_seed_variance.json", {
        "S": S, "lr": 5e-4, "per_seed": per_seed, "mean_primary": 0.5, "sd_primary": 0.0,
        "sd_ood_longer": 0.0, "sesoi": 0.02, "p4_gate_ood_sd_gt_5pp": False})
    for seed in range(5):
        fake_run(mod.RUNS, mod.pilots.p4_exp(seed, S), S, s_per_step, eval_s=eval_s)


def rows(mod) -> list[str]:
    mod.main()
    return (mod.ART / "pilot_tables.md").read_text(encoding="utf-8").splitlines()


def row(mod, prefix: str) -> str:
    return next(l for l in rows(mod) if l.startswith(prefix))


def test_tables_are_written_to_a_utf8_file(report):
    """Printing '≥' to a cp1251 Windows console raised UnicodeEncodeError."""
    pilot_artifacts(report, S=27_500)
    text = "\n".join(rows(report))
    assert "≥ 98% of best" in text and "**S = 27,500**" in text


def test_p3_header_states_the_rounding_of_0_3_S(report):
    """0.3 * 27,500 = 8,250, which pilots.p3 rounds to 8,200 steps."""
    pilot_artifacts(report, S=27_500)
    assert row(report, "## P3") == "## P3: LR sweep (8,200 steps = 0.3·S rounded to 100, seed 0)"


def test_projected_grid_counts_the_36_core_runs_with_overhead_and_eval(report):
    """RESEARCH_DESIGN.md §8: C-tok 6 + C-arch 9 + C-af 3 + C-data 9 + C-dec 6 + C-dtok 3 = 36 runs."""
    pilot_artifacts(report, S=10_000, s_per_step=0.06)
    # no P4 yet: 600 s training + 60 s per-run overhead + 70 s eval (P0) per run -> 36 * 730 s = 7.30 h
    assert row(report, "| **total**") == "| **total** | | **36** | | | **7.3** |"
    assert "P0, a 1K-step model" in "\n".join(rows(report))


def test_every_arm_is_scaled_to_the_speed_measured_at_S_once_p4_exists(report):
    """P4's REF runs at S set the scale for all arms, so REF and non-REF rows share one condition."""
    pilot_artifacts(report, S=10_000, s_per_step=0.06, arm_speed={"char": 0.12})
    p4_artifacts(report, S=10_000, s_per_step=0.09, eval_s=60.0)        # REF 1.5x slower at S than in P3
    # REF: 10K * 0.09 s = 15 min + 1 min overhead; eval 60 s from eval.json, not the 5 h mtime gap
    assert row(report, "| C-data | ref |") == "| C-data | ref | 9 | 16.0 | 1.0 | 2.55 |"
    # char: 0.12 * 1.5 = 0.18 s/step -> 30 min + 1; eval scaled by s/step relative to REF -> 2 min
    assert row(report, "| C-tok | char |") == "| C-tok | char | 3 | 31.0 | 2.0 | 1.65 |"
    # dec: mean of REF and bpe_num P3 speeds, scaled the same way
    assert row(report, "| C-dec | dec_db") .endswith("| 3 | 16.0 | 1.0 | 0.85 |")


def test_p4_table_labels_the_primary_endpoint_apart_from_split_accuracy(report):
    pilot_artifacts(report)
    p4_artifacts(report)
    header = row(report, "| seed |")
    assert header == "| seed | primary: test_id_unseen, G-set | test_id_unseen (all) | ood_longer (all) |"
