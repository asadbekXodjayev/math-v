"""
Pilot P0 (RESEARCH_DESIGN.md §8): environment, throughput and determinism.

Trains REF with seed 0 for 1,000 steps twice under different exp_ids, then compares the
metrics logs and final weights exactly, times a full evaluation of the first run, and writes
research/artifacts/p0_report.json.

Usage (from mathvis-model/):
    .venv/Scripts/python.exe scripts/pilot_p0.py
"""

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mathgen.compat  # noqa: E402,F401
import torch  # noqa: E402

from mathgen.eval.run import eval_run  # noqa: E402
from mathgen.training import RunConfig, train  # noqa: E402

STEPS = 1000


def power_status() -> str:
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Battery).BatteryStatus"],
                             capture_output=True, text=True, timeout=60).stdout.strip()
        return {"1": "battery (discharging)", "2": "AC"}.get(out, f"BatteryStatus={out}")
    except Exception as e:
        return f"unknown ({e!r})"


def main() -> None:
    report = {"power": power_status(), "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "runs": {}}
    dirs = {}
    for tag in ("a", "b"):
        cfg = RunConfig(exp_id=f"p0-ref-s0-st{STEPS}-{tag}", steps=STEPS, ckpt_every=500)
        start = time.perf_counter()
        dirs[tag] = train(cfg)
        lines = [json.loads(l) for l in (dirs[tag] / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
        report["runs"][tag] = {"wall_s": time.perf_counter() - start,
                               "mean_tok_s": sum(l["tok_s"] for l in lines) / len(lines),
                               "s_per_step": sum(l["elapsed_s"] for l in lines) / STEPS,
                               "final_train_loss": lines[-1]["train_loss"],
                               "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20}

    strip = lambda p: [{k: v for k, v in json.loads(l).items() if k not in ("tok_s", "elapsed_s")}
                       for l in (p / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    wa = torch.load(dirs["a"] / "final.pt", weights_only=True)["model"]
    wb = torch.load(dirs["b"] / "final.pt", weights_only=True)["model"]
    max_diff = max((wa[k].float() - wb[k].float()).abs().max().item() for k in wa)
    la, lb = strip(dirs["a"]), strip(dirs["b"])
    report["determinism"] = {
        "metrics_identical": la == lb,
        "weights_bitwise_identical": all(torch.equal(wa[k], wb[k]) for k in wa),
        "max_abs_weight_diff": max_diff,
        "max_abs_train_loss_diff": max(abs(x["train_loss"] - y["train_loss"]) for x, y in zip(la, lb)),
    }

    start = time.perf_counter()
    summary = eval_run(dirs["a"], device="cuda")
    report["eval"] = {"wall_s": time.perf_counter() - start,
                      "n_items": sum(s["n"] for s in summary["splits"].values()),
                      "primary_endpoint_after_1k_steps": summary["primary_endpoint"]}
    report["power_after"] = power_status()
    out = ROOT / "research" / "artifacts" / "p0_report.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
