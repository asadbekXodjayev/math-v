"""
P0 follow-up (research/artifacts/p0_determinism_followup.json): train REF seed 0 for 1,000
steps with or without torch's deterministic mode, so that pairs of runs can be compared bitwise.

Runs C, D and D2 were produced at commit 118709d by an equivalent scratch script that toggled
torch.use_deterministic_algorithms itself; RunConfig.deterministic (decisions.md I-008) now does
the same inside train().

Usage (from mathvis-model/):
    .venv/Scripts/python.exe scripts/p0_determinism_probe.py <exp_id> <deterministic: 0|1>
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mathgen.training import RunConfig, train  # noqa: E402


def main() -> None:
    exp_id, det = sys.argv[1], sys.argv[2] == "1"
    start = time.perf_counter()
    run = train(RunConfig(exp_id=exp_id, steps=1000, ckpt_every=500, deterministic=det,
                          notes=f"P0 follow-up, deterministic={det}"))
    lines = [json.loads(l) for l in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    print(json.dumps({"exp": exp_id, "det": det, "wall_s": time.perf_counter() - start,
                      "s_per_step": sum(l["elapsed_s"] for l in lines) / 1000, "final_loss": lines[-1]["train_loss"]}))


if __name__ == "__main__":
    main()
