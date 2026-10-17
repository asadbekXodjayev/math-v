"""
Evaluate a finished run on every split; writes runs/<id>/eval/*.jsonl and runs/<id>/eval.json.

Usage (from mathvis-model/):
    .venv/Scripts/python.exe scripts/eval.py --run runs/<exp_id> [--device cuda]
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mathgen.eval.run import eval_run  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    summary = eval_run(ROOT / args.run, device=args.device)
    print(json.dumps({s: {"n": v["n"], "acc": v["acc"], "acc_gset": v["acc_gset"]} for s, v in summary["splits"].items()},
                     indent=2))
    print(f"primary endpoint: {summary['primary_endpoint']:.4f}")


if __name__ == "__main__":
    main()
