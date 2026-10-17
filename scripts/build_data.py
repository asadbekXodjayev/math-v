"""
Build data v1 (RESEARCH_DESIGN.md §3) from a config.

Usage (from mathvis-model/):
    .venv/Scripts/python.exe scripts/build_data.py --config configs/base.yaml [--out data]

Writes data/{eval,fit,train}/*.jsonl (gitignored, regenerable) and data/manifests/build.json
(committed). Running it twice must give an identical manifest.
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mathgen.data.build import build  # noqa: E402
from mathgen.data.config import config_sha256, load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    # Relative paths are resolved against the repo root, so the script works from any directory.
    config_path, out_dir = ROOT / args.config, ROOT / args.out
    start = time.perf_counter()
    summary = build(load_config(config_path), out_dir, config_sha256(config_path))
    print(json.dumps({
        "files": {k: v["n_items"] for k, v in summary["files"].items()},
        "oracle_agreement": summary["oracle_agreement"],
        "context": summary["context"],
        "g_set": len(summary["g_set"]), "m_set": summary["m_set"],
        "shortfalls": summary["shortfalls"],
    }, indent=2))
    print(f"built in {time.perf_counter() - start:.1f}s -> {args.out}/manifests/build.json")


if __name__ == "__main__":
    main()
