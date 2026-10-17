"""
Fit the three tokenizer arms on the pre-registered fit sample (data/fit/tokenizer_fit.jsonl)
and save them to data/tokenizers/{char,digit,bpe_num}.json, with sha256 values in
data/manifests/tokenizers.json. Deterministic: refitting gives identical files.

Usage (from mathvis-model/):
    .venv/Scripts/python.exe scripts/fit_tokenizers.py
"""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mathgen.data.build import read_jsonl  # noqa: E402
from mathgen.data.formats import prompt_text, target_text  # noqa: E402
from mathgen.tokenizers import ARMS, fit_tokenizer  # noqa: E402


def main() -> None:
    fit = read_jsonl(ROOT / "data" / "fit" / "tokenizer_fit.jsonl")
    texts = [prompt_text(i["question"]) + target_text(i["hints"], i["answer"]) for i in fit]
    out = ROOT / "data" / "tokenizers"
    out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for arm in ARMS:
        tok = fit_tokenizer(arm, texts)
        path = out / f"{arm}.json"
        tok.save(path)
        manifest[arm] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "vocab_size": len(tok.vocab),
                         "text_merges": len(tok.text_merges), "digit_merges": len(tok.digit_merges)}
        print(f"{arm:8s} vocab={len(tok.vocab)}")
    fit_sha = json.loads((ROOT / "data" / "manifests" / "build.json").read_text(encoding="utf-8"))["files"]
    (ROOT / "data" / "manifests" / "tokenizers.json").write_text(
        json.dumps({"fit_file_sha256": fit_sha["fit/tokenizer_fit.jsonl"]["sha256"], "tokenizers": manifest},
                   indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
