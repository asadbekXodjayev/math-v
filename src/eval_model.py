"""
Quick eval — does best.pt generate valid problems for all 10 topics (incl. calculus)?

Runs the model through its real [GEN] inference path for every topic × difficulty,
checks the output is valid JSON with the right keys, and prints calculus samples
for eyeball inspection. Forces CPU by default for stability.

Usage:
  python src/eval_model.py            # CPU (safe)
  python src/eval_model.py --cuda     # use GPU if you want
"""

import argparse
import json
import re
import sys
from pathlib import Path

import torch

from model import MathGen, MathGenConfig
from tokenizer import BPETokenizer
from dataset import TOPIC_GENERATORS

CKPT = Path(__file__).parent.parent / "checkpoints"
BEST = CKPT / "best.pt"
TOK = str(CKPT / "tokenizer.json")


def load(device):
    tok = BPETokenizer()
    tok.load(TOK)
    cfg = MathGenConfig(vocab_size=len(tok), use_grad_ckpt=False)
    m = MathGen(cfg).to(device)
    ck = torch.load(BEST, map_location=device)
    m.load_state_dict(ck["model"])
    m.eval()
    print(f"Loaded best.pt | val_loss={ck.get('val_loss', 0):.4f} | device={device}\n")
    return m, tok


def infer(m, tok, prompt, device, max_new=300, temp=0.7):
    ids = tok.encode(prompt)
    inp = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        out = m.generate(inp, max_new=max_new, temperature=temp)
    return tok.decode(out[0, len(ids):].tolist())


def extract_json(text):
    mt = re.search(r"\{.*\}", text, re.DOTALL)
    if not mt:
        return None
    try:
        return json.loads(mt.group())
    except Exception:
        return None


def is_valid(data):
    return bool(
        data
        and isinstance(data.get("questionText"), str)
        and data["questionText"].strip()
        and "answerKey" in data
        and isinstance(data.get("hints"), list)
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda", action="store_true", help="Use GPU (default: CPU for stability)")
    ap.add_argument("--n", type=int, default=3, help="Samples per topic×difficulty")
    args = ap.parse_args()

    device = torch.device("cuda" if (args.cuda and torch.cuda.is_available()) else "cpu")
    m, tok = load(device)

    topics = list(TOPIC_GENERATORS.keys())
    diffs = ["easy", "medium", "hard"]
    results = {}

    print("Generating samples (this runs the real [GEN] inference path)...\n")
    for t in topics:
        ok = total = 0
        examples = []
        for d in diffs:
            for _ in range(args.n):
                total += 1
                prompt = f"[GEN] topic:{t} difficulty:{d} <SEP><JSON>"
                try:
                    text = infer(m, tok, prompt, device)
                except Exception as e:
                    if len(examples) < 3:
                        examples.append((d, "ERROR", str(e)[:90]))
                    continue
                data = extract_json(text)
                if is_valid(data):
                    ok += 1
                    if "calculus" in t and len([e for e in examples if e[1] == "OK"]) < 1:
                        examples.append((d, "OK", data))
                else:
                    if len(examples) < 3:
                        examples.append((d, "BAD", (data if data else text)[:160] if isinstance(data if data else text, str) else data))
        results[t] = (ok, total, examples)

    # ── Report ──────────────────────────────────────────────────────────────
    print("=" * 72)
    print("VALID-JSON RATE PER TOPIC  (model-generated [GEN] problems)")
    print("=" * 72)
    for t, (ok, total, _) in results.items():
        flag = "OK " if ok == total else ("~  " if ok >= total * 0.6 else "XX ")
        bar = "#" * round(20 * ok / total)
        print(f"{flag} {t:42s} {ok:2d}/{total:2d}  {bar}")

    print("\n" + "-" * 72)
    print("CALCULUS SAMPLE OUTPUTS (eyeball these for correctness)")
    print("-" * 72)
    for t in topics:
        if "calculus" not in t:
            continue
        for d, status, content in results[t][2]:
            if isinstance(content, dict):
                q = content.get("questionText", "")
                a = content.get("answerKey", "")
                print(f"[{t}/{d}] {status}")
                print(f"    Q: {q}")
                print(f"    A: {a}")
            else:
                print(f"[{t}/{d}] {status}: {content}")

    overall_ok = sum(r[0] for r in results.values())
    overall_tot = sum(r[1] for r in results.values())
    print("\n" + "=" * 72)
    print(f"OVERALL: {overall_ok}/{overall_tot} valid JSON ({100 * overall_ok / overall_tot:.0f}%)")
    print("=" * 72)


if __name__ == "__main__":
    main()
