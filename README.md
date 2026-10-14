# MathGen — Our Own AI

> **⚠ Audit notice (2026-10-14).** Parts of this README don't match the code. The trained
> checkpoint has **34.2M parameters** (d=512, d_ff=2048, 1,299-token digit-level vocab), not ~5M /
> d=256 / 8,192. The reported accuracies measure the model on problems it generated itself,
> mostly drawn from very small template spaces, so they are not evidence of generalization.
> The verified facts, known bugs and the research plan are in
> [`research/REPOSITORY_AUDIT.md`](research/REPOSITORY_AUDIT.md). This README will be rewritten
> with verified numbers once the controlled study is done.

Custom transformer model trained from scratch to generate math problems.
No external APIs. No open-source weights. Built and trained entirely by us.

## Architecture

```
model.py     — Decoder-only transformer (GPT-style)
               8 layers, 8 heads, d=256, SwiGLU FFN, RoPE positional encoding
               ~5M parameters — small enough to train on CPU/GPU

tokenizer.py — Custom BPE tokenizer trained on math problem text
               Vocab: 8192 tokens, includes LaTeX + JSON structure tokens

dataset.py   — Synthetic training data generator
               Produces (prompt, JSON problem) pairs from math templates
               Sources: synthetic templates + can add GSM8K / MATH dataset

train.py     — Training loop: AdamW + cosine LR schedule + gradient clipping
server.py    — FastAPI inference server at port 8787
```

## How the model works

**Input (prompt):**
```
<PROMPT> topic:linear-algebra/matrix-multiplication difficulty:medium type:compute-product <JSON>
```

**Output (generated):**
```json
{
  "questionText": "Compute AB where A = [[2,1],[3,0]] and B = [[1,4],[2,1]]",
  "hints": ["Multiply row 1 of A by column 1 of B", "c₁₁ = 2×1 + 1×2 = 4"],
  "answerKey": "[[4,9],[3,12]]"
}
```

The model learns the mapping from (topic, difficulty, type) → structured problem purely from training data.

## Step 1 — Install

```bash
cd mathvis-model
pip install -r requirements.txt
```

## Step 2 — Train

```bash
# CPU (takes ~30 min for 30 epochs on synthetic data)
python src/train.py

# GPU (much faster)
python src/train.py --device cuda

# More training data = better model
python src/train.py --device cuda --n-per-template 2000 --epochs 50
```

Training output:
- `checkpoints/tokenizer.json` — trained BPE tokenizer
- `checkpoints/best.pt`        — best model checkpoint
- `checkpoints/latest.pt`      — latest epoch checkpoint

## Step 3 — Serve

```bash
python src/server.py
# → MathGen API running at http://localhost:8787
```

Test it:
```bash
curl -X POST http://localhost:8787/generate \
  -H "Content-Type: application/json" \
  -d '{"topic_id":"linear-algebra/vectors","difficulty":"easy","problem_type":"dot-product"}'
```

## Step 4 — Connect to MathVis app

The Next.js app reads `MATHGEN_API_URL` from `.env.local`.
It's already set to `http://localhost:8787`.

Run both:
```bash
# Terminal 1 — model server
cd mathvis-model && python src/server.py

# Terminal 2 — Next.js app
cd mathvis-app && npm run dev
```

## Improving the model

**More synthetic data:**
- Add more topic templates to `dataset.py` → `SYNTHETIC_TEMPLATES`
- Increase `--n-per-template` when training

**Real math problem datasets (optional):**
- GSM8K: grade school math problems (public, on HuggingFace)
- MATH: competition problems (Hendrycks et al., 2021)
- Add a `load_gsm8k()` function in `dataset.py` using the `datasets` library

**Larger model:**
- Edit `MathGenConfig` in `model.py`: increase `n_layers`, `d_model`, `n_heads`
- Current (5M params): good for CPU training and demo
- 50M params (n_layers=12, d_model=512): needs GPU, much better quality

## Model design decisions

| Choice | What | Why |
|--------|------|-----|
| Decoder-only | GPT-style | Simplest for text generation; encoder-decoder adds complexity without clear benefit for this task |
| RMSNorm | Layer norm variant | More stable than LayerNorm, no mean subtraction needed |
| RoPE | Positional encoding | Relative position awareness, better extrapolation than learned embeddings |
| SwiGLU | Activation in FFN | Outperforms ReLU/GELU in practice (used in LLaMA, PaLM) |
| Weight tying | embed ↔ lm_head | Halves embedding parameters, improves training signal |
| BPE tokenizer | Custom, trained on math text | Handles LaTeX and JSON structure tokens correctly |
