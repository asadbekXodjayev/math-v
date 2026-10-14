# MathGen — Repository Audit (Milestone 1)

**Status:** complete. Awaiting user review (gate G1) before any refactor or experiment.
**Audited snapshot:** git tag `legacy-v0` (commit `ea6caf9`), the code exactly as found.
**Checkpoint provenance:** sha256 in [`artifacts/CHECKPOINTS.sha256`](artifacts/CHECKPOINTS.sha256). Checkpoints are not in git.
**Measurements:** every number below comes from [`audit_scripts/audit_probe.py`](audit_scripts/audit_probe.py) (seeded and deterministic), saved to [`artifacts/audit_results.json`](artifacts/audit_results.json), or from the run logs committed in `checkpoints/`. Nothing here is estimated unless it says *estimate*.

---

## A. What MathGen actually does today

MathGen is a ~34M-parameter decoder-only Transformer, trained from random initialization on purely synthetic data. It is multi-task, and the prompt prefix selects the task:

| Task | Prompt → completion | Used in production? |
|---|---|---|
| `[GEN]` | `topic:<t> difficulty:<d>` → JSON `{questionText, hints, answerKey}` | **No.** `/generate` defaults to the deterministic template generator (`MATHGEN_SOURCE=generator`). The neural path is opt-in and its answerKey is replaced by the verifier's |
| `[GRADE]` | question[:80] + student and correct answer → JSON verdict | **No.** `/grade` is a deterministic `smart_equal` |
| `[ASSESS]` | answer history → JSON skill profile | **No.** `/assess` is a deterministic formula |

The training data comes from 23 template functions (`src/dataset.py`, `TopicGenerators`). They cover integer arithmetic, fractions, linear and quadratic equations, function evaluation, 2-D vectors, 2×2 matrices, power-rule derivatives and integrals, and polynomial limits. These are mapped onto 10 topics × 3 difficulties, and the templates are shared across topics and difficulties.

The Next.js app (`mathvis-app`) only calls `POST /generate`, so by default it **never uses the neural model**.

## B. What is implemented correctly (strengths)

- A clean, standard modern decoder:
  - pre-norm RMSNorm, RoPE (rotate-half), SwiGLU FFN, weight tying, scaled init;
  - `F.scaled_dot_product_attention(is_causal=True)`, AdamW + warmup-cosine, grad clipping, fp16 AMP, grad accumulation.
- Both checkpoints load **strictly** into the code (`load_state_dict(strict=True)`) and generate valid JSON (smoke test below).
- A **hand-written BPE tokenizer** with lossless round-trip encoding and reserved special/math tokens.
- An **independent verifier** (`src/verifier.py`) that re-derives answers from the question text without trusting the model. It agrees 100% with the generator on all 23 templates (300 draws each, F4). This is a real asset and the right design instinct.
- The served product is safe: model output is never trusted for correctness (verifier-in-the-loop). That is good engineering judgement.
- Everything was trained from scratch on a 4GB laptop GPU. No pretrained weights and no external APIs (verified: no imports of `transformers`, `tiktoken` or `datasets` in the training path; `datasets` is only used by unused GSM8K/MATH loaders).

## C. What evidence of training and evaluation exists

| Artifact | Content | Limitation |
|---|---|---|
| `checkpoints/best.pt` | digit-level tokenizer run (`digit_retrain5.log`): 10 epochs, 9,010 steps, eff. batch 48 (12×4), 47,000 samples (`n_per_topic=1000`), val loss 0.1331 | Data not saved and not reproducible (no seed). Training flags inferred from log text, not recorded |
| `checkpoints/baseline_v2/best.pt` | multi-digit BPE tokenizer run: epoch index 4, step 6,610, val loss 0.1835 | `train_log.jsonl` has two segments (epochs 1–6 then 1–5). Resume or restart is not recoverable from logs |
| `digit_model_eval2.log` | [GEN] self-consistency accuracy 238/300 = 79.3% | Self-chosen questions, T=0.7 sampling, n=30 per topic |
| `baseline_accuracy.log` | 147/180 = 81.7% | Same protocol, n=18 per topic |
| `digit_retrain{1..4}.log`, `train_run.log` | failed or partial runs (OOM, encoding crash) | Undocumented. Kept as a record |

There was **no held-out test set, no seed, no record of run configuration, and no comparison under controlled conditions**.

## D. Claims: which can and cannot currently be supported

| Claim (where) | Verdict | Evidence |
|---|---|---|
| "~5M parameters, d=256" (README) | ❌ **False** | 34.23M total / 33.56M non-embedding, d=512, d_ff=2048 (F1) |
| "Vocab 8,192" (README) / "16K" (tokenizer docstring) | ❌ **False** | 1,299 (current) / 6,031 (baseline_v2). Merging stops early because words with frequency below 2 are excluded |
| "Trained from scratch, no external APIs / pretrained weights" | ✅ Supported | Code inspection |
| "Model learned the output format" | ✅ Supported (weakly) | 0 bad-JSON outputs in 480 logged generations; smoke test 3/3 valid JSON |
| "79.3% / 81.7% answer accuracy" | ⚠ True as measured, **not generalization** | Self-chosen operands from tiny instance spaces; 48.8% of val samples are exact copies of training samples (F5, F6) |
| "Digit-level tokenization is the #1 fix for accuracy" (`tokenizer.py:118-121`) | ❌ **Unsupported** | 79.3% vs 81.7%, two-proportion p = 0.53, plus confounds in data size, epochs and eval n (F7) |
| "Generalizes / no overfitting" (train 0.013 vs val 0.133 suggests a 10× gap) | ❌ The gap is a **logging artifact** | Corrected train 0.1328 ≈ val 0.1331 (F2). Val is also mostly duplicates (F5) |
| "Verifier is the single source of truth" | ⚠ Only in-distribution | Wrong ground truth on non-integer linear solutions; silent `None` on negatives (F4) |

## E. Findings with measured evidence

**F1 — README architecture claims are false.** `MathGenConfig` defaults are d=512, d_ff=2048, 8 layers, 8 heads, ctx 256. Measured: current 34.23M (vocab 1,299); baseline_v2 36.65M (vocab 6,031); non-embedding 33.56M in both.

**F2 — Train-loss logging bug.** `t_loss = epoch_loss / global_step` divides one epoch's loss sum by a step count that keeps growing across epochs (`src/train.py:184,214`). Corrected values for the final run:

| epoch | 1 | 2 | 5 | 10 |
|---|---|---|---|---|
| reported train | 1.607 | 0.099 | 0.031 | 0.013 |
| **corrected train** | 1.607 | 0.198 | 0.154 | **0.133** |
| val | 0.213 | 0.175 | 0.147 | **0.133** |

The tok/s metric has the same bug: 23.8k at the end of epoch 1 vs "241.4k" at epoch 10. Real throughput was ~24k tok/s, and that figure includes padding tokens.

**F3 — Subtraction generator bug.** `a = max(a,b); b = min(a,b)` uses the already-updated `a`. **51.3%** of subtraction items are `x − x = 0` (100k draws); a correct generator gives 3.4%. Training data and the deployed `/generate` endpoint are both affected.

**F4 — Verifier gaps off-distribution.** It agrees 100% with the generator, but on inputs a model can produce:

| question | true | verifier |
|---|---|---|
| Solve for x: 7x + 11 = 72 | 61/7 | **8** (floor division → wrong ground truth) |
| Solve for x: 4x + 3 = 12 | 9/4 | **2** |
| Solve for x: −2x + 3 = 7 / Compute −3 + 5. / ‖[−3,4]‖ / u·v with negatives | −2 / 2 / 5 / 5 | `None` (regex `\d+`) |

So model-side accuracy can be **scored against a wrong answer**, and OOD items with negatives would be silently unverifiable.

**F5 — Instance spaces and leakage.** Unique question counts (60k draws per template; "exhaustive" = saturated):

- **9 tiny templates (≤150):** limit_factor 5, integral_power 12, deriv_power 20, integral_definite 27, limit_infinity 50, deriv_at_point 80, algebra_quadratic 91, linear_transform 144, vec_magnitude 144.
- **6 small (≈500–1,250):** arith_pemdas 512, vec_angle 625, limit_substitution 864, deriv_polynomial 1,080, algebra_linear 1,120, arith_fractions 1,225.
- **8 large:** arith_integer ≥2.1K, func_evaluate ≥4.5K, mat_transpose / mat_det ≥6.5K, vec_dot ≥19.6K, mat_scalar ≥23.6K, mat_mul / mat_add ≥60K.

Re-creating the final run's recipe (47,000 samples, 92/8 row-level split): **48.8%** of val [GEN] samples are character-identical to a training sample, and **58.7%** of val questions appear in training. The 100% integrals score rests on two templates with only 12 and 27 unique instances, each seen hundreds of times in training. The derivative and limit topics mix tiny templates (5–80 instances) with ~1K-instance ones.

**F6 — The task and metric can't support generalization claims.** The model chooses its own question (sampling at T=0.7), so the evaluator controls neither operands nor templates. `eval_accuracy.py` also leaves bad-JSON and unparseable outputs out of the denominator. Both counts were 0 in the two logged runs, so those reported numbers aren't inflated, but the metric is biased in general.

**F7 — The tokenizer comparison is within noise and confounded.** Digit 79.3% (Wilson95 [74.4, 83.5]) vs BPE 81.7% ([75.4, 86.6]), p = 0.53. The runs differ in tokenizer, data size (47K vs 92K samples), epochs, step count and eval n.

**F8 — Tokenizer and format confounds for any future tokenizer study:**
- `json.dumps` defaults to `ensure_ascii=True`, so **84.3%** of real legacy [GEN] samples (2,100 samples, all topic × difficulty slots) contain `\uXXXX` escapes (e.g. `c\u2081\u2081 = 8\u00b75`). There are 4.0 per sample on average. Each one costs about 6 tokens, 4 of them fake "digits".
- `str.isdigit()` is True for `²³₁₂`, so the "digit-level" rule isn't [0-9].
- Length per sample: digit-BPE mean 109 tokens (p95 182, max 191); multi-digit BPE mean 74 (max 123); characters mean 246 (p95 339, max 346). In a char-level arm, **45.0%** of samples would be silently truncated at ctx 256 (`MathDataset` truncates without a warning).

**F9 — No determinism anywhere.** Zero seed calls in `src/*.py`. Two unseeded dataset builds give different sha256 (`0e851b88…` vs `2e1cf1ef…` in the recorded run; they differ on every run). Resume restores weights only (no optimizer, scaler or RNG). A catch-OOM-and-continue path drops micro-batches silently. All runs append to one `train_log.jsonl` with no run ID.

## F. Threats to experimental validity (ranked)

1. **Memorization mistaken for competence** (F5): row-level split, tiny instance spaces, duplicates in val, GRADE re-exposing answers.
2. **No control over the test distribution** (F6): self-generated questions make OOD and held-out evaluation impossible.
3. **Ground-truth errors off-distribution** (F4): the verifier can mark wrong answers right and right answers wrong.
4. **Corrupted training data** (F3): about 17% of arithmetic-integer items teach "x − x = 0".
5. **Confounded comparisons** (F7, F8): tokenizer arms also differ in escapes, pre-tokenization, data, epochs and eval n.
6. **Misleading diagnostics** (F2): loss curves suggest overfitting that doesn't exist; throughput numbers are wrong.
7. **No variance estimate**: a single run per configuration and no seeds, so no difference can be separated from noise.
8. **Sampling noise in eval**: T=0.7 sampling adds avoidable variance. Greedy decoding is needed for primary metrics.

## G. Reproducibility risks

- The datasets behind both checkpoints **can't be regenerated** (no seed, nothing saved).
- Training flags weren't recorded (batch 12 × accum 4 and `n_per_topic=1000` were inferred from log text).
- The original venv (`C:\Users\hp\Desktop\ai-edu\mathvis-model\.venv`) is gone; dependency versions are unknown. `requirements.txt` gives only lower bounds, lists unused packages (`tiktoken`, `datasets`), and **doesn't declare `sympy`**, which `src/verifier.py` imports. It only gets installed because torch depends on it.
- Python 3.14 on this machine needs a WMI monkey-patch before `import torch` (`src/server.py:14-27`).
- `src/server.py:57-58` says CUDA inference throws `cudaErrorUnknown`. **Re-tested:** with torch 2.11.0+cu128 and driver 592.82, legacy `best.pt` generates fine on CUDA (79 tokens), and bf16 is supported. The comment is stale; CPU serving stays as is.
- Checkpoints (137–147MB) exceed GitHub's file limit; they're tracked by sha256 instead.

## H. Missing information

- The exact command lines, seeds, and wall-clock per run. File timestamps put the final run at ≤ ~80 min, consistent with 110.7M padded tokens at ~24k tok/s ≈ 78 min.
- Whether the second baseline_v2 log segment was a resume or a fresh run.
- Library versions used for training.
- Any measurement of the model on problems it didn't generate itself.

---

## Required fixes, in priority order

| P | Fix | Why | Where it lands |
|---|---|---|---|
| P0 | Correct README claims (params, d, vocab, what the app actually uses) | Credibility: anyone checking the repo finds the mismatch | M10 (rewrite with verified numbers); interim note now |
| P0 | Fix the subtraction generator; fix verifier division and negative-number support; add timeouts | Ground truth must be right before any accuracy is measured | M4 (new `mathgen/data`, `mathgen/eval/oracle`) |
| P0 | Instance-keyed, hash-assigned splits with dedup; held-out test sets; GRADE/ASSESS out of the research mix | Removes leakage | M4 |
| P0 | Seed everything; record config, git hash, env, data/tokenizer checksums per run; full resume state | Reproducibility | M5 |
| P1 | Correct loss and throughput logging; per-run directories | Correct diagnostics | M5 (and in the legacy-repro branch, as a logging-only change) |
| P1 | `ensure_ascii=False`; ASCII-only digit rule; shared pre-tokenizer; length guard instead of silent truncation | Clean tokenizer comparison | M4–M5 |
| P1 | Greedy, batched, KV-cached decoding; failures count as wrong | Valid, fast eval | M5 |
| P2 | Remove the OOM-skip path; bf16; dynamic padding; completion-only, per-sequence-normalized loss | Fair matched-step comparisons | M5 |

**The smallest changes that make the existing baseline reproducible.** These go in a `legacy-repro` branch off `legacy-v0`, logging and seeding only, with training behaviour unchanged:
1. a `--seed` flag seeding `random` / `torch` / `cuda`;
2. fix the train-loss and tok/s formulas;
3. write `args`, git hash, `pip freeze` and the dataset sha256 to a per-run directory;
4. stop appending to a shared log.

The generator bug F3 is **kept** in that branch on purpose: Baseline A has to reproduce the original pipeline, not an improved one.

## Proposed research question and competing hypotheses

> In ~5M-parameter decoder-only Transformers trained from scratch on multi-family structured math (arithmetic, 2×2 linear algebra, polynomial calculus; JSON output), how do number/subword tokenization and positional-encoding/FFN choices affect answer correctness, and how much in-distribution accuracy is memorization of seen instances rather than generalization to unseen instances, held-out operand blocks, and larger operands?

Competing hypotheses, deliberately held side by side:
- **H1** Digit-split numbers beat multi-digit BPE at matched non-embedding params and steps, *vs.* **H1′** no detectable difference once escapes and pre-tokenization are equalized. The legacy evidence (F7) is consistent with both.
- **H3** Most in-distribution accuracy is memorization (a large seen-vs-unseen gap that shrinks with coverage), *vs.* **H3′** small models generalize within operand ranges, with near-equal seen and unseen accuracy for the large-space templates.
- **H4** Every arm collapses on larger operands, *vs.* **H4′** digit tokenization plus RoPE keeps meaningful accuracy on same-length OOD values. The literature to be verified in M2 suggests H4 for longer operands.
- **H6** RoPE/SwiGLU help, *vs.* **H6′** no detectable effect at this scale. A null result is acceptable and will be reported as one.
- **Control** Hints-first (scratchpad) vs answer-first ordering, to separate arithmetic-in-hints from answer production.

These are provisional until the literature review (M2) and get frozen in a pre-registration (M3).

## Recommended experiments

See the approved plan: tokenizer arms (char / digit / bpe-num on a shared pre-tokenizer), seen/unseen/block/OOD splits evaluated on every run, a data-efficiency curve, architecture ablations, and the answer-first control. That's ≈52 core runs at ~5M params with ≥3 seeds (5 for the reference). A seed-variance pilot comes first, and the matrix is cut by arms, not seeds.

## Compute estimates (actual hardware)

**Hardware:** Intel i5-11400H (6C/12T), 7.6GB RAM, NVIDIA RTX 3050 Laptop 4GB (driver 592.82), 41.9GB free disk, Windows 11, Python 3.14.2.

| Workload | Basis | Estimate |
|---|---|---|
| Legacy 34M run (P1 reproduction) | Measured: 9,010 steps × 48 × 256 = 110.7M padded tokens at ~24k tok/s | ~78 min |
| 5M REF run | *Estimate* (to be measured in P0): ~6× fewer non-embedding FLOPs/token than 34M and dynamic padding; 15K steps × 64 × ~100 tokens | 13–25 min each |
| Core grid (52 runs incl. LR sweep and pilots) | Plan estimate × 1.4 for thermal throttling | ≈18 GPU-h ≈ 25–30 wall-clock hours |
| Eval per run | Current decoding is batch-1 with no KV cache: 2.3–5.2 s per legacy generation on CPU (smoke test) → >30 min for 8K items | Needs batched KV-cached decoding (P1 fix) |

## Next three concrete actions

1. **M2 — Literature review and gap** (no code).
   - Output: `research/LITERATURE_REVIEW.md`, `RELATED_WORK_TABLE.md`, `RESEARCH_GAP.md`, `REFERENCES.bib`.
   - Acceptance: every entry has a verified arXiv ID, DOI or official URL; the gap statement is checked against the three closest papers; the RQ is reframed if it's already answered.
2. **M3 — Pre-registration**.
   - Output: `research/RESEARCH_DESIGN.md` (splits, block regions, metrics, decision rule, run matrix) and git tag `prereg-v1`.
   - Acceptance: every hypothesis has a control, a primary metric, an analysis method and a decision criterion, and **you approve it (G2)**.
3. **M4 — Data v1 and oracle integrity**.
   - Command: `python scripts/build_data.py --config configs/base.yaml && pytest -q tests/test_data.py tests/test_oracle.py`.
   - Expected: manifests in `data/manifests/`; tests 1–5 of the plan green.
   - Acceptance: two builds give identical sha256; generator and oracle agree on 100% of items in every split; subtraction a==b ≈ 3.4%; no instance key shared between train and any eval split.

---

## Appendix — commands executed and outcomes

| Command | Outcome |
|---|---|
| `git init`, commit `ea6caf9`, `git tag legacy-v0` | Legacy snapshot frozen (37 files; `*.pt` excluded via `.gitignore`) |
| `sha256sum checkpoints/**/*.pt …` | `research/artifacts/CHECKPOINTS.sha256` |
| `python -m venv .venv` + `pip install torch --index-url …/whl/cu128` + deps; `pip freeze > requirements.lock` | torch 2.11.0+cu128, CUDA available, bf16 supported; 39 pinned packages |
| `.venv/Scripts/python.exe -B research/audit_scripts/audit_probe.py` | All F1–F9 probes and the smoke test ran; `artifacts/audit_results.json` and `artifacts/audit_probe_stdout.txt` written. A first debug run under system torch 2.12.0+cpu gave identical F1–F7/F9 numbers |
| Smoke: legacy `best.pt`, 3 [GEN] prompts, CPU, seed 0 | 3/3 valid JSON; 2/3 verified-correct; 67–120 new tokens in 2.4–5.1 s (batch 1, no KV cache) |
| CUDA check (legacy `best.pt` on GPU) | Generates normally; the `cudaErrorUnknown` note is stale |
| `.venv/Scripts/python.exe -B research/audit_scripts/server_smoke.py` | `/health` ok; `/generate` valid for **30/30** topic × difficulty pairs (default generator source) and 3/3 on the opt-in model path. App contract intact |
