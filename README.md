# MathGen: memorization vs. generalization in small Transformers trained from scratch on math

MathGen started as the problem-generation model behind **MathVis**, a math-learning web app. I built it as a decoder-only Transformer trained from random initialization on synthetic math, and it seemed to score about 80%. When I audited my own pipeline, that number turned out to be mostly memorization: **48.8% of the validation samples were exact copies of training samples**.

This repository holds the original model and the audit. It also holds the **pre-registered, controlled study** that replaced the original evaluation. The study asks which design choices (number tokenization, positional encoding, FFN type, answer position, amount of training data) change how well a ~5M-parameter model solves **instances it has never seen**, compared with the ones it has.

> **Status (2026-10-19): work in progress. The main experiment has not been run yet.**
> - Done: audit, literature review, pre-registration (tag `prereg-v1`), data v1, the full training and evaluation code (316 tests passing), and pilots P0, P2, P3 and P4.
> - Not done:
>   - P1, the reproduction of the legacy model: two attempts were stopped, see [Pilot results](#4-pilot-results-so-far).
>   - The 36-run core grid, which produces every confirmatory result. It is waiting for the go/no-go review (gate G3).
> - **No hypothesis has been tested yet.** Every number below is a pilot measurement of the reference arm only.

---

## Contents

1. [Why this project exists: the audit](#1-why-this-project-exists-the-audit)
2. [Research questions](#2-research-questions)
3. [Study design (pre-registered)](#3-study-design-pre-registered)
4. [Pilot results so far](#4-pilot-results-so-far)
5. [What happens next](#5-what-happens-next)
6. [Repository layout](#6-repository-layout)
7. [How to reproduce](#7-how-to-reproduce)
8. [Legacy model and the MathVis server](#8-legacy-model-and-the-mathvis-server)
9. [Limitations](#9-limitations)
10. [Research documents](#10-research-documents)

---

## 1. Why this project exists: the audit

The original MathGen ("legacy", tag `legacy-v0`) is a **34.2M-parameter** decoder-only Transformer:
- 8 layers, d = 512, d_ff = 2048;
- pre-norm RMSNorm, RoPE, SwiGLU and tied embeddings;
- a hand-written BPE tokenizer;
- trained from scratch on a 4 GB laptop GPU, on synthetic data from **23 template generators**: integer arithmetic, fractions, linear and quadratic equations, function evaluation, 2-D vectors, 2×2 matrices, power-rule derivatives and integrals, and limits.

It reported 79.3% and 81.7% answer accuracy. The audit ([`research/REPOSITORY_AUDIT.md`](research/REPOSITORY_AUDIT.md)) re-measured everything with a seeded probe script and found that those numbers do not show generalization:

| # | Finding | Measured evidence |
|---|---|---|
| F1 | The old README's architecture claims were wrong | 34.23M parameters (not ~5M); vocab 1,299 (not 8,192) |
| F2 | Train-loss logging bug | The apparent 10× train/val gap was an artifact: corrected train loss 0.1328 vs. val 0.1331 |
| F3 | Subtraction generator bug | 51.3% of subtraction items were `x − x = 0` (a correct generator gives 3.4%) |
| F4 | Verifier gaps outside the training distribution | Wrong ground truth on non-integer linear solutions; silent `None` on some negative inputs |
| F5 | Tiny instance spaces and train/val leakage | **48.8%** of val samples were character-identical to a training sample; 58.7% of val questions appeared in training. Some templates have only 5–27 distinct instances |
| F6 | The metric could not measure generalization | The model chose its own questions (sampling at T = 0.7), and unparseable outputs were left out of the denominator |
| F7 | "Digit tokenization helps" was not supported | 79.3% vs. 81.7%, p = 0.53, with data size, epochs and eval size all different between runs |
| F8 | Tokenizer and format confounds | e.g. `str.isdigit()` treats `²` and `₁` as digits; `\uXXXX` escapes inflate lengths |
| F9 | No determinism | No seeds anywhere; resume restored weights only; an OOM-skip path dropped batches silently |

Things that were right and are kept: the modern decoder architecture, a lossless tokenizer, and an **independent verifier** that re-derives each answer from the question text (100% agreement with the generator on all 23 templates).

---

## 2. Research questions

From [`research/RESEARCH_DESIGN.md`](research/RESEARCH_DESIGN.md) §1. The study uses ~5M-parameter decoder-only Transformers trained from scratch on 23 structured-math templates with JSON output.

- **RQ-A (tokenization):** Single-digit number tokenization is known to beat multi-digit tokenization for integer arithmetic. Does that advantage hold for structured families (fractions, vectors/matrices, calculus)? How does it change with the number of distinct training examples? Is it an input-side or an output-side effect?
- **RQ-B (memorization):** How large is the accuracy gap between instances the model has **seen** in training and instances it has **not**, and how does that gap depend on how much of the instance space the training data covers?
- **RQ-C (generalization):** Which effects survive on held-out value blocks *inside* the trained range, and which collapse on operands *outside* it?
- **Secondary:** positional encoding (RoPE / learned / none), FFN type (SwiGLU / GELU), and answer position (hints first / answer first).

---

## 3. Study design (pre-registered)

The design was frozen as git tag **`prereg-v1`** before any study run. Any later change is logged as a deviation, with date, reason and expected impact, in [`research/decisions.md`](research/decisions.md). Values that can only be measured (step budget, learning rates, smallest effect of interest) are fixed by **rules** written down in advance, then applied in the pilots.

### 3.1 Task

- **Prompt:** `[SOLVE] <questionText> <SEP><JSON>`
- **Target:** `{"hints": [...], "answerKey": ...}<EOS>`. "Hints first" is the default; "answer first" is a control.
- **Loss:** completion tokens only.
- **Decoding:** greedy. Bad JSON, a missing key, truncation or an oracle timeout all count as **wrong**; nothing is dropped from the denominator.

### 3.2 Data v1

Built deterministically by `scripts/build_data.py` from [`configs/base.yaml`](configs/base.yaml). Every item is checked against an independent oracle as it is written. See [`data/README.md`](data/README.md) and [`data/dataset_card.md`](data/dataset_card.md).

- **G-set (14 templates, instance space ≥ 500):** split into train / val / test pools by a hash of the instance key: `sha256(key) mod 1000`, with train < 700, val 700–749 and test ≥ 750. The model can never train on a test instance.
- **M-set (9 templates, instance space ≤ 150):** too small to split, so they are trained on in full and used only as a memorization and format control.
- **Block regions (7):** value regions *inside* the training range that are removed from training entirely. Examples: multiplication with both operands in [20, 30], linear equations with solution x = 7, 2×2 determinants with all entries ≥ 7.
- **Out-of-distribution (OOD) operands:** OOD-same uses values outside the training range with the same number of digits; OOD-longer uses one extra digit.

| Split | Items | Used for |
|---|---:|---|
| `test_id_unseen` | 2,672 | **primary endpoint**: unseen instances inside the training range |
| `val` | 644 | step-budget and learning-rate selection only, never for claims |
| `block` | 1,069 | held-out value blocks inside the range (H4a) |
| `ood_same` | 1,248 | out-of-range operands, same digit count (H4b) |
| `ood_longer` | 2,189 | operands with one more digit (H4b, length generalization) |
| `train-recall` | per run | instances that run actually trained on (memorization, H3) |
| `train/seed{0..4}` | 128,000 each | nested training sets: the first N lines are the size-N set |

Every arm with the same seed sees the **identical** training set in the same order, so all comparisons are **paired**.

### 3.3 Arms

| Arm | What changes compared with REF |
|---|---|
| **REF** | T-digit tokenizer (every number token is a single digit), RoPE, SwiGLU, hints first. d = 256, 6 layers, 4 heads, ≈ 4.7M non-embedding parameters, context 256 |
| `char` | character-level tokenizer |
| `bpe_num` | BPE merges allowed inside numbers; identical to T-digit on all non-number text |
| `dec_db`, `dec_bd` | digit input / BPE output and the reverse, which separates input-side from output-side tokenization effects |
| `learned`, `none` | learned positional embeddings / no positional encoding (NoPE) |
| `gelu` | GELU FFN, width-matched so non-embedding parameters stay within 1% |
| `answer_first` | the answer is written before the hints |
| REF at N ∈ {2K, 8K, 32K} | less training data, which varies instance-space coverage (128K is REF itself) |

**Training (identical for every arm):**
- AdamW (β = 0.9 / 0.95, weight decay 0.1), cosine schedule with 5% warmup, gradient clipping 1.0, batch 64, bf16 autocast.
- Deterministic GPU kernels.
- Full resume state: model, optimizer, scheduler and every RNG.
- The **final** checkpoint is evaluated; nothing is picked by test or val score.

### 3.4 Hypotheses and decision rule

There are twelve confirmatory contrasts, C1–C12. Examples:
- **C1:** digit vs. BPE tokenization on integer templates (a replication).
- **C2:** the same contrast on structured families.
- **C3:** input-side vs. output-side tokenization.
- **C10:** whether the seen-vs-unseen gap shrinks with more training data.
- **C12:** whether accuracy on held-out blocks matches accuracy on unseen in-range instances.

The full table is in [`RESEARCH_DESIGN.md` §6](research/RESEARCH_DESIGN.md#6-hypotheses-and-confirmatory-contrasts).

An effect is **claimed** only if all three hold:
1. it has the same sign in **every** paired seed;
2. its Holm-adjusted p < 0.05 (seed-stratified Cochran–Mantel–Haenszel test, or a hierarchical bootstrap for the gap contrasts);
3. its size is larger than the **smallest effect size of interest (SESOI) = 2.00 pp**.

Equivalence is claimed only if the 90% CI lies inside ±SESOI. Everything else is reported as **inconclusive**.

---

## 4. Pilot results so far

Full narrative: [`research/PILOT_REPORT.md`](research/PILOT_REPORT.md). The tables are generated from the run directories by `scripts/pilot_report.py` into [`research/artifacts/pilot_tables.md`](research/artifacts/pilot_tables.md), so no number is typed in by hand.

Hardware for every run: RTX 3050 Laptop GPU (4 GB VRAM), 7.6 GB system RAM, torch 2.11.0 + CUDA 12.8, bf16.

| Pilot | What it fixes | Result |
|---|---|---|
| **P0** | environment, speed, determinism | Deterministic mode reproduces bitwise and costs ≈ 7% throughput. Battery power is ~6× slower than AC (0.352 vs. 0.059 s/step). Peak VRAM is 796 MB, so system RAM, not the GPU, is the constraint |
| **P2** | step budget S | **S = 27,500 steps.** Val accuracy was still rising at 30K, so arms are compared at a fixed compute budget, not at convergence |
| **P3** | learning rate per arm | 5e-4 for REF, char, bpe_num and none; 1e-3 for learned and gelu. Four arms chose the lowest grid value; this is kept as pre-registered and reported as a limitation |
| **P4** | seed variance and SESOI | Seed SD on the primary endpoint 0.84 pp, so **SESOI = max(2 pp, 2 × 0.84) = 2.00 pp**. The "cut arms" gate was not triggered |
| **P1** | legacy reproduction (Baseline A) | **Not complete.** Attempt 1 crashed on a Windows console-encoding error (fixed in `72caf6c`). Attempt 2 was stopped by a low-memory guard at step ~875 of 901 in epoch 1. P1 is provenance only and feeds no decision |

**REF over 5 seeds at S = 27,500.** This is a *descriptive* pilot observation of one arm, not a tested result:

| Split | Mean accuracy | Seed SD (pp) |
|---|---:|---:|
| train-recall (seen instances) | 99.7% | 0.23 |
| test-ID-unseen (**primary**) | 92.0% | 0.84 |
| val | 92.7% | 0.87 |
| block (held-out values inside the range) | 39.4% | 3.07 |
| OOD-same (outside the range, same length) | 19.2% | 2.93 |
| OOD-longer (one more digit) | 0.3% | 0.13 |

The pattern is clear: near-perfect recall of seen instances, high accuracy on unseen instances that resemble training data, and a steep drop on held-out value regions and longer operands. Whether the block gap is real is exactly what confirmatory contrast C12 will test. These pooled numbers are not matched by template.

Each REF seed trains in ~0.48 h; evaluation on all 11,195 items takes ~63 s.

Bugs found during the pilots were fixed test-first, and none affected a reported number. The list is in `PILOT_REPORT.md` §8.

---

## 5. What happens next

1. **G3 review:** approve the core grid (36 runs, ≈ 17.9 GPU-hours, measured) and freeze SESOI = 2.00 pp. The open decisions are listed in `PILOT_REPORT.md` §10.
2. **P1:** rerun the legacy reproduction from a separate terminal (≈ 2 h; it cannot resume).
3. **Core grid**, in the pre-registered order so the central contrasts finish first:
   C-tok → C-dec → C-arch → C-af → C-data → C-dtok (≈ 20 h in total on AC power). A core-grid orchestrator still needs to be written; `scripts/pilots.py` covers P1–P4 only.
4. **Analysis and technical report:** confirmatory contrasts C1–C12 with per-seed values and hierarchical-bootstrap CIs, plus the exploratory analyses listed in the design.

---

## 6. Repository layout

```
configs/
  base.yaml              data v1 build config (pools, split sizes, operand ranges)
  arms/*.yaml            one file per arm (tokenizer, architecture, answer order, LR)
data/
  README.md, dataset_card.md
  manifests/             sha256 of every built file + config/code hashes (committed)
  tokenizers/            the three fitted tokenizers (char, digit, bpe_num)
  eval/, fit/, train/    built JSONL files (gitignored, regenerable)
src/mathgen/             the study code
  data/                  templates, operand ranges, hash splits, formats, build
  tokenizers/            shared pre-tokenizer + char / digit / bpe_num / decoupled arms
  model/                 decoder (RoPE / learned / none; SwiGLU / GELU); also loads the legacy checkpoint
  training/              resumable loop, optimizer and schedule, batching
  eval/                  oracle (independent of the generator), greedy decoding, metrics
  compat.py              Windows import workaround and deterministic-cuBLAS setup
src/*.py                 legacy MathGen code (model, tokenizer, dataset, verifier, server)
scripts/                 build_data, fit_tokenizers, train, eval, pilots, pilot_report, baseline_a
tests/                   pytest suite (316 passed, 10 skipped on 2026-10-19)
research/                audit, literature review, design, pilot report, decisions, lab log
checkpoints/             legacy run logs and tokenizers (weights are not in git, see below)
docs/diagrams/           architecture and training-loop diagrams
graph-mathvis-model/     generated HTML architecture pages
```

Model weights (`*.pt`) and run outputs (`runs/`) are **not committed**. Their sha256 hashes are recorded in [`research/artifacts/CHECKPOINTS.sha256`](research/artifacts/CHECKPOINTS.sha256), and every study run writes its config, git hash, `pip freeze`, dataset and tokenizer checksums into `runs/<exp_id>/`.

---

## 7. How to reproduce

Requires Python ≥ 3.12. A CUDA GPU is strongly recommended; everything was run on a 4 GB card. On Windows, use `.venv\Scripts\python.exe` instead of `python` in the commands below. All commands run from the repository root.

**Setup**

```bash
python -m venv .venv
source .venv/bin/activate                     # Windows: .venv\Scripts\activate
pip install -r requirements.lock --extra-index-url https://download.pytorch.org/whl/cu128
```

`requirements.lock` pins the exact versions used, including `torch==2.11.0+cu128`. `requirements.txt` holds the looser legacy requirements.

**1. Build the data (~45 s) and fit the tokenizers**

```bash
python scripts/build_data.py --config configs/base.yaml
python scripts/fit_tokenizers.py
pytest -q tests/test_data.py::test_built_data_matches_committed_manifest   # checks your build against the committed manifest
```

**2. Train and evaluate one run**

```bash
python scripts/train.py --config configs/arms/ref.yaml --seed 0 --steps 27500 --then-eval
# or evaluate a finished run separately:
python scripts/eval.py --run runs/<exp_id>
```

- Runs resume automatically from `runs/<exp_id>/ckpt.pt`.
- Reusing an `exp_id` with a different config is refused.
- Useful flags: `--n-train`, `--lr`, `--eval-every`, `--exp-id`, `--device`.

**3. Pilots and the pilot report**

```bash
python scripts/pilots.py p2 p3 p4        # each stage applies its pre-registered rule
python scripts/pilots.py p1              # legacy reproduction; needs the legacy-repro worktree (below)
python scripts/pilot_report.py           # regenerates research/artifacts/pilot_tables.md
```

P1 runs the legacy code from a `legacy-repro` worktree:

```bash
git worktree add .worktrees/legacy-repro legacy-repro
```

**4. Tests**

```bash
pytest -q        # 316 passed, 10 skipped (~90 s)
```

**Practical notes from the pilots**
- Train on AC power: battery mode was ~6× slower.
- On a machine with ~8 GB RAM, close memory-heavy apps. Every study run resumes from its last checkpoint (every 2.5K steps); the legacy P1 run cannot resume.

---

## 8. Legacy model and the MathVis server

The original code is still in `src/*.py`, unchanged except for the one-line F3 subtraction fix in the served generator (decision DV-003). The exact original snapshot is tag **`legacy-v0`**. Branch **`legacy-repro`** adds only seeding and logging fixes for the Baseline A reproduction and keeps F3 on purpose.

`src/server.py` is the FastAPI service the MathVis app calls (port 8787):

| Endpoint | What it does |
|---|---|
| `POST /generate` | Returns a problem as JSON `{questionText, hints, answerKey}`. By default it uses the deterministic template generator. `MATHGEN_SOURCE=model` switches to the neural model, but its answerKey is always replaced by the verifier's |
| `POST /grade` | Checks a student's answer with the deterministic `smart_equal` |
| `POST /assess` | Builds a skill profile with a deterministic formula |
| `GET /health` | Health check |

```bash
python src/server.py                     # http://localhost:8787
```

The neural path needs a locally trained legacy checkpoint in `checkpoints/`; weights are not distributed in this repository.

---

## 9. Limitations

From the design (§10) and the pilots (`PILOT_REPORT.md` §9):

- **Scope.** The results will say nothing about pretrained LLMs or models much larger than ~5M parameters, nothing about natural-language word problems, and nothing about length generalization beyond one extra digit.
- **Instances, not templates.** Instances are held out but templates are not, so there is no compositional (template-level) generalization test.
- **Small numbers.** Most numbers have 1–3 digits, which limits where tokenizers can differ. A per-item "number-token divergence" covariate and a zero-divergence negative control address this.
- **Fixed compute budget.** The step budget comes from a val curve that was still rising, so slower-learning arms are compared before they finish learning.
- **Learning-rate grid edge.** Four of six arms chose the lowest learning rate in the grid.
- **Noisy endpoints.** Block and OOD-same accuracy vary ~3 pp across seeds, more than the SESOI, so C12 may end inconclusive with 3 seeds.
- **Floor effect.** C7 (NoPE vs. RoPE on OOD-longer) sits near a floor: REF scores 0.3%, so an effect is claimable only if NoPE reaches roughly 2.3%.

---

## 10. Research documents

| Document | Contents |
|---|---|
| [`research/REPOSITORY_AUDIT.md`](research/REPOSITORY_AUDIT.md) | M1: what the legacy model does, findings F1–F9, reproducibility plan |
| [`research/LITERATURE_REVIEW.md`](research/LITERATURE_REVIEW.md), [`RELATED_WORK_TABLE.md`](research/RELATED_WORK_TABLE.md), [`REFERENCES.bib`](research/REFERENCES.bib) | M2: related work on number tokenization, memorization and length generalization (references verified by script) |
| [`research/RESEARCH_GAP.md`](research/RESEARCH_GAP.md) | M2: what is already known and what this study adds |
| [`research/RESEARCH_DESIGN.md`](research/RESEARCH_DESIGN.md) | M3: the pre-registration (tag `prereg-v1`) |
| [`research/PILOT_REPORT.md`](research/PILOT_REPORT.md) | M6: pilots P0–P4, costs, decisions requested at G3 |
| [`research/decisions.md`](research/decisions.md) | every design decision and deviation, with date and reason |
| [`research/lab_log.md`](research/lab_log.md) | day-by-day log, including interruptions and fixes |

---

**Author:** Asadbek Xodjayev (Webster University) · [github.com/asadbekXodjayev](https://github.com/asadbekXodjayev)
