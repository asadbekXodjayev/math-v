# MathGen — Research Design and Pre-registration (Milestone 3)

**Status:** frozen as git tag `prereg-v1`. Approved by the user via the 2026-10-17 decision ("MathGen to G3", "Adopt RQ v2 + all"; see `lab_log.md`).
**After the tag:** any change is a *deviation*. It is logged in [`decisions.md`](decisions.md) with date, reason and expected impact, and reported in the paper.
**Evidence base:** [`REPOSITORY_AUDIT.md`](REPOSITORY_AUDIT.md) (M1), [`LITERATURE_REVIEW.md`](LITERATURE_REVIEW.md) and [`RESEARCH_GAP.md`](RESEARCH_GAP.md) (M2).
**What this freezes:** the questions, hypotheses, data splits, arms, endpoints, statistics and run matrix.

Some values (context length, step budget, learning rates) can only be measured in the pilots. For those, this document freezes the **rule** that sets them, not the value.

---

## 1. Research question (RQ v2)

> In ~5M-parameter decoder-only Transformers trained from scratch on a multi-family structured-math distribution (23 templates: integer arithmetic, fractions, equations, function evaluation, 2-D vectors, 2×2 matrices, power-rule calculus, limits; JSON output):
> **RQ-A:** does the single-digit number-tokenization advantage established for integer arithmetic hold for structured families (fractions, vectors/matrices, calculus), how does it change with the number of distinct training examples, and is it an output-side effect?
> **RQ-B:** how large is the accuracy gap between *seen* and *unseen* instances, and how does it depend on instance-space coverage?
> **RQ-C:** which effects survive held-out value blocks within the trained range, and which collapse on out-of-range operands?
> **Secondary:** positional encoding (RoPE / learned / none), FFN (SwiGLU / GELU), and answer position (hints-first / answer-first).

---

## 2. Task and format

- **Task:** conditional solving.
  - Prompt: `[SOLVE] <questionText> <SEP><JSON>`.
  - Target (hints-first, the default): `{"hints": [...], "answerKey": ...}<EOS>`.
  - Serialized with `json.dumps(..., ensure_ascii=False)`, so there are no `\uXXXX` escapes (fixes audit F8).
- **Loss:** completion tokens only, averaged per sequence and then over the batch.
- **Decoding for every metric:** greedy, max new tokens = context length − prompt length. A truncated output counts as wrong.
- **Baseline A (provenance only):**
  - the legacy `[GEN]` self-generation task, through the seeded 34M legacy reproduction (P1);
  - the legacy checkpoint is also scored on the new test items with a forced `[SOLVE]`-style prefix.

  Both are labelled "saw most ID instances; not comparable".

---

## 3. Data v1 (implemented in M4, `mathgen/data/`)

### 3.1 Templates and families
All 23 legacy generators are ported as pure functions `fn(rng, spec) -> Instance`, with explicit operand ranges equal to the legacy ranges (`src/dataset.py` at `legacy-v0`).
- The ground truth comes from the generator.
- An independent oracle (the fixed verifier) must agree on 100% of built items.
- The subtraction bug (F3) is fixed: the operands are ordered (max, min) *before* assignment.

**Sets:**
- **G-set** (14 templates, instance space ≥ 500): used for every hash-split endpoint.
- **M-set** (9 templates, instance space ≤ 150): trained on in full and reported only as a seen-instance memorization and format control. It has no unseen split; their spaces are too small (audit F5).

| Family | G-set templates | M-set templates |
|---|---|---|
| **F-int** (scalar integer arithmetic and evaluation) | arith_integer, arith_pemdas, algebra_linear, func_evaluate, limit_substitution | — |
| **F-frac** (rational / real-valued outputs) | arith_fractions, vec_angle | — |
| **F-mat** (2-D vectors, 2×2 matrices) | vec_dot, mat_add, mat_scalar, mat_mul, mat_det, mat_transpose | vec_magnitude, linear_transform |
| **F-calc** (power-rule calculus, limits) | deriv_polynomial | deriv_power, deriv_at_point, integral_power, integral_definite, limit_factor, limit_infinity, algebra_quadratic* |

\*algebra_quadratic (91 instances) is grouped with the M-set by size.

**"Structured families" for H1b** = F-frac ∪ F-mat ∪ F-calc (G-set only). The M1 audit's measured instance-space sizes define G vs M; M4 re-measures them and logs any reclassification as a deviation.

### 3.2 Instance keys and splits
- **Instance key:** `template|canonical operands`. For example, the quadratic's roots are stored as an unordered pair, and subtraction as (max, min).
- **Hash bucket:** `h = int(sha256(key)[:16], 16) mod 1000`, with these pools:

| Pool | Buckets | Use |
|---|---|---|
| train pool | `h < 700` | training instances are drawn only from here |
| val pool | `700 ≤ h < 750` | LR and budget selection, monitoring; never used for claims |
| test pool | `h ≥ 750` | test-ID-unseen |

- **Block regions** (§3.3) are removed from *every* pool before hashing and live only in the block split.
- M-set templates skip hashing: every instance is in the train pool.

**Evaluation splits.** Each one is built once and checksummed into `data/manifests/`.

| Split | Content | Size | Endpoint |
|---|---|---|---|
| test-ID-unseen | G-set test-pool instances | up to 200 per template (all if fewer) | **primary** |
| train-recall | instances in *that run's* training set (G and M) | up to 200 per template, built per run from its training set | H3 |
| block | the 7 block regions (§3.3) | up to 200 per region | H4a |
| OOD-same | operands outside the ID range, **same digit count** (§3.4) | 100 per applicable template | H4b, H6-pe |
| OOD-longer | operands with **one more digit** (§3.4) | 100 per applicable template | H4b, H6-pe |
| val | val pool | 50 per G-set template | selection only |

**Training sets:**
- For run seed *s*, the training stream draws a template uniformly at random, then an instance uniformly from that template's ID distribution, rejecting it unless it is in the train pool.
- The training set of size N is the first N draws of the stream for seed *s*, so sets for different N are **nested**.
- Every arm with the same seed gets the identical set and order, which makes all arm comparisons **paired**.
- Seeds change initialization, data sampling and order.

**Tokenizers** are fit on a fixed fit sample: 50K draws from the train pool with stream seed 0. Their vocabularies therefore don't change across run seeds.

### 3.3 Block regions (held out of training, within the ID range) — H4a

| ID | Template | Region removed from training |
|---|---|---|
| B1 | arith_integer | op = `*` and a, b ∈ [20, 30] |
| B2 | algebra_linear | solution x = 7 |
| B3 | func_evaluate | evaluation point x = −3 |
| B4 | mat_det | all four entries ≥ 7 |
| B5 | vec_dot | u₀ ∈ [10, 12] and v₀ ∈ [10, 12] |
| B6 | arith_fractions | d₁ = 7 |
| B7 | deriv_polynomial | n = 4 and m = 2 |

### 3.4 OOD operand ranges — H4b

The rule:
- **OOD-same:** each varied operand is drawn outside its ID range, with the same digit count as the ID maximum.
- **OOD-longer:** each varied operand has one more digit than the ID maximum.
- Only the operands listed are varied; the others stay in their ID ranges.

| Template | ID range of the varied operand(s) | OOD-same | OOD-longer |
|---|---|---|---|
| arith_integer | a, b ∈ [2, 30] | a, b ∈ [31, 99] | a, b ∈ [100, 300] |
| arith_pemdas | a, b, c ∈ [2, 9] | — (no unused 1-digit values ≥ 2) | a, b, c ∈ [10, 30] |
| algebra_linear | x ∈ [1, 10] | x ∈ [11, 99] | x ∈ [100, 300] |
| func_evaluate | x ∈ [−4, 4] | x ∈ ±[5, 9] | x ∈ ±[10, 30] |
| limit_substitution | x₀ ∈ [1, 4] | x₀ ∈ [5, 9] | x₀ ∈ [10, 30] |
| arith_fractions | d₁, d₂ ∈ [2, 8] | d₁, d₂ = 9 | d₁, d₂ ∈ [10, 30] |
| vec_angle | entries ∈ [1, 5] | entries ∈ [6, 9] | entries ∈ [10, 30] |
| vec_dot | entries ∈ [1, 12] | entries ∈ [13, 99] | entries ∈ [100, 300] |
| mat_add, mat_mul, mat_det, mat_transpose | entries ∈ [1, 9] | — | entries ∈ [10, 30] |
| mat_scalar | k ∈ [2, 5] | k ∈ [6, 9] | k ∈ [10, 30] |
| deriv_polynomial | a ∈ [2, 5] | a ∈ [6, 9] | a ∈ [10, 30] |
| M-set templates (all 9) | their own coefficient ranges | the same rule where a same-length value exists | the same rule |

The exact samplers live in `configs/base.yaml` (M4). Any range the rule leaves ambiguous is fixed there and logged in `decisions.md` before data is built.

### 3.5 Per-item covariates (recorded for analysis)
For each item:
- template and family;
- instance-space size;
- coverage, i.e. the number of distinct training instances of that template ÷ its space size;
- times the item's own instance was seen in training (train-recall);
- operand magnitude;
- prompt and target length in characters and per arm in tokens;
- number of hint steps;
- **number-token divergence**: whether digit and bpe-num tokenize the item's numbers differently. Items with zero divergence form a built-in negative control for H1.

---

## 4. Arms

### 4.1 Tokenizers (M5, `mathgen/tokenizers/`)
- **Shared pre-tokenizer:** ASCII digit runs `[0-9]+` | runs of other non-space characters | spaces. The same special and JSON-structure tokens exist in every arm. `str.isdigit()` is never used (fixes F8: `²₁` are not digits).
- **T-char:** single characters, no merges.
- **T-digit (REF):** BPE merges only inside non-digit chunks; every number token is a single ASCII digit.
- **T-bpe-num:** T-digit's merge list plus extra merges inside digit runs, fit on the same sample. Non-number text tokenizes identically in T-digit and T-bpe-num.
- **Decoupled (H1c, adopted proposal P-b):** both use the union vocabulary of T-digit and T-bpe-num.
  - **D-in/B-out:** prompt numbers digit-split, completion numbers bpe-merged.
  - **B-in/D-out:** the reverse.
- **Hard guard:** no sequence may exceed the context length. If one does, the build fails; nothing is silently truncated.

### 4.2 Models (M5, `mathgen/model/`)
- **REF:**
  - decoder-only, d = 256, 6 layers, 4 heads;
  - pre-norm RMSNorm, **RoPE**, **SwiGLU** with d_ff = 688;
  - tied embeddings;
  - ≈ 4.7M non-embedding parameters.
- **Variants:**
  - position ∈ {rope, learned, none};
  - FFN ∈ {swiglu(688), gelu(1032)}.
- **Matching:** non-embedding parameter counts within 1% across arms (test 10). Totals are reported. Learned positions are counted as embedding parameters.
- **Context length (rule):** the smallest power of two ≥ the longest T-char sequence over *all* built splits (including OOD-longer), and at least 256. One value for all arms. M4 measures it and logs it.

### 4.3 Training protocol (M5, `mathgen/training/`)
- **Optimizer:**
  - AdamW, β = (0.9, 0.95), weight decay 0.1 (none on norms, biases or embeddings);
  - cosine schedule with 5% warmup to 10% of peak LR;
  - gradient-clip 1.0; batch size 64; bf16 autocast.
- **Removed:** no GradScaler, no OOM-skip path (audit F9).
- **Batching:** dynamic right-padding, no packing.
- **Checkpoints:** full resume state (model, optimizer, scheduler, every RNG). The **final** checkpoint is evaluated; there is no best-by-val selection.
- **Step budget S (rule, set by P2):**
  - run REF for 30K steps with checkpoints every 2.5K, and evaluate val math-correct accuracy at each;
  - S = the smallest checkpoint reaching ≥ 98% of the run's best val accuracy, rounded up to a multiple of 2.5K and clamped to [10K, 30K];
  - S then applies to every run, including every N.
- **Learning rate (rule, set by P3):**
  - each of the six main arms (REF, T-char, T-bpe-num, learned, none, gelu) sweeps peak LR ∈ {5e-4, 1e-3, 2e-3} at 0.3·S steps, seed 0;
  - choose the LR with the lowest final val completion loss (per-sequence normalized);
  - a tie within 1% relative goes to the lower LR;
  - every other arm (decoupled, answer-first, data-size) uses its parent arm's LR. For the decoupled arms that is REF's LR.
- **Run record** (`runs/<exp_id>/`):
  - config, git hash and dirty flag, `pip freeze`, GPU;
  - dataset, manifest and tokenizer checksums;
  - `metrics.jsonl` (correct per-step loss, real tok/s excluding padding), `eval.json`;
  - wall clock and peak VRAM.

### 4.4 Answer-position control (C-AF)
Answer-first serializes `{"answerKey": ..., "hints": [...]}` with the same content and arm settings as REF.

---

## 5. Metrics

- **Primary endpoint: math-correct accuracy** on G-set **test-ID-unseen**, REF arm and contrasts.
  - Bad JSON, a missing key, unparseable output, truncation or an oracle timeout all count as **wrong**.
  - Fixes audit F6: failures are not dropped from the denominator.
- **Math-correct**, compared by the oracle per answer type:

  | Answer type | Comparison |
  |---|---|
  | integers | exact |
  | fractions | as rationals (an unreduced correct fraction is math-correct but not exact) |
  | decimals (vec_angle) | \|Δ\| ≤ 5e-5 |
  | `sqrt(n)` | symbolic |
  | lists (roots) | as multisets |
  | matrices / vectors | element-wise |
  | polynomials | symbolic (sympy, 2 s timeout); "+ C" required for indefinite integrals |

- **Secondary metrics:**
  - exact match (whitespace-normalized);
  - JSON/format validity;
  - **copy consistency** (the last hint's final value equals the answerKey, when that template's hints state it);
  - hint-step arithmetic consistency;
  - bits per character of the gold completion (comparable across tokenizers);
  - per-item log-likelihood of the gold answer (a continuous, lower-variance companion);
  - accuracy on train-recall, block, OOD-same and OOD-longer;
  - all broken down by template, family, magnitude, length, coverage and hint-step count.
- **Error taxonomy** (automatic): scratchpad error, copy error, format, truncation, repetition, OOD-specific. Plus the free mat_mul diagnostic: c₁₁/c₁₂, which appear in the hints, vs c₂₁/c₂₂, which don't.
- **Cost:** parameters, training FLOPs, tokens seen (logged per run; adopted proposal P-g), wall clock.

**Compute matching (P-g).**
- *Primary:* the same optimizer steps, batch size and examples for every arm.
- *Reported, not matched:* tokens seen and FLOPs, which differ by tokenizer. The T-char arm sees about 2× the tokens per example.

---

## 6. Hypotheses and confirmatory contrasts

All contrasts are **paired by seed** (the same data, order and initialization seed across arms). Δ is in percentage points.

| # | Hypothesis | Contrast (endpoint) | Prediction | Type |
|---|---|---|---|---|
| C1 | **H1a** (replication) | T-digit − T-bpe-num on F-int G-set templates (primary) | Δ > 0 | confirmatory |
| C2 | **H1b** vs H1b′ | T-digit − T-bpe-num on F-frac ∪ F-mat ∪ F-calc G-set templates (primary) | H1b: Δ > 0; H1b′: Δ within ±SESOI | confirmatory (two-sided + equivalence) |
| C3 | **H1c** vs H1c′ | B-in/D-out − D-in/B-out (primary; equals output effect − input effect in the 2×2) | Δ > 0 | confirmatory |
| C4 | **H2** | T-digit − T-char (primary) | two-sided; equivalence possible | confirmatory |
| C5 | **H6-pe (ID)** | rope − learned (primary) | Δ ≥ 0 | confirmatory |
| C6 | **H6-pe (ID)** | rope − none (primary) | Δ ≥ 0 | confirmatory |
| C7 | **H6-pe (length OOD)** | none − rope (OOD-longer) | Δ ≥ 0 | confirmatory |
| C8 | **H6-ffn** vs H6-ffn′ | swiglu − gelu (primary) | two-sided; equivalence possible | confirmatory |
| C9 | **C-AF** | hints-first − answer-first (primary) | Δ > 0, larger for templates with ≥ 2 computational hint steps | confirmatory (main effect); step interaction exploratory |
| C10 | **H3** vs H3′ | (train-recall − test-ID-unseen) on G-set at N = 2K, minus the same gap at N = 128K, REF | Δ > 0 (the gap shrinks with coverage) | confirmatory |
| C11 | **H5-tok** vs H5-tok′ | (T-digit − T-bpe-num) at N = 8K minus the same at N = 128K (primary) | two-sided | confirmatory |
| C12 | **H4a** vs H4a′ | REF block accuracy vs REF test-ID-unseen | equivalence within ±SESOI | confirmatory (equivalence) |

**Descriptive / exploratory (no confirmatory claims):**
- H4b collapse per arm. "Collapse" means OOD-longer accuracy < 10% absolute, or < 25% of that arm's test-ID-unseen accuracy.
- The H5 learning curve over N ∈ {2K, 8K, 32K, 128K}, with diminishing returns judged by Δ per log₂(N) step.
- learned − rope on OOD;
- per-family breakdowns of C2 (F-frac, F-mat, F-calc separately);
- M-set train-recall (memorization/format control);
- the zero-divergence negative control for C1/C2: Δ is expected ≈ 0 on items whose numbers tokenize identically. A clear nonzero Δ there would point to a confound;
- the error taxonomy;
- Baseline A.

---

## 7. Statistics and decision rule

1. **Estimate.**
   - Δ is the mean over seeds of the paired per-item difference in math-correct.
   - 95% CI by **hierarchical bootstrap**: resample seeds, then items within each seed; 10,000 resamples.
   - Per-seed values are always reported.
2. **Test.** A seed-stratified **Cochran–Mantel–Haenszel** test on paired item outcomes: McNemar tables per seed, combined. C10, C11 and C12 compare differences of gaps; for those, the test is the hierarchical bootstrap (two-sided p from the bootstrap distribution).
3. **Multiplicity.** **Holm** correction across the 11 non-equivalence tests C1–C11, family-wise α = 0.05. The equivalence parts of C2, C4, C8 and C12 use TOST at α = 0.05 each; they are reported, not added to the Holm family.
4. **SESOI** = max(2 pp, 2 × the seed SD of REF's primary endpoint from pilot P4), fixed at G3 and written into `decisions.md`.
5. **Claiming an effect** requires all three:
   - (a) the same sign in **every** paired seed;
   - (b) a Holm-adjusted p < 0.05;
   - (c) |Δ| > SESOI.

   **Claiming equivalence** requires the 90% hierarchical-bootstrap CI to lie inside ±SESOI. Anything else is reported as **inconclusive**.
6. **Seeds.**
   - REF: 5 seeds (the P4 runs).
   - Other core arms: 3 seeds (0, 1, 2), paired with REF's seeds 0–2.
   - **P4 gate:** if the REF seed SD on OOD-longer is > 5 pp, arms are cut rather than seeds. Any cut is decided at G3 and logged.
7. **No result-dependent changes.** Analyses not listed here are labelled exploratory.

---

## 8. Run matrix

| Stage | What | Runs | ≈GPU-h (plan estimate; P0 measures) |
|---|---|---|---|
| P0 | env, throughput, determinism (REF seed 0 twice, 1K steps) | — | 0.3 |
| P1 | legacy 34M reproduction from `legacy-v0` with seeding and logging fixes only (Baseline A) | 1 | 2.0 |
| P2 | REF at 30K steps → sets S | 1 | 0.7 |
| P3 | LR sweep, 3 LRs × 6 arms at 0.3·S | 18 | 2.5 |
| P4 | REF × 5 seeds (seed-variance pilot; also REF's final runs) | 5 | 1.8 |
| **G3** | **user approval of the core grid, with measured costs** | | |
| C-tok | T-bpe-num, T-char × 3 seeds | 6 | 2.8 |
| C-arch | learned, none, gelu × 3 | 9 | 3.2 |
| C-af | answer-first × 3 | 3 | 1.1 |
| C-data | REF at N ∈ {2K, 8K, 32K} × 3 (128K = REF) | 9 | 3.2 |
| C-dec | D-in/B-out, B-in/D-out × 3 (**adopted P-b**) | 6 | ≈2.1 |
| C-dtok | T-bpe-num at N = 8K × 3 (**required by adopted H5-tok/C11**) | 3 | ≈1.1 |
| **Total through the core** | | **61** | **≈ 21 GPU-h** |
| Stretch (separate approval) | T-bpe-num at N = 32K × 3; +2 seeds on borderline contrasts; vocab 2K vs 6K | ~13 | ≈ 3.2 |

The order is P0 → P4, then G3, then C-tok → C-dec → C-arch → C-af → C-data → C-dtok, so the most central contrasts land first if the budget gets cut.

---

## 9. Threats to validity (planned mitigations)

| Threat | Mitigation |
|---|---|
| Input and output tokenization confounded | C-dec 2×2 (H1c) |
| Separators / format differ between arms | shared pre-tokenizer; identical non-number tokenization in T-digit and T-bpe-num; no separator tokens in any arm |
| Seed noise (especially OOD) | 5-seed REF pilot, paired design, sign-consistency rule |
| Memorization from tiny spaces mistaken for skill | G/M split; M-set excluded from hash-split endpoints; coverage covariate |
| Ground-truth errors | oracle independent of the generator, 100% agreement test on every split (test 1) |
| Silent truncation | hard length guard; context-length rule |
| Logging/diagnostic bugs (audit F2) | corrected per-step loss, real tok/s, tests 6–8 |
| Hidden test peeking | val pool used for selection only; test splits evaluated only on final checkpoints; no re-selection after seeing test results |
| Small absolute numbers in MathGen (mostly ≤ 3 digits) limit where tokenizers differ | number-token divergence covariate and the zero-divergence negative control |

---

## 10. What this design cannot show

- Anything about pretrained LLMs or about models larger than ~5M parameters.
- Length generalization beyond one extra digit.
- Whether effects hold for natural-language word problems.
- Template-level (compositional) generalization. Instances are held out, not templates; a template-level split is a possible v-next extension.
