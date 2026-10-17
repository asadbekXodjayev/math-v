# Dataset card — MathGen data v1

## Summary
Synthetic, template-generated structured-math problems for **conditional solving**: question → JSON with worked-solution `hints` and an `answerKey`.
- 23 templates ported from the legacy MathGen generators (`legacy-v0`).
- Built deterministically from `configs/base.yaml`; build manifest in `manifests/build.json`.
- Purpose: a controlled study of number tokenization and memorization in ~5M-parameter Transformers ([`research/RESEARCH_DESIGN.md`](../research/RESEARCH_DESIGN.md)).
- Not intended for training general-purpose math models.

## Composition

| Family | G-set (hash-split endpoints; ID space size) | M-set (memorization/format control) |
|---|---|---|
| F-int | arith_integer 2,117 · arith_pemdas 512 · algebra_linear 1,120 · func_evaluate 4,455 · limit_substitution 864 | — |
| F-frac | arith_fractions 1,225 · vec_angle 625 | — |
| F-mat | vec_dot 20,736 · mat_add 43.0M · mat_scalar 26,244 · mat_mul 43.0M · mat_det 6,561 · mat_transpose 6,561 | vec_magnitude 144 · linear_transform 144 |
| F-calc | deriv_polynomial 1,080 | deriv_power 20 · deriv_at_point 80 · integral_power 12 · integral_definite 27 · limit_factor 5 · limit_infinity 50 · algebra_quadratic 91 |

**Format:**
- Prompt: `[SOLVE] <question> <SEP><JSON>`.
- Target: `{"hints": [...], "answerKey": "..."}<EOS>`, or answer-first for the C-AF control.
- JSON is written with `ensure_ascii=False`.

**Lengths:**
- The longest prompt + target is **239 character-level tokens** (mat_mul, OOD-longer), so the pre-registered context-length rule gives **256**.
- Mean over test-ID-unseen + OOD-longer: 144.1.

## Collection and generation
- Each operand is drawn uniformly and independently from its configured range, as in the legacy generators.
- The answer comes from the template code. An **independent oracle** re-derives it from the question text alone, and all 697,822 built items agree with it.
- **Instance key:** `template|canonical operands`.
- **Pool:** `int(sha256(key)[:16], 16) mod 1000` → train (< 700) / val (< 750) / test.
- **Block regions** B1–B7 are removed from every pool.
- **M-set** templates (instance space < 500) are not hashed: their whole space is in the train pool.

**Training stream for seed *s*:**
- Pick a template uniformly, then an instance uniformly from that template's train pool (rejection sampling).
- The first N lines form the size-N set, so sets are nested.
- Seeds give different sets. Every arm with the same seed gets the same set and order.

Distinct instances / ID space, seed 0 (the G-set ceiling is about 0.70, the train-pool share):

| N | algebra_linear | arith_integer | func_evaluate | mat_det | vec_dot | mat_mul | M-set (all) |
|---|---|---|---|---|---|---|---|
| 2K | 0.055 | 0.042 | 0.017 | 0.014 | 0.004 | ≈0 | 0.41–1.00 |
| 8K | 0.219 | 0.151 | 0.073 | 0.054 | 0.018 | ≈0 | 0.92–1.00 |
| 32K | 0.546 | 0.418 | 0.246 | 0.188 | 0.066 | ≈0 | 1.00 |
| 128K | 0.635 | 0.644 | 0.547 | 0.493 | 0.220 | ≈0 | 1.00 |

## Changes from the legacy data
- **Subtraction bug fixed** (audit F3): operands are ordered before rendering. The a == b rate in seed-0 training data is 3.02% (n = 1,955; expected 1/29 = 3.45%), against 51.3% in legacy.
- **No `\uXXXX` escapes** (audit F8).
- **algebra_quadratic roots are sorted**, so each question has one target.
- **Task:** conditional solving replaces self-generation; GRADE/ASSESS tasks are excluded.

## Eval cells smaller than requested
Some pre-registered ranges contain fewer distinct instances than the requested cell size. The build takes all of them and records the shortfall in `manifests/build.json` → `shortfalls` (logged as DV-002 in `research/decisions.md`):

| Split | Smaller cells (got / requested) |
|---|---|
| test-ID-unseen | arith_pemdas 129/200, vec_angle 143/200 |
| block | B1 121, B2 112, B4 81, B6 175, B7 180 (of 200) |
| OOD-same | limit_factor 3, deriv_power 12, limit_infinity 18, algebra_quadratic 21, integral_power 24, arith_fractions 25, integral_definite 45 (of 100) |
| OOD-longer | limit_factor 21, deriv_power 84, integral_power 84 (of 100) |

## Known limitations
- **Small numbers.** Most numbers are 1–3 digits, so digit vs multi-digit tokenization can only differ on multi-digit numbers. M5 records a per-item number-token-divergence covariate.
- **M-set overrepresentation.** Uniform template sampling puts the 9 tiny M-set templates at ~39% of training examples, each repeated thousands of times. This is intended: they are the memorization control.
- **Legacy renderings kept as-is.** Examples: quadratic hints like `(x - -3)`, rounded floats in vec_angle hints, the question using `*` while the hint uses `×` in arith_integer.
- **Instances are held out, templates are not.** No template-level generalization test.
- **Hints usually end by stating the answer.** So hints-first output is close to a scratchpad followed by a copy (see the literature review §3.8).
