# MathGen — Decisions and Deviation Log

Append-only. It records:
- every point where the frozen design (`RESEARCH_DESIGN.md`, tag `prereg-v1`) differs from the original 12-milestone plan;
- every later deviation from the frozen design.

Each entry gives the date, what changed, why, who decided, and the expected impact on claims.

---

## Part A — Design choices fixed at pre-registration (2026-10-17, before tag `prereg-v1`)

| ID | Change relative to the original plan | Reason | Decided by | Impact |
|---|---|---|---|---|
| D-001 | RQ reworded to RQ v2 (RQ-A/B/C + secondary) | M2: the integer-arithmetic tokenization effect is already established; the open part is transfer to structured families plus memorization accounting | user ("Adopt RQ v2 + all") | Framing only; same four research areas |
| D-002 | H1 split into H1a (integer replication), H1b (structured families), H1c (output vs input) | M2 §3.1; Baeumel et al. 2026 | user | More specific claims; H1a is labelled a replication |
| D-003 | New arms C-dec (D-in/B-out, B-in/D-out × 3 seeds) | Adopted proposal P-b; confound named by Baeumel et al. 2026 | user | +6 runs, ≈ +2.1 GPU-h |
| D-004 | H4 split into H4a (block holdout, equivalence) and H4b (larger operands, descriptive collapse) | M2 §3.2: the literature predicts opposite outcomes | user | H4b is no longer a confirmatory contrast |
| D-005 | H6 positional contrasts split by test type (C5/C6 in-distribution, C7 length OOD) | M2 §3.2/§3.6: RoPE best ID, NoPE best length OOD (Number Cookbook) | user | Directional predictions instead of two-sided |
| D-006 | New H5-tok (C11); T-bpe-num at N = 8K × 3 promoted from stretch item S1 into the core (arm C-dtok) | H5-tok was adopted and can't be tested without it | Claude, as a consequence of the adopted hypothesis; **flagged for G3** | +3 runs, ≈ +1.1 GPU-h; the user can cut it at G3 |
| D-007 | 23 templates, not 22 | M1 measured 23 generator functions | measurement | none |
| D-008 | Families defined (F-int / F-frac / F-mat / F-calc); G-set = 14 templates with space ≥ 500, M-set = 9 with ≤ 150 | Needed to operationalize H1a/H1b. The plan said G-set was "about ≥ 1K"; ≥ 500 keeps arith_pemdas, vec_angle, limit_substitution and deriv_polynomial in the hash-split endpoints | Claude | Smaller test pools for 512–1,250-instance templates (≈ 25% of the space) |
| D-009 | 7 block regions (B1–B7), not "about 6"; B5–B7 added so F-mat, F-frac and F-calc each have one | Each family has a block | Claude | +3 block regions in eval only |
| D-010 | Training sets vary with the run seed (seeds change data sampling as well as init and order); tokenizers fit on one fixed seed-0 sample | Captures data-sampling variance (Bouthillier et al. 2021) while pairing arms within a seed | Claude | Seed SD includes data variance, so it's a more conservative SESOI |
| D-011 | LR chosen by lowest val completion loss (not val accuracy) | Accuracy at 0.3·S is noisy; loss is comparable within an arm | Claude | none on claims |
| D-012 | Step-budget rule fixed (≥ 98% of best val accuracy, multiple of 2.5K, clamped to [10K, 30K]) | The plan said only "set in P2" | Claude | none on claims |
| D-013 | Context-length rule: smallest power of two ≥ the longest T-char sequence over all splits, ≥ 256 | Avoids the char-arm truncation found in audit F8 | Claude | Probably 512, measured in M4 |
| D-014 | Decision rule extended: hierarchical bootstrap CIs, TOST equivalence for nulls, SESOI = max(2 pp, 2 × REF seed SD) | M2 §3.9; the plan's rule could not support null claims | user (adopted P-i) | Null results can be reported as equivalence, not only "inconclusive" |
| D-015 | Core run count 52 → 61 (≈ 18 → ≈ 21 GPU-h) | D-003 + D-006 | user (D-003); flagged (D-006) | Shown again with measured costs at G3 |

---

## Part B — Deviations after `prereg-v1`

| ID | Date | Deviation | Reason | Impact on claims |
|---|---|---|---|---|
| DV-001 | 2026-10-17 (M4) | Polynomial and antiderivative answers are compared as exact `{power: coefficient}` dictionaries under a restricted grammar, not with "sympy, 2 s timeout" (RESEARCH_DESIGN.md §5). No answer type uses sympy. The grammar accepts ASCII-digit terms `c`, `cx`, `cx^n` (optionally `c*x`, `x**n`), joined by `+`/`-`, in any order; antiderivatives need a trailing `+ C` | No timeouts and no timing dependence; hostile strings are rejected fast | **Can only lower math-correct**, never raise it. The narrower grammar rejects some forms sympy would accept, e.g. `x^3/3 + C`, `2x^(3)`, `3·x^2`, `C + x^2` (found in M4 code review). The training targets use exactly the canonical form, so the expected effect is small, and it is measured: the error taxonomy reports "unparseable" predictions per arm |
| DV-002 | 2026-10-17 (M4) | Several eval cells hold fewer items than RESEARCH_DESIGN.md §3.2 lists, because the pre-registered ranges contain fewer distinct instances. Recorded per cell in `data/manifests/build.json` → `shortfalls`. test-ID-unseen: arith_pemdas 129, vec_angle 143 (of 200). Block: B1 121, B2 112, B4 81, B6 175, B7 180 (of 200). OOD-same: limit_factor 3, deriv_power 12, limit_infinity 18, algebra_quadratic 21, integral_power 24, arith_fractions 25, integral_definite 45 (of 100). OOD-longer: limit_factor 21, deriv_power 84, integral_power 84 (of 100) | The pre-registered ranges are kept; enlarging them would change the design | Per-template estimates for those cells are low-n and reported with their n. Pooled endpoints are barely affected (test-ID-unseen has 2,672 items; OOD-same has 1,248) |

| DV-003 | 2026-10-17 (M5) | **Server shims kept minimal.** The plan wanted `src/*` to become thin shims over `mathgen/`. Instead, the only `src/` change is the one-line F3 fix in the generator that `/generate` serves (`src/dataset.py`, arith_integer). `src/verifier.py` (`/grade`, `smart_equal`) is unchanged | Swapping in the stricter research oracle (DV-001 grammar) could change how real app users' answers are graded, which is a product decision outside this study. Plan test 12 (server contract, 30/30 pairs) passes, plus a test that served subtraction is unbiased | None on research claims. Baseline A (P1) runs from the `legacy-repro` worktree, which keeps F3 on purpose |

## Part D — Implementation details that `prereg-v1` does not specify (fixed before any training)

| ID | Date | Detail | Choice | Rationale |
|---|---|---|---|---|
| I-001 | 2026-10-17 (M5) | BPE merge budgets | Up to 2,000 text merges and 1,000 digit merges. Each stops early when the best pair occurs fewer than 2 times; ties broken lexicographically | Fitting the 50K fit sample stops early at **400 text merges** and **384 digit merges** in practice. Vocabularies: T-char 89, T-digit 489, T-bpe-num 873. Mean / max tokens per example on test-ID-unseen + OOD-longer: 144.1/239, 82.1/170, 74.6/134 |
| I-002 | 2026-10-17 (M5) | Pre-tokenizer regex | special tokens, then `[0-9]+` \| `[^0-9\s]+` \| `\s+` | Makes RESEARCH_DESIGN.md §4.1's "ASCII digit runs \| other non-space runs \| spaces" concrete |
| I-003 | 2026-10-17 (M5) | Dropout | 0.0 in every arm (legacy used 0.1) | Not pre-registered. Weight decay (0.1) is the only regularizer, so memorization isn't confounded by a second one, and training stays deterministic |
| I-004 | 2026-10-17 (M5) | "Per-item log-likelihood of the gold answer" (RESEARCH_DESIGN.md §5) | Implemented as the gold **completion's** summed NLL and bits per character | The answer span isn't token-aligned across arms (pre-tokenization can merge the opening quote into the first token), so a per-character completion measure is the comparable one |
| I-005 | 2026-10-17 (M5, code review) | Embedding initialization | Token embedding std 0.02; the tied head is skipped by the Linear init | Legacy re-initialised the shared tensor through the head (std ≈ 0.0058), so learned position embeddings started ~3.5× larger than token embeddings: a confound for H6-pe. Loading legacy weights is unaffected |
| I-006 | 2026-10-17 (M5, P0) | Performance | Fused `F.rms_norm` (same math), one RoPE cos/sin table per forward, fused AdamW on CUDA, loss accumulated on the device | P0 measured 0.51 s/step (9.4K tok/s) on battery. Profiling showed elementwise kernels bound by memory bandwidth (~23 GB/s effective) plus 866 launches/step. After the changes: ~0.36 s/step on battery. Numerics are equivalent (legacy-logit parity still holds) |
| I-007 | 2026-10-17 (M5, code review) | Run identity | exp_id carries exact N and steps; reusing an exp_id with a different config raises (only `device`, `eval_every` and `notes` may differ) | Prevents results from being silently reused or mixed |

## Part C — Values set by pre-registered rules

| Quantity | Rule (RESEARCH_DESIGN.md) | Value | Measured in | Evidence |
|---|---|---|---|---|
| Context length | §4.2: smallest power of two ≥ the longest T-char sequence over all built splits, min 256 | **256** (longest = 239: mat_mul OOD-longer) | M4 build, 2026-10-17 | `data/manifests/build.json` → `context`. D-013 had expected ~512; the legacy overflow came from `\uXXXX` escapes, which data v1 doesn't produce |
| Step budget S | §4.3 | *pending P2* | | |
| Per-arm learning rate | §4.3 | *pending P3* | | |
| SESOI | §7.4 | *pending P4 → G3* | | |
