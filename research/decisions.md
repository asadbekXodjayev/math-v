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

*(none yet)*
