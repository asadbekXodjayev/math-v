# MathGen — Pilot Report (Milestone 6)

**Status: draft.** P0, P2, P3 and P4 are complete. P1 (legacy reproduction) is still running, and its section is marked *pending*. Gate G3 has not been requested yet.

All tables are generated from the run directories and pilot artifacts by `scripts/pilot_report.py` into `research/artifacts/pilot_tables.md`. This file adds the narrative and the decisions G3 needs. Numbers quoted here come from those artifacts; where they disagree, the artifact is right.

---

## 1. What the pilots are for

The pre-registration (`RESEARCH_DESIGN.md`, tag `prereg-v1`) leaves four values to rules applied in the pilots, so that none of them is chosen after seeing the confirmatory results:

| Value | Rule | Pilot | Value |
|---|---|---|---|
| Step budget S | §4.3: smallest checkpoint ≥ 98% of best val accuracy, rounded up to 2.5K, clamped to [10K, 30K] | P2 | **27,500** |
| Per-arm LR | §4.3: lowest val completion loss at 0.3·S over {5e-4, 1e-3, 2e-3}, seed 0 | P3 | see §4 |
| SESOI | §7.4: max(2 pp, 2 × REF seed SD on the primary endpoint) | P4 | **2.00 pp** (SD 0.84 pp; the 2 pp floor binds) |
| Arms to cut, if any | §7.6: if REF's seed SD on OOD-longer is > 5 pp, cut arms rather than seeds | P4 | **none** (SD 0.13 pp) |

The pilots also check reproducibility (P0), measure what the core grid will cost on this laptop, and reproduce the legacy model as Baseline A (P1).

---

## 2. P0: environment, throughput and determinism

Hardware: RTX 3050 Laptop GPU (4 GB), 7.6 GB system RAM, torch 2.11.0+cu128, bf16.

- **Power matters more than any code change.** The same 1K-step REF run took 0.352 s/step on battery and 0.059 s/step on AC (~6×). Every pilot after P0 ran on AC.
- **Determinism.** Two runs of identical training code (A, B) were *not* bitwise identical. They matched through logged step 700, then drifted (max |Δw| 0.0116, final-loss difference 4e-4). Run A ran on battery in an earlier session with very little free RAM; the cause was not identified. With the same code and conditions, both kernel modes reproduce bitwise (default: B = C; deterministic: D = D2).
- **Decision I-008:** every study run and evaluation uses `torch.use_deterministic_algorithms(True)`. That is the only mode PyTorch guarantees. It selects different kernels from the default (B ≠ D from step 100), and costs ≈ 7% throughput (0.0594 → 0.0634 s/step). It was adopted before any study run.
- **Eval:** all 11,195 eval items in 58–59 s on a 1K-step model, byte-identical records across two evaluations. Peak VRAM: 796 MB training, 593 MB eval, so the 4 GB card is not the constraint. System RAM is (§7).

## 3. P2: step budget S

REF, seed 0, 30K steps, val accuracy every 2.5K (n = 644).

- Best val accuracy 93.3% at 30K; 92.4% at 27.5K is the first checkpoint ≥ 98% of best → **S = 27,500**.
- **The curve had not plateaued.** It was still rising at 30K, and it is not monotone at n = 644 (89.9% at 22.5K, 89.6% at 25K). S is set by the 98% rule close to the upper clamp, not by a plateau. Every arm trains for the same S, so comparisons are at a fixed compute budget, not at convergence. An arm that learns more slowly than REF is compared before it has finished learning. This is reported as a limitation, not corrected.
- Training wall time: 32 min.

## 4. P3: per-arm learning rate

3 LRs × 6 arms at 8,200 steps, seed 0. 0.3 · 27,500 = 8,250, which `round()` (half-to-even) took to 8,200. The 50-step difference is logged in `decisions.md` Part C.

| arm | chosen LR | position in the grid |
|---|---|---|
| ref, char, bpe_num, none | 5e-4 | **lowest value** |
| learned, gelu | 1e-3 | interior |

- **4 of 6 arms chose the lowest LR in the grid**, so their optimum may lie below 5e-4. The grid is kept as pre-registered: widening it now would be a result-dependent change. Consequence: those arms may be slightly under-tuned relative to learned and gelu. That matters most for C-arch (C5–C7), where REF (5e-4, possibly under-tuned) is compared with learned and gelu (interior optima).
- The selection rule (val loss, D-011) and val accuracy agree for 5 of 6 arms. For gelu, loss picks 1e-3 while accuracy is 0.3 pp higher at 5e-4, well within the noise at n = 644.
- *Observation, not a result:* at 0.3·S, seed 0, T-bpe-num reached a higher val accuracy than REF (T-digit). One seed at a short budget says nothing about C1–C3, which are tested at S with three paired seeds.

## 5. P4: seed variance and SESOI

REF × 5 seeds at S = 27,500, LR 5e-4, full eval. These are also REF's final runs in the core grid. Per-seed values, means and SDs for every split are in `pilot_tables.md`.

| endpoint | mean over 5 seeds | seed SD (pp) |
|---|---:|---:|
| **primary** (test-ID-unseen, G-set) | 92.0% | **0.84** |
| val | 92.7% | 0.87 |
| block | 39.4% | 3.07 |
| OOD-same | 19.2% | 2.93 |
| OOD-longer | 0.3% | 0.13 |
| train-recall | 99.7% | 0.23 |

- **SESOI = max(2 pp, 2 × 0.84 pp) = 2.00 pp.** The 2 pp floor binds, so the primary endpoint's seed noise is not what limits the contrasts.
- **P4 gate: not triggered** (OOD-longer SD 0.13 pp ≤ 5 pp). But the gate passes only because every seed sits at the floor (0.1–0.4%), not because OOD-longer is stable. No arms are cut.
- **Block and OOD-same vary 3.5–3.7× more across seeds than the primary endpoint**: SD ≈ 3 pp, above the 2 pp SESOI. Of the confirmatory contrasts, only C12 (REF block vs test-ID-unseen) uses one of these endpoints. OOD-same is descriptive only.
- **C7 is near a floor.** C7 is NoPE − RoPE on OOD-longer, where REF (RoPE) scores 0.3%. An effect is claimable only if NoPE reaches roughly 2.3% or more. Otherwise C7 will be inconclusive, which is a property of the design, not evidence that NoPE fails to help.
- *Descriptive, not a test:* REF's pooled block accuracy (39.4%) is far below its test-ID-unseen accuracy (92.0%), and train-recall is 99.7%. The pooled numbers are not matched by template; the pre-registered C12 test runs in M9.
- Training wall time per seed: 0.48 h (0.0632 s/step). Eval: 63 s per run.

## 6. P1: legacy reproduction (Baseline A) — *pending*

The legacy 34M pipeline is retrained from the `legacy-repro` worktree (seeding and logging fixes only; F3 kept on purpose). It is then scored by the legacy self-consistency eval and by forced-prefix scoring on data v1. Baseline A is provenance only: the legacy pipeline saw most ID instances in training, so its numbers are not comparable to the study's arms.

---

## 7. Cost of the core grid

The full projection table is in `pilot_tables.md`. Every arm's P3 speed is scaled to REF's measured speed at S (P4, 0.0632 s/step; scale 0.993), and eval time comes from the P4 runs (63 s). Projection: **36 runs ≈ 17.9 GPU-h**. The largest items are C-data (9 runs, 4.5 h) and T-char (3 runs, 2.2 h; 0.092 s/step).

For comparison: the pre-registration's plan estimate for the same 36 core runs was ≈ 13.5 GPU-h (§8). That estimate assumed the 15K-step draft budget; S came out at 27,500.

**Wall-clock risk: system RAM, not the GPU.** The pilot chain has been stopped five times by Claude Code's low-memory guard (`lab_log.md`). A training process holds ~1.4 GB working set, leaving ~0.9 GB free on this 7.6 GB laptop. Every study run resumes from its last checkpoint, so a stop costs at most one checkpoint interval (2.5K steps, ~2.5 min) plus the time until relaunch. P1 is the exception: the legacy reproduction always restarts from epoch 1, so a stop there loses the whole attempt. The relaunch is the expensive part, because the session does not restart stopped work on its own. For the core grid, running `scripts/` from a separate terminal avoids the guard.

## 8. Bugs found and fixed during the pilots

Each was fixed test-first. None of them affected a reported number. The P1 attempt now running loaded `pilots.py` before 58084fa; any relaunch uses the fixed code, and the generated P1 table repeats the `best.pt` check for this attempt.

| Commit | Bug | Would have caused |
|---|---|---|
| ee2bffa | Checkpoints loaded with `map_location=cuda` moved the CPU RNG state to the GPU | Every GPU resume crashed (`RNG state must be a torch.ByteTensor`) |
| 003dd9d (I-010) | The cost projection read eval time from file timestamps | Re-evaluation on relaunch inflated the G3 projection (a test reproduced 300 min of eval per run instead of 1) |
| 58084fa | P1 treated ≥ 10 lines of the legacy train log as a finished run, but the legacy `--fresh` run appends to that log | Two stopped attempts (e.g. 6 + 5 epochs) would have skipped training and scored an unfinished model as Baseline A |
| 47f3f17 | Follow-ups from code review: a half-written last log line crashed the P1 check; archive names could collide; P1 did not check that `best.pt` is the best logged epoch (legacy logs an epoch before saving `best.pt`) | A crash on relaunch, or Baseline A scored with an older checkpoint after a stop in a ~1 s window |

## 9. Limitations that come from the pilots

1. S comes from a curve that was still rising (§3).
2. 4 of 6 arms sit at the lower edge of the LR grid (§4).
3. LRs were chosen on one seed at 0.3·S.
4. Three arms (answer_first, dec_db, dec_bd) have no P3 run. They take REF's LR by design, and their cost is projected from borrowed speeds.
5. P4 seeds 0 and 1 record commit 003dd9d with `git_dirty = true`. The chosen LRs had been written into `configs/arms/*.yaml` before they were committed. Seeds 2–3 (fe4633a) and seed 4 (7e41fb9) record clean trees. Between those commits `src/` is unchanged, and the only `configs/` difference is those 9 LR lines, so all five seeds trained the same code and config.
6. The P4 gate passes at a floor (§5), and C7 can only detect an effect if NoPE lifts OOD-longer above roughly 2.3%.
7. With 3 seeds per arm, contrasts on endpoints whose seed SD exceeds the SESOI (block, for C12) are more likely to end inconclusive under the same-sign rule.

## 10. Decisions requested at G3 — *draft; final once P1 is in*

1. **Approve the core grid**: 36 runs, ≈ 17.9 GPU-h, in the pre-registered order C-tok → C-dec → C-arch → C-af → C-data → C-dtok.
2. **D-006 (flagged at G2):** keep or cut C-dtok (T-bpe-num at N = 8K × 3, ≈ 1.4 GPU-h). Without it H5-tok (C11) cannot be tested. Recommendation: keep it.
3. **P4 gate:** not triggered, so no arms are cut (§5: it passes at a floor).
4. **SESOI = 2.00 pp**, by the pre-registered rule. Frozen into `decisions.md` Part C on G3 approval.
5. **LR grid edge (§4):** recommendation is to keep the pre-registered LRs. Optionally, add one sensitivity run labelled exploratory (REF at 2.5e-4, 0.3·S, seed 0, ≈ 10 min) to show how much the grid edge costs. It would not change any selected LR.
6. **Seed noise on block (§5, limitation 7):** recommendation is to keep 3 seeds per arm as pre-registered. The plan's stretch item S2 (+2 seeds on borderline contrasts, separate approval) is the pre-registered remedy if contrasts end near the threshold.
