# Lab log

Append-only, newest entries last. Record only what was actually run, with the commands and their real outcomes. Failed attempts and deviations are logged too.

---

## 2026-10-14 — M1 repository audit

- **Provenance:** `git init`; committed the legacy tree untouched as `ea6caf9` and tagged it `legacy-v0`. `.gitignore` excludes `*.pt`, `.venv/`, `runs/` and `data/*` (except manifests and cards). Checkpoint and tokenizer sha256 hashes are in `research/artifacts/CHECKPOINTS.sha256`.
- **Probe:** `python -B research/audit_scripts/audit_probe.py` (system Python 3.14.2, torch 2.12.0+cpu). It reproduced findings F1–F9; results are in `research/artifacts/audit_results.json`. Full write-up: `research/REPOSITORY_AUDIT.md`.
- **Corrections to my own planning notes** (found by measurement):
  - the generator has **23** templates, not 22;
  - the baseline_v2 tokenizer has **6,031** tokens. The "6,009" in `train_run.log` belongs to a different run that crashed.
- **New finding during the audit:** `sympy` is imported by `src/verifier.py` but not declared in `requirements.txt`. It's only installed because torch depends on it.
- **Environment:** created `.venv` (Python 3.14.2), with torch 2.11.0+cu128 from the cu128 index. `torch.cuda.is_available()` is True and bf16 is supported. Pinned with `pip freeze > requirements.lock` (39 packages).
- **Probe rerun in the venv:** same F1–F7/F9 numbers as the system-Python debug run. F8 was re-measured on real (topic, difficulty) prompts, because the first version used a dummy topic string and under-counted lengths:
  - escapes in 84.3% of samples;
  - 45.0% of char-level samples are longer than 257 chars.
- **CUDA check:** legacy `best.pt` generates on the GPU. The `cudaErrorUnknown` comment in `src/server.py` is stale for this driver and torch combination.
- **Server smoke** (`research/audit_scripts/server_smoke.py`, FastAPI TestClient): `/health` ok; `/generate` valid for 30/30 topic × difficulty pairs on the generator source and 3/3 on the model source.
- **Repo changes after `legacy-v0`:** an audit notice at the top of the README (claims corrected, nothing rewritten); `research/` (audit report, scripts, artifacts, this log); `requirements.lock`. No changes to `src/`.
- **Gate G1:** stopped here for user review of `research/REPOSITORY_AUDIT.md`.

---

## 2026-10-17 — M2 literature review and research gap

- **Gate G1:** the user said "resume the task" after the G1 stop. I read this as approval of the M1 audit and started M2. M2 involves no code changes to `src/` and no GPU work.
- **Outputs:**
  - `research/LITERATURE_REVIEW.md`
  - `research/RELATED_WORK_TABLE.md` (part B generated)
  - `research/RESEARCH_GAP.md`
  - `research/REFERENCES.bib` (generated)
  - tooling in `research/lit_scripts/` (`candidates.json`, `verify_refs.py`, `build_bib.py`)
  - evidence in `research/artifacts/m2_*`
- **Verification:** `.venv/Scripts/python.exe -B research/lit_scripts/verify_refs.py`. **92/92** references found and title-matched against arXiv API / Crossref / official page.
  - The first run flagged 1 title: xVal had been renamed on arXiv after v1. Corrected.
  - `build_bib.py` refuses to build if any reference fails verification.
- **Search:** 75-paper seed list from prior knowledge, plus a delegated novelty search:
  - 102 arXiv API queries (≤50 hits each, title-screened) and 12 web searches;
  - 30 shortlisted, of which 13 were already in the seed list;
  - 15 of the 17 new ones included; 2 out of scope (2604.27551, 2604.13082);
  - 2 more added via a Crossref follow-up of one unverified hit (MDPI DOI 10.3390/electronics15091847; arXiv 2511.10566).
- **Full text:** 12 closest papers read by a delegated agent. Its 227 quotes of 25+ characters were machine-checked against the fetched text. I checked 6 more by full-text search.
- **Tool problems (logged so they aren't repeated):**
  - DBLP and OpenReview APIs now sit behind bot challenges; not usable from scripts.
  - Semantic Scholar's per-paper endpoint returned 429; the batch endpoint works.
  - Semantic Scholar venues proved unreliable: the year is usually the arXiv year, and SCAN is listed as ICLR while the authors' arXiv note says ICML 2018. Not used.
  - OpenAlex's free daily budget ran out during the search agent's run. Citation counts are deliberately not used.
  - A first verification run that included the DBLP step hung and was killed (two `verify_refs.py` processes, PIDs 21212/12388).
- **Corrections to delegated-agent reports, found by checking sources:**
  - "Are Non-BPE Tokenizers Ready to Improve LLM Numeracy?" is listed under the 3rd AI for Math **Workshop** at ICML 2026, not the main track.
  - BitTokens' "small" models are 117M parameters.
  - Kasliwal 2026 trains arithmetic after WikiText-103 pre-training and uses a single seed.
- **Key findings for M3** (details in the review §0 and §4):
  - Single-digit > multi-digit tokenization is established for **integer** arithmetic from scratch, and is output-side (Baeumel et al. 2026, ≈3.4M parameters, 10 seeds).
  - Extrapolation to larger operands fails by default.
  - RoPE is best in-distribution, NoPE/ALiBi best for length OOD.
  - MathGen hints are worked solutions ending in the answer, so hints-first ≈ scratchpad.
- **Gap:** no study crosses tokenization with instance-level seen/unseen/held-out-value/larger-operand splits on multi-family structured math in sub-10M from-scratch models with seeds. Checked against Baeumel 2026, Zhou 2024 and Number Cookbook (plus Lee 2023, Saxton 2019, Kasliwal 2026).
- **Proposed, not adopted:**
  - RQ v2 (`RESEARCH_GAP.md` §6), which re-weights the emphasis to "does the integer result transfer beyond integers" plus memorization accounting;
  - hypothesis splits H4a/H4b and H6-pe;
  - new H2 (tokenizer × data budget);
  - optional decoupled input/output tokenization check (+6 runs).
  These need user approval at G2.
- **Repo changes:** only new files under `research/` and this log entry. No changes to `src/` or the README.
- **Venues (added after the entry above):** a delegated pass confirmed venues on official pages. 64/92 references now cite a confirmed venue (60 official page, 4 publisher Crossref records); 28 are cited as arXiv. Semantic Scholar's venue hint disagreed with the official page for 33 of the 58 checked. Evidence: `research/artifacts/m2_venue_checks.json`, merged by `research/lit_scripts/merge_venues.py`.
- **Correction to my own M2 text:** `RESEARCH_GAP.md` §6 first said the H2 and H5 labels were unused. The approved plan defines H2 (digit vs char) and H5 (data efficiency), so the table was renumbered to the plan's H1–H6 + C-AF.

---

## 2026-10-17 — User decisions after M2 (asked via /idea-loop)

- **Loop scope:** "MathGen to G3". Run M3 (pre-registration) → M4 (data v1 + oracle, tests) → M5 (framework, P0, P1) → M6 (pilots P2–P4) autonomously, then stop at G3 before the core grid. Choosing to continue past M3 is treated as approval of G2. **G3 still needs an explicit go.**
- **RQ:** "Adopt RQ v2 + all". This adopts RQ v2 (`RESEARCH_GAP.md` §6), the hypothesis refinements (H1a/b/c, H4a/b, H5-tok, H6-pe/ffn) and proposals P-a…P-i, including P-b, the decoupled input/output tokenization check (+6 runs).

---

## 2026-10-17 — M3 pre-registration

- **Outputs:**
  - `research/RESEARCH_DESIGN.md`: RQ v2, task and format, data v1 (families, G/M sets, hash pools, 7 block regions, OOD rule and ranges), arms, training protocol rules, metrics, 12 contrasts, decision rule, run matrix;
  - `research/decisions.md`: D-001…D-015, where the frozen design differs from the original plan.
- **Values set by rule, not frozen now:** context length (M4 measurement), step budget S (P2), per-arm LR (P3), SESOI (P4 → G3).
- **Run matrix:** 61 runs through the core (≈ 21 GPU-h, plan estimates) vs 52 in the original plan. The increase is C-dec (+6, adopted P-b) and C-dtok (+3, needed to test the adopted H5-tok). D-006 is flagged for G3.
- **G2:** treated as approved by the user's "MathGen to G3" choice (see the entry above). Tag `prereg-v1` marks the frozen design.

---

## 2026-10-17 — M4 data v1 and oracle integrity

- **Code (test-first; each test was watched failing before implementation):**
  - `src/mathgen/data/`: `templates.py` (23 ports), `ranges.py`, `splits.py`, `formats.py`, `build.py`, `config.py`;
  - `src/mathgen/eval/oracle.py`;
  - `configs/base.yaml`, `scripts/build_data.py`, `pyproject.toml` (pytest config), `.gitattributes` (LF).
- **Tests:** `tests/test_oracle.py`, `tests/test_templates.py`, `tests/test_data.py`. `pytest`: **243 passed, 10 skipped**. The skips are the 5 templates with no OOD-same range, × 2 tests.
  - Byte-for-byte parity with the legacy generators on 200 ID and OOD draws per template (legacy fed the same operands through a scripted `random`).
  - Generator vs oracle agreement on 2,000 draws per template per split.
  - F3 fix: a == b at 1/29 ± 0.005 over 150K draws.
  - Space sizes equal the M1 audit; G/M split equals the pre-registration.
  - Block predicates; determinism (byte-identical manifest); split integrity; nested training subsets; uniform templates.
- **Build:** `.venv/Scripts/python.exe -B scripts/build_data.py --config configs/base.yaml`, ~40 s.
  - Run twice → `cmp` of `data/manifests/build.json`: **identical**.
  - **697,822 items, 0 oracle disagreements.**
  - Eval sizes: test-ID-unseen 2,672; val 644; block 1,069; OOD-same 1,248; OOD-longer 2,189. Train 5 × 128K; fit 50K.
  - Subtraction a == b in seed-0 training data: 3.02% (n = 1,955; expected 3.45%; legacy 51.3%).
- **Context length (rule):** longest T-char sequence = 239 → **256**. I had expected ~512 (D-013); the difference is the removed `\uXXXX` escapes. Logged in `decisions.md` Part C.
- **Code review (REVIEW gate, `/code-review medium`)** found 10 issues; all fixed with failing tests first:
  - oracle crash on `1/00x`;
  - dangling `*` accepted;
  - Unicode digits accepted;
  - DV-001 over-claimed equivalence (rewritten: the narrower grammar can only lower scores);
  - block-before-M-set ordering in `pool()`, plus a G-set assertion;
  - silent eval-cell shortfalls (now recorded; DV-002);
  - CLI path resolution;
  - CRLF-dependent hashes (normalized, plus `.gitattributes`);
  - a determinism test that compared only part of the manifest;
  - repeated enumeration.
- **Deviations:** DV-001 (no sympy in scoring) and DV-002 (small eval cells). Both are in `research/decisions.md`.

---

## 2026-10-17 — M5 framework (built), P0 interrupted

- **Built, test-first:** tokenizer arms, model arms, training loop, decoding/eval, pilot scripts, Baseline A scorer, the `legacy-repro` worktree (2 commits on branch `legacy-repro`), served F3 fix. Full suite: **all passing**. Plan tests 1–12 are covered.
- **Code review (M5):** 10 findings, all fixed with failing tests first; see `decisions.md` I-005…I-007 and DV-003.
- **P0 run A** (`runs/p0-ref-s0-st1000-a`): completed 1,000 steps on GPU.
  - Final train loss 0.076; ~13.9K tok/s, ~0.35 s/step **on battery power**.
  - Profiling found elementwise kernels bound by memory bandwidth, consistent with a throttled laptop GPU (BatteryStatus = 1). See I-006.
- **P0 run B was stopped by Claude Code because system memory was critically low** (0.7 GB free of 7.6 GB). It had only just started.
  - P0 is therefore incomplete: the determinism comparison and eval timing have not been measured.
  - Not restarted automatically.
- **Before resuming pilots:** connect AC power and free RAM (close other apps).

---

## 2026-10-18 — P0 complete (on AC power)

- The user resumed the task and left the session to run unattended ("you have permission"). The scope is unchanged: pilots P0–P4, then **stop at G3**.
- **Memory:** free RAM was 0.9 GB with nothing of ours running. I measured a training process at ~0.9 GB working set before the model, of which ~0.2 GB was raw item dicts. Cuts (commit 118709d):
  - training drops the item dicts after encoding;
  - eval drops the train list after sampling recall items;
  - P3's orchestrator no longer initialises CUDA (~0.6 GB).
  - Code review of that commit: no blocking defects. Its two robustness points (atomic `final.pt` and `p3_val.json`) are applied (I-009).
- **Throughput on AC: 0.059 s/step (82K tok/s)**, ~6× faster than on battery (0.35 s/step). The plan's 13–25 min per 15K-step run now holds (~15 min).
- **Determinism, the main P0 finding:**
  - Runs A and B were **not** bitwise identical: A matched B through step 700, then drifted (max |Δw| 0.012).
  - Their training code is the same; the commits differ only by two `del` lines. A ran on battery with very little free RAM in the earlier session. I could not identify the cause.
  - Follow-up runs: B = C bitwise (default kernels, same conditions) and D = D2 bitwise (deterministic mode). B ≠ D, because deterministic mode picks other kernels; it costs ~7%.
  - **Decision I-008:** every study run, and every evaluation, uses torch's deterministic mode. It is the only mode PyTorch guarantees, and no study run had started yet.
- **Eval timing:** all 11,195 eval items in ~58–67 s per run. Two evaluations of the same run gave byte-identical records. Peak VRAM: 796 MB training, 593 MB eval.
- **Added (I-009):** each run writes `train_stats.json` (wall clock and peak VRAM, as §4.3 requires).
- Full suite: 293 passed, 10 skipped.

---

## 2026-10-18 — P2 interrupted (low system memory)

- P2 (`runs/p2-ref-s0-st30000`) reached step 7,000 of 30,000 on AC power. **Claude Code then stopped it because system memory was critically low** while the session was idle.
- Saved state: checkpoint at step 5,000 (resume loses ~2K steps, about 2 min). Val accuracy so far: 46.3% at 2.5K, 62.1% at 5K (n = 644). These are partial, so S is not set yet.
- Not restarted automatically: the stop notice says to restart only when the user asks. Free RAM after the stop: 1.4 GB of 7.6 GB.
- To resume, either:
  - restart Claude Code with `CLAUDE_CODE_DISABLE_BG_SHELL_PRESSURE_REAP=1` in its environment, or
  - run `scripts/pilots.py p2 p3 p4 p1` from a separate terminal (every run resumes from its checkpoint).

  Closing other memory-heavy apps also helps.
