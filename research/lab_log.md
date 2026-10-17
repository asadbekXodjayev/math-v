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
