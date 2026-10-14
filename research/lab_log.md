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
