"""
One training run (RESEARCH_DESIGN.md §4.3), with full resume state and a provenance record.

runs/<exp_id>/
    record.json       config, environment (git commit and dirty flag, torch, GPU), data/tokenizer hashes, model size
    pip_freeze.txt
    metrics.jsonl     per log interval: mean train loss (fixes audit F2), lr, grad norm, real tok/s (no padding)
    ckpt.pt           latest full state: model, optimizer, scheduler, step, RNG
    final.pt          final weights; the run is complete once this exists
"""

import hashlib
import json
import platform
import random
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .. import compat  # noqa: F401  (must precede torch)

import torch

from ..data.build import read_jsonl
from ..model import ModelConfig, Transformer, arch_config
from .data import TOK_ARMS, batch_indices, collate, encode_example, load_tokenizer
from .optim import lr_at, param_groups, sequence_loss

REPO = Path(__file__).resolve().parents[3]
# Fields that may differ when an existing run is resumed or reused; any other difference is refused.
RESUMABLE_FIELDS = {"device", "eval_every", "notes"}


@dataclass
class RunConfig:
    exp_id: str
    tok_arm: str = "digit"
    arch_arm: str = "ref"
    order: str = "hints_first"
    seed: int = 0
    n_train: int = 128_000
    steps: int = 15_000
    batch_size: int = 64
    lr: float = 1e-3
    warmup_frac: float = 0.05
    min_lr_frac: float = 0.1
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    precision: str = "bf16"      # bf16 | fp32
    deterministic: bool = True   # torch deterministic algorithms (decisions.md I-008)
    device: str = "cuda"
    log_every: int = 100
    ckpt_every: int = 2500
    eval_every: int = 0          # >0: val-split accuracy at these steps (pilot P2)
    context_len: int = 256
    data_dir: str = "data"
    runs_dir: str = "runs"
    model_overrides: dict = field(default_factory=dict)
    notes: str = ""

    def __post_init__(self):
        if self.tok_arm not in TOK_ARMS:
            raise ValueError(f"unknown tok_arm {self.tok_arm!r}")
        if self.ckpt_every % self.log_every:
            raise ValueError("ckpt_every must be a multiple of log_every (so resumed logs line up)")


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO / p


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unavailable"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _first_lines(path: Path, n: int) -> bytes:
    with open(path, "rb") as f:
        return b"".join(line for _, line in zip(range(n), f))


def _write_record(cfg: RunConfig, run_dir: Path, model: Transformer, tok_file: Path, train_file: Path) -> None:
    data_dir = _resolve(cfg.data_dir)
    record = {
        "config": asdict(cfg),
        "env": {
            "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
            "python": sys.version.split()[0], "platform": platform.platform(),
            "torch": torch.__version__, "cuda": torch.version.cuda, "device": cfg.device,
            "gpu": torch.cuda.get_device_name(0) if cfg.device == "cuda" and torch.cuda.is_available() else None,
        },
        "data": {
            "build_manifest_sha256": _sha256_file(data_dir / "manifests" / "build.json"),
            "train_file": str(train_file.relative_to(data_dir)),
            "train_subset_sha256": hashlib.sha256(_first_lines(train_file, cfg.n_train)).hexdigest(),
            "tokenizer_file": tok_file.name, "tokenizer_sha256": _sha256_file(tok_file),
        },
        "model": {"config": asdict(model.cfg), "params": model.num_params(),
                  "non_embedding_params": model.non_embedding_params()},
    }
    (run_dir / "record.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    (run_dir / "pip_freeze.txt").write_text(freeze, encoding="utf-8", newline="\n")


def build_model(cfg: RunConfig, vocab_size: int) -> Transformer:
    base = arch_config(cfg.arch_arm, vocab_size=vocab_size, context_len=cfg.context_len)
    return Transformer(ModelConfig(**{**asdict(base), **cfg.model_overrides}))


def _check_same_config(cfg: RunConfig, run_dir: Path) -> None:
    record = run_dir / "record.json"
    if not record.exists():
        return
    stored = json.loads(record.read_text(encoding="utf-8"))["config"]
    diffs = {k: (stored.get(k), v) for k, v in asdict(cfg).items() if k not in RESUMABLE_FIELDS and stored.get(k) != v}
    if diffs:
        raise ValueError(f"run {cfg.exp_id!r} already exists with a different config (stored, requested): {diffs}")


def _save_ckpt(path: Path, step: int, model, opt, sched, interval: dict) -> None:
    state = {"step": step, "interval": interval,
             "model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
             "rng_torch": torch.get_rng_state(), "rng_py": random.getstate(),
             "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)   # atomic: a crash mid-save never leaves a corrupt checkpoint


@contextmanager
def deterministic_algorithms(enabled: bool = True):
    """torch's deterministic mode inside the block; the previous setting is restored after it."""
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(enabled)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous)


def train(cfg: RunConfig, stop_after: int | None = None, on_eval=None) -> Path:
    """Run (or resume) to completion. `stop_after`: stop at that global step after checkpointing
    (used to test resume). `on_eval(model, tok, step) -> dict` is called every `eval_every` steps."""
    with deterministic_algorithms(cfg.deterministic):
        return _train(cfg, stop_after, on_eval)


def _train(cfg: RunConfig, stop_after: int | None, on_eval) -> Path:
    run_dir = _resolve(cfg.runs_dir) / cfg.exp_id
    _check_same_config(cfg, run_dir)
    if (run_dir / "final.pt").exists():
        return run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    session_start = time.perf_counter()
    device = torch.device(cfg.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    data_dir = _resolve(cfg.data_dir)

    tok = load_tokenizer(cfg.tok_arm, data_dir)
    train_file = data_dir / "train" / f"seed{cfg.seed}.jsonl"
    items = read_jsonl(train_file)[:cfg.n_train]
    if len(items) < cfg.n_train:
        raise ValueError(f"{train_file} has only {len(items)} examples")
    examples = [encode_example(tok, cfg.tok_arm, it, cfg.order) for it in items]
    del items   # ~200 MB of dicts that training never reads again (the study laptop has 7.6 GB of RAM)
    if (longest := max(len(ids) for ids, _ in examples)) > cfg.context_len:
        raise ValueError(f"an example has {longest} tokens > context {cfg.context_len}")

    torch.manual_seed(cfg.seed)
    model = build_model(cfg, len(tok.vocab)).to(device)
    opt = torch.optim.AdamW(param_groups(model, cfg.weight_decay), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), eps=1e-8,
                            fused=device.type == "cuda")
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_at(s, cfg) / cfg.lr)

    ckpt_path, metrics_path = run_dir / "ckpt.pt", run_dir / "metrics.jsonl"
    start, interval = 0, {"loss_sum": 0.0, "steps": 0, "tokens": 0}
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        torch.set_rng_state(state["rng_torch"])
        random.setstate(state["rng_py"])
        if state["rng_cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["rng_cuda"])
        start, interval = state["step"], state["interval"]
        kept = [l for l in metrics_path.read_text(encoding="utf-8").splitlines() if json.loads(l)["step"] <= start]
        metrics_path.write_text("".join(l + "\n" for l in kept), encoding="utf-8", newline="\n")
    else:
        _write_record(cfg, run_dir, model, data_dir / "tokenizers" / f"{TOK_ARMS[cfg.tok_arm][0]}.json", train_file)
        metrics_path.write_text("", encoding="utf-8")

    use_bf16 = cfg.precision == "bf16" and device.type == "cuda"
    # Loss is accumulated on the device and read only when a line is logged (no per-step sync).
    loss_acc = torch.tensor(interval["loss_sum"], device=device)
    n_steps, tokens, t0 = interval["steps"], interval["tokens"], time.perf_counter()
    model.train()
    with open(metrics_path, "a", encoding="utf-8", newline="\n") as log:
        for step in range(start, cfg.steps):
            if stop_after is not None and step >= stop_after:
                _save_ckpt(ckpt_path, step, model, opt, sched,
                           {"loss_sum": loss_acc.item(), "steps": n_steps, "tokens": tokens})
                return run_dir
            batch = collate([examples[i] for i in batch_indices(cfg.seed, cfg.n_train, step, cfg.batch_size)],
                            tok.pad_id)
            tokens += int((batch["input"] != tok.pad_id).sum())   # counted on the CPU copy
            inp, tgt, mask = (batch[k].to(device, non_blocking=True) for k in ("input", "target", "mask"))
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                logits = model(inp)
            loss = sequence_loss(logits, tgt, mask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            lr = opt.param_groups[0]["lr"]
            opt.step()
            sched.step()
            loss_acc += loss.detach()
            n_steps += 1

            done = step + 1
            if done % cfg.log_every == 0:
                elapsed = time.perf_counter() - t0
                log.write(json.dumps({"step": done, "train_loss": loss_acc.item() / n_steps, "lr": lr,
                                      "grad_norm": float(grad_norm), "tok_s": tokens / elapsed,
                                      "elapsed_s": elapsed}) + "\n")
                log.flush()
                loss_acc.zero_()
                n_steps, tokens, t0 = 0, 0, time.perf_counter()
            if cfg.eval_every and on_eval is not None and done % cfg.eval_every == 0:
                model.eval()
                log.write(json.dumps({"step": done, **on_eval(model, tok, done)}) + "\n")
                log.flush()
                model.train()
                t0 = time.perf_counter()   # eval time is not charged to training throughput
            if done % cfg.ckpt_every == 0 or done == cfg.steps:
                _save_ckpt(ckpt_path, done, model, opt, sched,
                           {"loss_sum": loss_acc.item(), "steps": n_steps, "tokens": tokens})

    logged = [json.loads(l) for l in metrics_path.read_text(encoding="utf-8").splitlines()]
    stats = {   # §4.3: wall clock and peak VRAM. train_wall_s sums the logged intervals over every session
        "train_wall_s": sum(m.get("elapsed_s", 0) for m in logged),
        "resumed_from_step": start,
        "last_session_wall_s": time.perf_counter() - session_start,
        "peak_vram_mb_last_session": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
    }
    (run_dir / "train_stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8", newline="\n")
    tmp = run_dir / "final.pt.tmp"   # atomic, like checkpoints: final.pt existing means the run is complete
    torch.save({"model": model.state_dict(), "model_config": asdict(model.cfg), "run": asdict(cfg)}, tmp)
    tmp.replace(run_dir / "final.pt")
    return run_dir
