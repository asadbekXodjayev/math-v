"""
MathGen v2 — GPU-optimized training for RTX 3050 4GB.

Optimizations:
  - fp16 AMP (torch.cuda.amp)          → halves VRAM
  - Gradient checkpointing             → cuts activation memory ~60%
  - Gradient accumulation (8 steps)    → effective batch = 64 at 4GB
  - AdamW + cosine warmup LR
  - torch.compile (PyTorch 2+)         → ~30% speed boost

Usage:
  python src/train.py                   # auto-detects CUDA
  python src/train.py --epochs 50
  python src/train.py --resume          # continue from checkpoint
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast

from model import MathGen, MathGenConfig, build_model
from tokenizer import BPETokenizer
from dataset import build_all_samples, build_dataloader

CKPT_DIR = Path(__file__).parent.parent / "checkpoints"
TOK_PATH = str(CKPT_DIR / "tokenizer.json")
BEST_PT  = CKPT_DIR / "best.pt"
LAST_PT  = CKPT_DIR / "latest.pt"
LOG_PATH = CKPT_DIR / "train_log.jsonl"


def cosine_warmup_lr(step: int, warmup: int, total: int, max_lr: float, min_lr: float) -> float:
    if step < warmup:
        return max_lr * step / max(warmup, 1)
    t = (step - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + torch.cos(torch.tensor(t * 3.14159)).item())


def log_line(path: Path, data: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(data) + "\n")


def train(args: argparse.Namespace) -> None:
    import os
    import sys
    # Force all output to flush immediately
    sys.stdout.reconfigure(line_buffering=True)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"\n{'='*60}", flush=True)
    print(f"MathGen Training  |  device={device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", flush=True)
    print(f"{'='*60}\n", flush=True)

    CKPT_DIR.mkdir(exist_ok=True)

    # ── 1. Data ───────────────────────────────────────────────────────────────
    print("=== Building dataset ===")
    all_samples = build_all_samples(n_synthetic_per_topic=args.n_per_topic)
    split = int(len(all_samples) * 0.92)
    train_samples = all_samples[:split]
    val_samples   = all_samples[split:]

    # ── 2. Tokenizer ──────────────────────────────────────────────────────────
    print("\n=== Tokenizer ===")
    tok = BPETokenizer()
    if Path(TOK_PATH).exists() and not args.fresh:
        tok.load(TOK_PATH)
        print(f"Loaded existing tokenizer ({len(tok)} tokens)")
    else:
        tok.train(train_samples, vocab_size=args.vocab_size)
        tok.save(TOK_PATH)

    # ── 3. Model ──────────────────────────────────────────────────────────────
    print("\n=== Model ===")
    model, cfg = build_model(vocab_size=len(tok), device=str(device))
    cfg.use_grad_ckpt = args.grad_ckpt
    print(f"Gradient checkpointing: {'ON' if cfg.use_grad_ckpt else 'OFF (faster, uses more VRAM)'}")

    start_epoch  = 0
    best_val     = float("inf")
    global_step  = 0

    if LAST_PT.exists() and not args.fresh:
        ckpt = torch.load(LAST_PT, map_location=device)
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt["epoch"] + 1
        best_val    = ckpt.get("best_val", float("inf"))
        global_step = ckpt.get("global_step", 0)
        print(f"Resumed from epoch {start_epoch}  (best_val={best_val:.4f})")

    # torch.compile disabled — RTX 3050 has limited SMs and causes inductor crashes
    print("torch.compile() skipped (RTX 3050 SM count too low for inductor)")

    # ── 4. Optimizer ──────────────────────────────────────────────────────────
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=False,  # fused AdamW not stable on all CUDA setups
    )
    scaler = GradScaler("cuda", enabled=device.type == "cuda")

    train_loader = build_dataloader(
        train_samples, tok, cfg.context_len,
        batch_size=args.batch_size, n_workers=0,  # Windows: multiprocess workers crash with CUDA
    )
    val_loader = build_dataloader(
        val_samples, tok, cfg.context_len,
        batch_size=args.batch_size, shuffle=False,
    )

    total_steps = len(train_loader) * args.epochs // args.grad_accum
    warmup_steps = total_steps // 10
    print(f"\nSteps: {total_steps}  |  Warmup: {warmup_steps}  |  Accum: {args.grad_accum}")
    print(f"Effective batch: {args.batch_size * args.grad_accum}\n")

    # ── 5. Training loop ──────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        accum_loss = 0.0
        t0 = time.time()

        optimizer.zero_grad(set_to_none=True)

        for local_step, (x, y) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            # LR update
            lr = cosine_warmup_lr(global_step, warmup_steps, total_steps, args.lr, args.lr * 0.05)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # Forward + backward with fp16 — resilient to a transient OOM on a 4GB
            # GPU shared with the display/other apps: drop the micro-batch, clear the
            # cache, and keep going instead of killing the whole run.
            try:
                with autocast("cuda", enabled=device.type == "cuda"):
                    _, loss = model(x, y)
                    loss = loss / args.grad_accum
                scaler.scale(loss).backward()
                accum_loss += loss.item()
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if "out of memory" not in str(e).lower():
                    raise
                optimizer.zero_grad(set_to_none=True)
                del x, y
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                print(f"  [OOM] skipped micro-batch near step {global_step} "
                      f"(transient VRAM spike)", flush=True)
                continue

            # Optimizer step every grad_accum batches
            if (local_step + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                epoch_loss += accum_loss
                global_step += 1
                accum_loss = 0.0

                if global_step % 25 == 0:
                    elapsed = time.time() - t0
                    tok_per_sec = args.batch_size * args.grad_accum * cfg.context_len * global_step / elapsed
                    print(
                        f"E{epoch+1} step {global_step} | "
                        f"loss={epoch_loss/max(global_step,1):.4f} | "
                        f"lr={lr:.2e} | "
                        f"{tok_per_sec/1e3:.1f}k tok/s"
                    )

                    if device.type == "cuda":
                        used = torch.cuda.memory_allocated() / 1e9
                        total = torch.cuda.get_device_properties(0).total_memory / 1e9
                        print(f"           VRAM: {used:.2f}/{total:.2f} GB")

                # Periodically release cached blocks to fight VRAM fragmentation —
                # this is what caused the OOM at epoch 5 on a 4GB card (the
                # expandable_segments allocator is unsupported on this platform).
                if device.type == "cuda" and global_step % 100 == 0:
                    torch.cuda.empty_cache()

        # ── Validation ────────────────────────────────────────────────────────
        if device.type == "cuda":
            torch.cuda.empty_cache()
        model.eval()
        val_loss = 0.0
        with torch.no_grad(), autocast("cuda", enabled=device.type == "cuda"):
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                _, loss = model(x, y)
                val_loss += loss.item()
        val_loss /= max(len(val_loader), 1)
        if device.type == "cuda":
            torch.cuda.empty_cache()   # release validation memory before next epoch

        t_loss = epoch_loss / max(global_step, 1)
        print(f"\n{'─'*55}")
        print(f"Epoch {epoch+1}/{args.epochs}  train={t_loss:.4f}  val={val_loss:.4f}  lr={lr:.2e}")
        print(f"{'─'*55}\n")

        # Checkpointing
        state = {
            "model":       model.state_dict(),
            "epoch":       epoch,
            "val_loss":    val_loss,
            "best_val":    best_val,
            "global_step": global_step,
            "cfg":         vars(cfg),
        }
        torch.save(state, LAST_PT)
        log_line(LOG_PATH, {"epoch": epoch+1, "train_loss": t_loss, "val_loss": val_loss, "lr": lr})

        if val_loss < best_val:
            best_val = val_loss
            torch.save(state, BEST_PT)
            print(f"  ✓ New best saved  (val={best_val:.4f})\n")

    print(f"\nTraining complete — best val_loss={best_val:.4f}")
    print(f"Model: {BEST_PT}")
    print("\nNext step: python src/server.py")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",       type=int,   default=40)
    p.add_argument("--batch-size",   type=int,   default=32,
                   help="Per-GPU batch size (effective = batch × grad_accum)")
    p.add_argument("--grad-accum",   type=int,   default=2,
                   help="Gradient accumulation steps → effective batch = 64")
    p.add_argument("--grad-ckpt",    action="store_true",
                   help="Enable gradient checkpointing (saves VRAM, slower). "
                        "Off by default — only needed for larger models on 4GB.")
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--vocab-size",   type=int,   default=16384)
    p.add_argument("--n-per-topic",  type=int,   default=2000,
                   help="Synthetic samples per (topic, difficulty)")
    p.add_argument("--fresh",        action="store_true",
                   help="Ignore checkpoints, train from scratch")
    p.add_argument("--cpu",          action="store_true",
                   help="Force CPU training (slower but no VRAM issues)")
    args = p.parse_args()

    if args.cpu:
        # Force CPU by hiding CUDA from torch
        import os; os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("Forced CPU mode", flush=True)

    train(args)
