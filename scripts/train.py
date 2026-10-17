"""
Train one run (resumes automatically if runs/<exp_id>/ckpt.pt exists).

Usage (from mathvis-model/):
    .venv/Scripts/python.exe scripts/train.py --config configs/arms/ref.yaml --seed 0 [--steps 15000]
        [--n-train 128000] [--lr 1e-3] [--exp-id NAME] [--eval-every 2500] [--then-eval]

The exp_id defaults to <arm>-s<seed>-n<N/1000>k-<steps/1000>k[-lr<lr>]. Every flag except
--then-eval overrides the arm config; overrides are stored in the run's record.json.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from mathgen.training import RunConfig, train  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--n-train", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--exp-id")
    ap.add_argument("--eval-every", type=int, default=0, help="val accuracy every K steps (pilot P2)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--then-eval", action="store_true", help="evaluate all splits after training")
    args = ap.parse_args()

    arm = Path(args.config).stem
    fields = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    fields.update({k: v for k, v in {"steps": args.steps, "n_train": args.n_train, "lr": args.lr}.items() if v is not None})
    fields.update(seed=args.seed, device=args.device, eval_every=args.eval_every)
    cfg = RunConfig(exp_id="", **fields)
    lr_tag = f"-lr{args.lr:g}" if args.lr is not None else ""
    cfg.exp_id = args.exp_id or f"{arm}-s{cfg.seed}-n{cfg.n_train // 1000}k-{cfg.steps // 1000}k{lr_tag}"

    on_eval = None
    if cfg.eval_every:
        from mathgen.eval.run import val_callback
        on_eval = val_callback(cfg)
    run_dir = train(cfg, on_eval=on_eval)
    print(f"trained -> {run_dir}")
    if args.then_eval:
        from mathgen.eval.run import eval_run
        summary = eval_run(run_dir, device=args.device)
        print(f"primary endpoint (G-set test-ID-unseen math-correct): {summary['primary_endpoint']:.4f}")


if __name__ == "__main__":
    main()
