"""Training protocol (RESEARCH_DESIGN.md §4.3): plan tests 6 (loss mask) and 7 (resume)."""

import json
import math
from pathlib import Path

import pytest
import torch

from mathgen.training import (RunConfig, batch_indices, collate, encode_example, load_tokenizer, lr_at,
                              param_groups, sequence_loss, train)

ROOT = Path(__file__).resolve().parents[1]
needs_data = pytest.mark.skipif(not (ROOT / "data" / "tokenizers" / "digit.json").exists(), reason="data/ not built")

ITEM = {"question": "Compute 12 * 3.", "hints": ["12 × 3 = 36"], "answer": "36"}


def tiny(tmp_path: Path, **kw) -> RunConfig:
    base = dict(exp_id="tiny", tok_arm="digit", arch_arm="ref", seed=0, n_train=256, steps=50, batch_size=8,
                lr=1e-3, device="cpu", precision="fp32", log_every=10, ckpt_every=10, runs_dir=str(tmp_path),
                model_overrides={"n_layers": 2, "d_model": 32, "n_heads": 2, "d_ff": 64})
    base.update(kw)
    return RunConfig(**base)


# ── encoding and loss masking (plan test 6) ──────────────────────────────────────

@needs_data
def test_encode_example_concatenates_prompt_and_target():
    tok = load_tokenizer("digit")
    ids, n_prompt = encode_example(tok, "digit", ITEM, "hints_first")
    assert tok.decode(ids[:n_prompt]) == "[SOLVE] Compute 12 * 3. <SEP><JSON>"
    assert tok.decode(ids[n_prompt:]).endswith("<EOS>") and ids[-1] == tok.eos_id


@needs_data
def test_decoupled_arms_tokenize_numbers_differently_in_prompt_and_target():
    tok = load_tokenizer("dec_db")   # digit-in / bpe-out
    ids, n = encode_example(tok, "dec_db", ITEM, "hints_first")
    prompt_tokens = [tok.vocab[i] for i in ids[:n]]
    target_tokens = [tok.vocab[i] for i in ids[n:]]
    assert "12" not in prompt_tokens and "1" in prompt_tokens
    assert "36" in target_tokens
    ids_bd, n_bd = encode_example(tok, "dec_bd", ITEM, "hints_first")
    assert "12" in [tok.vocab[i] for i in ids_bd[:n_bd]] and "36" not in [tok.vocab[i] for i in ids_bd[n_bd:]]


def test_collate_masks_everything_but_completion_predictions():
    batch = collate([([5, 6, 7, 8, 9], 2), ([5, 6, 7], 1)], pad_id=0)
    assert batch["input"].tolist() == [[5, 6, 7, 8], [5, 6, 0, 0]]
    assert batch["target"].tolist() == [[6, 7, 8, 9], [6, 7, 0, 0]]
    # position t predicts token t+1; only tokens at index >= n_prompt are completion tokens
    assert batch["mask"].tolist() == [[0, 1, 1, 1], [1, 1, 0, 0]]


def test_padded_batch_loss_equals_mean_of_per_example_losses():
    torch.manual_seed(0)
    examples = [([1, 2, 3, 4, 5, 6], 2), ([1, 2, 3], 1), ([4, 4, 4, 4], 3)]
    logits_fn = torch.nn.Embedding(10, 10)
    batched = collate(examples, pad_id=0)
    loss = sequence_loss(logits_fn(batched["input"]), batched["target"], batched["mask"])
    singles = []
    for ex in examples:
        b = collate([ex], pad_id=0)
        singles.append(sequence_loss(logits_fn(b["input"]), b["target"], b["mask"]))
    assert torch.allclose(loss, torch.stack(singles).mean(), atol=1e-6)


def test_prompt_and_padding_logits_do_not_affect_loss():
    b = collate([([1, 2, 3, 4], 2), ([1, 2], 1)], pad_id=0)
    logits = torch.randn(2, 3, 10)
    changed = logits.clone()
    changed[0, 0] += 100.0   # predicts a prompt token
    changed[1, 1:] += 100.0  # padding positions
    assert torch.equal(sequence_loss(logits, b["target"], b["mask"]), sequence_loss(changed, b["target"], b["mask"]))


# ── optimization protocol ─────────────────────────────────────────────────────────

def test_lr_schedule_warmup_then_cosine_to_ten_percent(tmp_path):
    cfg = tiny(tmp_path, steps=1000, lr=1e-3)
    assert lr_at(0, cfg) == pytest.approx(1e-3 / 50)
    assert lr_at(49, cfg) == pytest.approx(1e-3)
    assert lr_at(999, cfg) == pytest.approx(1e-4, rel=1e-3)
    mid = 50 + (1000 - 50) // 2
    assert lr_at(mid, cfg) == pytest.approx(1e-4 + 0.5 * 9e-4 * (1 + math.cos(math.pi * (mid - 50) / 950)), rel=1e-3)


def test_no_weight_decay_on_norms_biases_or_embeddings():
    from mathgen.model import Transformer, arch_config
    m = Transformer(arch_config("learned", vocab_size=50, context_len=64))
    decay, no_decay = param_groups(m, 0.1)
    names = {id(p): n for n, p in m.named_parameters()}
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
    assert all(names[id(p)].endswith(".g") or "embed" in names[id(p)] for p in no_decay["params"])
    assert all(p.dim() >= 2 and "embed" not in names[id(p)] for p in decay["params"])


def test_data_order_is_seeded_and_covers_each_epoch():
    a = [batch_indices(0, 100, s, 10) for s in range(10)]
    assert sorted(i for b in a for i in b) == list(range(100))           # one epoch = a permutation
    assert a == [batch_indices(0, 100, s, 10) for s in range(10)]         # deterministic
    assert a != [batch_indices(1, 100, s, 10) for s in range(10)]         # seed-dependent


# ── full runs (plan test 7) ───────────────────────────────────────────────────────

@needs_data
def test_resume_matches_an_uninterrupted_run(tmp_path):
    full = train(tiny(tmp_path / "a"))
    interrupted = tiny(tmp_path / "b")
    train(interrupted, stop_after=20)
    resumed = train(interrupted)
    w_full = torch.load(full / "final.pt", weights_only=True)["model"]
    w_res = torch.load(resumed / "final.pt", weights_only=True)["model"]
    assert all(torch.equal(w_full[k], w_res[k]) for k in w_full)
    strip = lambda p: [{k: v for k, v in json.loads(l).items() if k not in ("tok_s", "elapsed_s")}
                       for l in (p / "metrics.jsonl").read_text().splitlines()]
    assert strip(full) == strip(resumed)


@needs_data
def test_resume_mid_log_interval_keeps_the_loss_log_exact(tmp_path):
    """stop_after inside a logging interval must not drop or mis-average the interval's losses."""
    full = train(tiny(tmp_path / "a", steps=30))
    interrupted = tiny(tmp_path / "b", steps=30)
    train(interrupted, stop_after=15)
    resumed = train(interrupted)
    losses = lambda p: [json.loads(l)["train_loss"] for l in (p / "metrics.jsonl").read_text().splitlines()]
    assert losses(full) == pytest.approx(losses(resumed), rel=1e-6)


@needs_data
def test_existing_run_with_a_different_config_is_refused(tmp_path):
    train(tiny(tmp_path, steps=10))
    with pytest.raises(ValueError, match="config"):
        train(tiny(tmp_path, steps=10, lr=5e-4))


@needs_data
def test_run_record_has_provenance(tmp_path):
    run = train(tiny(tmp_path, steps=10))
    rec = json.loads((run / "record.json").read_text())
    assert rec["config"]["exp_id"] == "tiny"
    assert {"git_commit", "git_dirty", "python", "torch", "device"} <= set(rec["env"])
    assert rec["data"]["train_subset_sha256"] and rec["data"]["tokenizer_sha256"]
    assert (run / "pip_freeze.txt").read_text().strip()
    losses = [json.loads(l)["train_loss"] for l in (run / "metrics.jsonl").read_text().splitlines()]
    assert all(math.isfinite(x) for x in losses)
