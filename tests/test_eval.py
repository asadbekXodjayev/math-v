"""Decoding (plan test 8) and evaluation scoring (RESEARCH_DESIGN.md §5)."""

from pathlib import Path

import pytest
import torch

from mathgen.eval.decode import greedy_decode
from mathgen.eval.run import parse_completion, score_item, train_recall_items
from mathgen.model import Transformer, arch_config

ROOT = Path(__file__).resolve().parents[1]


def model(arm: str = "ref") -> Transformer:
    torch.manual_seed(0)
    return Transformer(arch_config(arm, vocab_size=40, context_len=64)).eval()


def naive_greedy(m, prompt, eos_id, context_len):
    """Reference: full forward every step, one sequence at a time, no cache."""
    seq = list(prompt)
    out = []
    while len(seq) < context_len:
        nxt = int(m(torch.tensor([seq]))[0, -1].argmax())
        out.append(nxt)
        seq.append(nxt)
        if nxt == eos_id:
            break
    return out


@pytest.mark.parametrize("arm", ["ref", "learned", "none"])
def test_batched_cached_greedy_equals_naive_greedy(arm):
    """Plan test 8: batched == batch-1 and KV-cache == no-cache (fp32)."""
    m = model(arm)
    g = torch.Generator().manual_seed(1)
    prompts = [torch.randint(1, 40, (n,), generator=g).tolist() for n in (3, 5, 5, 7, 3, 9, 5)]
    eos = 2
    expected = [naive_greedy(m, p, eos, 64) for p in prompts]
    assert greedy_decode(m, prompts, eos_id=eos, context_len=64, batch_size=3) == expected
    assert greedy_decode(m, prompts, eos_id=eos, context_len=64, batch_size=1) == expected


# ── completion parsing and scoring ──────────────────────────────────────────────

GOLD = {"question": "Evaluate: 4 × (3 + 5)", "hints": ["Parentheses first: 3 + 5 = 8", "Then multiply: 4 × 8 = 32"],
        "answer": "32", "atype": "int", "template": "arith_pemdas"}


def test_parse_completion_reads_answer_and_hints_in_either_order():
    hf = parse_completion('{"hints": ["a = 1"], "answerKey": "1"}<EOS>')
    af = parse_completion('{"answerKey": "1", "hints": ["a = 1"]}<EOS>')
    for p in (hf, af):
        assert p["valid_json"] and not p["truncated"] and p["answer"] == "1" and p["hints"] == ["a = 1"]


@pytest.mark.parametrize("text, truncated, valid", [
    ('{"hints": ["a = 1"], "answerKey": "1"}', True, True),     # no <EOS> within context
    ('{"hints": ["a = 1"], "answerKey": 1}<EOS>', False, False),  # answerKey must be a string
    ('{"hints": ["a = 1"]}<EOS>', False, False),                  # missing key
    ("not json<EOS>", False, False),
])
def test_parse_completion_flags_failures(text, truncated, valid):
    p = parse_completion(text)
    assert p["truncated"] == truncated and p["valid_json"] == valid


def test_score_counts_every_failure_as_wrong():
    for text in ('{"hints": [], "answerKey": "32"}', "garbage<EOS>", '{"hints": [], "answerKey": "thirty-two"}<EOS>'):
        assert score_item(GOLD, text)["correct"] is False
    assert score_item(GOLD, '{"hints": [], "answerKey": "32"}')["error"] == "truncated"
    assert score_item(GOLD, "garbage<EOS>")["error"] == "format"
    assert score_item(GOLD, '{"hints": [], "answerKey": "thirty-two"}<EOS>')["error"] == "unparseable"


def test_score_classifies_copy_and_scratchpad_errors():
    ok = '{"hints": ["Parentheses first: 3 + 5 = 8", "Then multiply: 4 × 8 = 32"], "answerKey": "32"}<EOS>'
    copy_err = '{"hints": ["Parentheses first: 3 + 5 = 8", "Then multiply: 4 × 8 = 32"], "answerKey": "33"}<EOS>'
    scratch_err = '{"hints": ["Parentheses first: 3 + 5 = 8", "Then multiply: 4 × 8 = 33"], "answerKey": "33"}<EOS>'
    assert score_item(GOLD, ok)["correct"] is True
    assert score_item(GOLD, ok)["error"] == "correct" and score_item(GOLD, ok)["copy_consistent"] is True
    assert score_item(GOLD, copy_err)["error"] == "copy_error"
    assert score_item(GOLD, scratch_err)["error"] == "scratchpad_error"


def test_unreduced_fraction_is_correct_but_not_exact():
    gold = {**GOLD, "answer": "5/6", "atype": "rational", "hints": ["x"]}
    s = score_item(gold, '{"hints": ["x"], "answerKey": "10/12"}<EOS>')
    assert s["correct"] is True and s["exact"] is False


# ── end to end: train a tiny model, evaluate every split ─────────────────────────

@pytest.mark.skipif(not (ROOT / "data" / "tokenizers" / "digit.json").exists(), reason="data/ not built")
def test_eval_run_writes_every_split_and_a_summary(tmp_path):
    import json
    from mathgen.eval.run import EVAL_SPLITS, eval_run
    from mathgen.training import RunConfig, train
    cfg = RunConfig(exp_id="e2e", n_train=128, steps=20, batch_size=8, device="cpu", precision="fp32",
                    log_every=10, ckpt_every=10, runs_dir=str(tmp_path),
                    model_overrides={"n_layers": 1, "d_model": 32, "n_heads": 2, "d_ff": 64})
    run = train(cfg)
    summary = eval_run(run, device="cpu", limit=12)
    assert set(summary["splits"]) == set(EVAL_SPLITS) | {"train_recall"}
    assert summary["primary_endpoint"] is not None and 0.0 <= summary["primary_endpoint"] <= 1.0
    recs = [json.loads(l) for l in (run / "eval" / "test_id_unseen.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(recs) == 12 and all(r["gold_nll"] > 0 and r["error"] in
                                   ("correct", "truncated", "format", "unparseable", "copy_error",
                                    "scratchpad_error", "wrong") for r in recs)
    assert all(isinstance(r["div_in"], bool) for r in recs)
    assert json.loads((run / "eval.json").read_text())["exp_id"] == "e2e"


@pytest.mark.skipif(not (ROOT / "data" / "tokenizers" / "digit.json").exists(), reason="data/ not built")
def test_val_callback_logs_val_accuracy_during_training(tmp_path):
    import json
    from mathgen.eval.run import val_callback
    from mathgen.training import RunConfig, train
    cfg = RunConfig(exp_id="valcb", n_train=64, steps=20, batch_size=8, device="cpu", precision="fp32",
                    log_every=10, ckpt_every=10, eval_every=10, runs_dir=str(tmp_path),
                    model_overrides={"n_layers": 1, "d_model": 32, "n_heads": 2, "d_ff": 64})
    run = train(cfg, on_eval=val_callback(cfg, limit=8))
    lines = [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines()]
    val = [l for l in lines if "val_acc" in l]
    assert [l["step"] for l in val] == [10, 20] and all(0.0 <= l["val_acc"] <= 1.0 for l in val)


# ── train-recall split (built per run) ───────────────────────────────────────────

@pytest.mark.skipif(not (ROOT / "data" / "train" / "seed0.jsonl").exists(), reason="data/ not built")
def test_train_recall_items_come_from_the_runs_training_set():
    from mathgen.data.build import read_jsonl
    train = read_jsonl(ROOT / "data" / "train" / "seed0.jsonl")[:2000]
    recall = train_recall_items(train, per_template=10, seed=0)
    keys = {i["key"] for i in train}
    assert all(r["key"] in keys and r["exposure"] >= 1 for r in recall)
    assert len({r["key"] for r in recall}) == len(recall)
    counts = {}
    for r in recall:
        counts[r["template"]] = counts.get(r["template"], 0) + 1
    assert len(counts) == 23 and max(counts.values()) <= 10
    assert recall == train_recall_items(train, per_template=10, seed=0)
