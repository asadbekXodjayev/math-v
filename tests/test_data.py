"""Data v1 build: hash pools, blocks, determinism, split integrity (plan tests 1, 3, 4)."""

import copy
import json
from pathlib import Path

import pytest

from mathgen.data.build import build, read_jsonl
from mathgen.data.formats import prompt_text, target_text, tchar_len
from mathgen.data.splits import BLOCKS, bucket, in_block
from mathgen.eval.oracle import solve

ROOT = Path(__file__).resolve().parents[1]


def small(config: dict) -> dict:
    cfg = copy.deepcopy(config)
    cfg["split_sizes"] = {"test_id_unseen": 20, "val": 5, "block": 20, "ood_same": 10, "ood_longer": 10}
    cfg["tokenizer_fit"] = {"n": 2000, "stream_seed": 0}
    cfg["train"] = {"seeds": [0, 1], "n_max": 3000, "n_subsets": [500, 1000, 3000]}
    return cfg


@pytest.fixture(scope="module")
def built(tmp_path_factory, config):
    out = tmp_path_factory.mktemp("data")
    summary = build(small(config), out, config_sha256="test")
    return out, summary


# ── units ────────────────────────────────────────────────────────────────────────

def test_bucket_is_sha256_prefix_mod_modulus():
    # sha256("abc") = ba7816bf8f01cfea...
    assert bucket("abc", 1000) == int("ba7816bf8f01cfea", 16) % 1000


@pytest.mark.parametrize("block_id, template, inside, outside", [
    ("B1", "arith_integer", {"a": 20, "b": 30, "op": "*"}, {"a": 20, "b": 30, "op": "+"}),
    ("B2", "algebra_linear", {"a": 3, "x": 7, "b": 0}, {"a": 3, "x": 6, "b": 0}),
    ("B3", "func_evaluate", {"a": 1, "b": 0, "c": 0, "x": -3}, {"a": 1, "b": 0, "c": 0, "x": 3}),
    ("B4", "mat_det", {"a11": 7, "a12": 9, "a21": 8, "a22": 7}, {"a11": 7, "a12": 9, "a21": 6, "a22": 7}),
    ("B5", "vec_dot", {"u0": 10, "u1": 1, "v0": 12, "v1": 1}, {"u0": 9, "u1": 1, "v0": 12, "v1": 1}),
    ("B6", "arith_fractions", {"n1": 1, "d1": 7, "n2": 1, "d2": 2}, {"n1": 1, "d1": 2, "n2": 1, "d2": 7}),
    ("B7", "deriv_polynomial", {"a": 2, "n": 4, "b": 2, "m": 2, "c": 1}, {"a": 2, "n": 4, "b": 2, "m": 1, "c": 1}),
])
def test_block_regions_match_preregistration(block_id, template, inside, outside):
    assert BLOCKS[block_id][0] == template
    assert in_block(template, inside) == block_id
    assert in_block(template, outside) is None


def test_formats_use_no_ascii_escapes_and_answer_order():
    hints, answer = ["2 × 3 = 6"], "6"
    hf = target_text(hints, answer, "hints_first")
    af = target_text(hints, answer, "answer_first")
    assert "\\u" not in hf and "×" in hf  # audit F8
    assert hf.index('"hints"') < hf.index('"answerKey"')
    assert af.index('"answerKey"') < af.index('"hints"')
    assert hf.endswith("<EOS>") and prompt_text("Compute 2 * 3.") == "[SOLVE] Compute 2 * 3. <SEP><JSON>"


def test_tchar_len_counts_each_special_token_once():
    assert tchar_len("[SOLVE] ab <SEP><JSON>") == len(" ab ") + 3


# ── build ────────────────────────────────────────────────────────────────────────

def test_build_is_deterministic(tmp_path, config, built):
    """Plan test 3: the same config gives byte-identical files and a byte-identical manifest."""
    out, _ = built
    build(small(config), tmp_path, config_sha256="test")
    assert (tmp_path / "manifests" / "build.json").read_bytes() == (out / "manifests" / "build.json").read_bytes()


def test_block_regions_never_reach_training_even_for_small_templates(config):
    """A block template in the M-set would otherwise be put wholesale into the train pool."""
    from mathgen.data.splits import pool
    ops = {"a": 2, "x": 7, "b": 0}
    assert pool("algebra_linear", "k", ops, config, m_set={"algebra_linear"}) == "block"


def test_build_rejects_block_templates_outside_the_g_set(tmp_path, config):
    cfg = small(config)
    cfg["templates"] = dict(cfg["templates"])
    cfg["templates"]["algebra_linear"] = {**cfg["templates"]["algebra_linear"], "id": {"a": [2, 3], "x": [1, 10], "b": [0, 5]}}
    with pytest.raises(ValueError, match="block"):
        build(cfg, tmp_path, config_sha256="test")


def test_manifest_records_cells_smaller_than_requested(built):
    _, summary = built
    short = summary["shortfalls"]
    # e.g. limit_factor has only 3 same-length OOD instances (c in 7..9) while 10 were requested
    assert short["eval/ood_same.jsonl"]["limit_factor"] == {"requested": 10, "got": 3}
    assert all(v["got"] < v["requested"] for cells in short.values() for v in cells.values())


def test_code_hash_ignores_line_endings(tmp_path):
    from mathgen.data.build import normalized_sha256
    assert normalized_sha256(b"a\r\nb\r\n") == normalized_sha256(b"a\nb\n")


def test_every_built_item_agrees_with_the_oracle(built):
    """Plan test 1 on every built split."""
    out, summary = built
    n = 0
    for rel in summary["files"]:
        for item in read_jsonl(out / rel):
            assert solve(item["question"]) == (item["answer"], item["atype"]), item["question"]
            n += 1
    assert n == summary["oracle_agreement"]["checked"] > 0


def test_eval_splits_come_from_their_pools(built, config):
    out, _ = built
    pools = config["pools"]
    for item in read_jsonl(out / "eval" / "test_id_unseen.jsonl"):
        assert item["set"] == "G" and item["bucket"] >= pools["val_below"] and item["block_id"] is None
    for item in read_jsonl(out / "eval" / "val.jsonl"):
        assert item["set"] == "G" and pools["train_below"] <= item["bucket"] < pools["val_below"]
    for item in read_jsonl(out / "eval" / "block.jsonl"):
        assert item["block_id"] == in_block(item["template"], item["operands"]) is not None


def test_no_eval_instance_appears_in_any_training_set(built):
    """Plan test 4: no key shared between train and test/val/block; block regions absent from train."""
    out, summary = built
    held_out = {i["key"] for s in ("test_id_unseen", "val", "block") for i in read_jsonl(out / "eval" / f"{s}.jsonl")}
    for rel in [f for f in summary["files"] if f.startswith("train/") or f.startswith("fit/")]:
        for item in read_jsonl(out / rel):
            assert item["key"] not in held_out
            assert in_block(item["template"], item["operands"]) is None


def test_m_set_only_in_training_and_ood(built, config):
    out, summary = built
    m_set = set(summary["m_set"])
    for s in ("test_id_unseen", "val", "block"):
        assert not m_set & {i["template"] for i in read_jsonl(out / "eval" / f"{s}.jsonl")}
    assert m_set <= {i["template"] for i in read_jsonl(out / "train" / "seed0.jsonl")}


def test_training_templates_are_uniform(built):
    out, _ = built
    items = read_jsonl(out / "train" / "seed0.jsonl")
    counts = {}
    for i in items:
        counts[i["template"]] = counts.get(i["template"], 0) + 1
    expected = len(items) / 23
    assert len(counts) == 23 and all(abs(c - expected) < 5 * expected ** 0.5 for c in counts.values())


def test_seeds_give_different_training_sets(built):
    out, _ = built
    keys0 = [i["key"] for i in read_jsonl(out / "train" / "seed0.jsonl")]
    keys1 = [i["key"] for i in read_jsonl(out / "train" / "seed1.jsonl")]
    assert keys0 != keys1


def test_manifest_records_context_length_rule(built):
    _, summary = built
    ctx = summary["context"]
    assert ctx["max_tchar_len"] > 0
    assert ctx["context_length"] >= max(256, ctx["max_tchar_len"])
    assert ctx["context_length"] & (ctx["context_length"] - 1) == 0  # power of two


# ── the real build in data/ (skipped until scripts/build_data.py has been run) ────

def test_built_data_matches_committed_manifest():
    summary_path = ROOT / "data" / "manifests" / "build.json"
    if not summary_path.exists() or not (ROOT / "data" / "eval").exists():
        pytest.skip("data/ not built")
    import hashlib
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for rel, meta in summary["files"].items():
        digest = hashlib.sha256((ROOT / "data" / rel).read_bytes()).hexdigest()
        assert digest == meta["sha256"], rel
