"""Tokenizer arms (RESEARCH_DESIGN.md §4.1; plan test 5)."""

import re
from pathlib import Path

import pytest

from mathgen.data.build import read_jsonl
from mathgen.data.formats import prompt_text, target_text
from mathgen.tokenizers import ARMS, Tokenizer, fit_tokenizer, pretokenize

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
pytestmark = pytest.mark.skipif(not (DATA / "fit" / "tokenizer_fit.jsonl").exists(), reason="data/ not built")

SPLITS = ["eval/test_id_unseen.jsonl", "eval/val.jsonl", "eval/block.jsonl", "eval/ood_same.jsonl",
          "eval/ood_longer.jsonl", "train/seed0.jsonl"]


def texts(rel: str, limit: int | None = None) -> list[tuple[str, str]]:
    items = read_jsonl(DATA / rel)[:limit]
    return [(prompt_text(i["question"]), target_text(i["hints"], i["answer"])) for i in items]


@pytest.fixture(scope="module")
def tokenizers() -> dict[str, Tokenizer]:
    fit = texts("fit/tokenizer_fit.jsonl")
    return {arm: fit_tokenizer(arm, [p + t for p, t in fit]) for arm in ARMS}


@pytest.fixture(scope="module")
def all_pairs() -> list[tuple[str, str]]:
    return [pair for rel in SPLITS for pair in texts(rel, limit=20_000)]


def test_pretokenizer_isolates_ascii_digit_runs_and_specials():
    # ASCII digit runs are their own chunks; '²' and '−' are not digits (audit F8).
    assert pretokenize("[SOLVE] Compute 12x^3 − 4² <SEP><JSON>") == [
        "[SOLVE]", " ", "Compute", " ", "12", "x^", "3", " ", "−", " ", "4", "²", " ", "<SEP>", "<JSON>"]


@pytest.mark.parametrize("arm", ARMS)
def test_round_trip_and_no_unknown_tokens_on_every_split(arm, tokenizers, all_pairs):
    tok = tokenizers[arm]
    for p, t in all_pairs:
        ids = tok.encode(p + t)
        assert tok.unk_id not in ids, p + t
        assert tok.decode(ids) == p + t


@pytest.mark.parametrize("arm", ARMS)
def test_no_sequence_exceeds_context(arm, tokenizers, all_pairs):
    tok = tokenizers[arm]
    assert max(len(tok.encode(p + t)) for p, t in all_pairs) <= 256


def test_char_arm_has_one_token_per_character(tokenizers):
    tok = tokenizers["char"]
    assert tok.encode("ab12") == [tok.token_to_id[c] for c in "ab12"]


def test_digit_arm_has_no_multi_character_token_containing_a_digit(tokenizers):
    tok = tokenizers["digit"]
    assert not [s for s in tok.vocab if len(s) > 1 and re.search(r"[0-9]", s) and s not in tok.specials]


def test_bpe_num_never_fuses_digits_with_non_digits(tokenizers):
    tok = tokenizers["bpe_num"]
    mixed = [s for s in tok.vocab if s not in tok.specials and re.search(r"[0-9]", s) and re.search(r"[^0-9]", s)]
    assert not mixed
    assert any(len(s) > 1 and s.isascii() and s.isdigit() for s in tok.vocab), "no multi-digit merges learned"


def test_digit_and_bpe_num_tokenize_non_number_text_identically(tokenizers, all_pairs):
    digit, bpe = tokenizers["digit"], tokenizers["bpe_num"]
    assert bpe.vocab[:len(digit.vocab)] == digit.vocab, "T-digit vocab must be a prefix of T-bpe-num vocab"
    for p, t in all_pairs[:5000]:
        no_numbers = re.sub(r"[0-9]+", "", p + t)
        assert digit.encode(no_numbers) == bpe.encode(no_numbers)


def test_bpe_num_merges_numbers_the_digit_arm_splits(tokenizers):
    digit, bpe = tokenizers["digit"], tokenizers["bpe_num"]
    assert len(digit.encode("128")) == 3
    assert len(bpe.encode("128")) < 3


def test_decoupled_number_modes(tokenizers):
    bpe = tokenizers["bpe_num"]
    assert bpe.encode("x = 128", numbers="digit") == [bpe.token_to_id[s] for s in ("x", " ", "=", " ", "1", "2", "8")]
    assert bpe.decode(bpe.encode("x = 128", numbers="digit")) == "x = 128"
    assert bpe.encode("x = 128", numbers="merged") == bpe.encode("x = 128")


def test_fitting_is_deterministic(tokenizers):
    fit = texts("fit/tokenizer_fit.jsonl")
    again = fit_tokenizer("bpe_num", [p + t for p, t in fit])
    assert again.vocab == tokenizers["bpe_num"].vocab


@pytest.mark.parametrize("arm", ARMS)
def test_committed_tokenizers_equal_a_fresh_fit(arm, tokenizers):
    """data/tokenizers/*.json (written by scripts/fit_tokenizers.py) must be reproducible."""
    path = DATA / "tokenizers" / f"{arm}.json"
    if not path.exists():
        pytest.skip("tokenizers not fitted yet")
    assert Tokenizer.load(path).vocab == tokenizers[arm].vocab


def test_save_and_load_round_trip(tmp_path, tokenizers):
    tok = tokenizers["bpe_num"]
    tok.save(tmp_path / "tok.json")
    loaded = Tokenizer.load(tmp_path / "tok.json")
    assert loaded.vocab == tok.vocab and loaded.encode("Compute 12 * 3.") == tok.encode("Compute 12 * 3.")
