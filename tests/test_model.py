"""Model arms (RESEARCH_DESIGN.md §4.2): plan tests 10, 11 and the KV-cache half of test 8."""

from pathlib import Path

import pytest
import torch

from mathgen.model import ARCH_ARMS, KVCache, Transformer, arch_config

ROOT = Path(__file__).resolve().parents[1]
LEGACY_CKPT = ROOT / "checkpoints" / "best.pt"


def make(arm: str, vocab: int = 500, ctx: int = 256, seed: int = 0) -> Transformer:
    torch.manual_seed(seed)
    return Transformer(arch_config(arm, vocab_size=vocab, context_len=ctx)).eval()


def test_ref_is_about_4_7m_non_embedding_params():
    assert 4.6e6 < make("ref").non_embedding_params() < 4.8e6


def test_non_embedding_params_match_within_1_percent_across_arms():
    """Plan test 10."""
    counts = {arm: make(arm).non_embedding_params() for arm in ARCH_ARMS}
    ref = counts["ref"]
    assert all(abs(c - ref) / ref < 0.01 for c in counts.values()), counts


@pytest.mark.parametrize("arm", ARCH_ARMS)
def test_model_is_causal(arm):
    m = make(arm)
    x = torch.randint(0, 500, (2, 30))
    y = x.clone()
    y[:, 20:] = torch.randint(0, 500, (2, 10))
    with torch.no_grad():
        assert torch.allclose(m(x)[:, :20], m(y)[:, :20], atol=1e-5)


@pytest.mark.parametrize("arm", ARCH_ARMS)
def test_kv_cache_matches_full_forward(arm):
    """Plan test 8 (KV-cache == no-cache)."""
    m = make(arm)
    x = torch.randint(0, 500, (3, 25))
    with torch.no_grad():
        full = m(x)
        cache = KVCache()
        steps = [m(x[:, :10], cache=cache)]
        for t in range(10, 25):
            steps.append(m(x[:, t:t + 1], cache=cache))
    assert torch.allclose(torch.cat(steps, dim=1), full, atol=1e-5)


def test_learned_positions_are_embedding_params():
    m = make("learned")
    assert m.non_embedding_params() == make("ref").non_embedding_params()
    assert m.num_params() > make("ref").num_params()


@pytest.mark.skipif(not LEGACY_CKPT.exists(), reason="legacy checkpoint not present")
def test_legacy_checkpoint_loads_strictly_with_identical_logits(legacy_dataset):
    """Plan test 11: the legacy 34M best.pt loads into the new module and reproduces its logits."""
    import model as legacy_model  # src/model.py (torch already imported safely by the fixture)

    state = torch.load(LEGACY_CKPT, map_location="cpu", weights_only=True)["model"]
    vocab = state["embed.weight"].shape[0]
    old = legacy_model.MathGen(legacy_model.MathGenConfig(vocab_size=vocab, use_grad_ckpt=False)).eval()
    old.load_state_dict(state, strict=True)
    new = Transformer.from_legacy_state_dict(state).eval()

    x = torch.randint(0, vocab, (2, 64), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        assert torch.allclose(old(x)[0], new(x), atol=1e-5)
