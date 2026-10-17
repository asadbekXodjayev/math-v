"""Templates v1: legacy parity, the F3 fix, oracle agreement (plan tests 1-3), space sizes."""

import random

import pytest

from mathgen.data.ranges import domains_for, enumerate_space, sample_operands, space_size
from mathgen.data.templates import TEMPLATES
from mathgen.eval.oracle import solve

LEGACY_NAMES = [
    "arith_pemdas", "arith_integer", "arith_fractions", "algebra_linear", "algebra_quadratic",
    "func_evaluate", "vec_magnitude", "vec_dot", "vec_angle", "mat_add", "mat_scalar", "mat_mul",
    "mat_det", "mat_transpose", "linear_transform", "deriv_power", "deriv_polynomial",
    "deriv_at_point", "integral_power", "integral_definite", "limit_substitution",
    "limit_factor", "limit_infinity",
]

# Measured in the M1 audit (research/REPOSITORY_AUDIT.md, F5); "large" spaces are exact products.
AUDIT_SPACE_SIZES = {
    "limit_factor": 5, "integral_power": 12, "deriv_power": 20, "integral_definite": 27,
    "limit_infinity": 50, "deriv_at_point": 80, "algebra_quadratic": 91, "linear_transform": 144,
    "vec_magnitude": 144, "arith_pemdas": 512, "vec_angle": 625, "limit_substitution": 864,
    "deriv_polynomial": 1080, "algebra_linear": 1120, "arith_fractions": 1225,
}

# Pre-registered in RESEARCH_DESIGN.md §3.1.
M_SET = {"vec_magnitude", "linear_transform", "deriv_power", "deriv_at_point", "integral_power",
         "integral_definite", "limit_factor", "limit_infinity", "algebra_quadratic"}


class ScriptedRandom:
    """Stands in for the `random` module inside legacy dataset.py: returns given values in order."""

    def __init__(self, values):
        self.values = list(values)

    def randint(self, lo, hi):
        return self.values.pop(0)

    def choice(self, seq):
        return self.values.pop(0)


def test_all_23_legacy_templates_are_ported():
    assert sorted(TEMPLATES) == sorted(LEGACY_NAMES)


@pytest.mark.parametrize("name", LEGACY_NAMES)
@pytest.mark.parametrize("split", ["id", "ood_same", "ood_longer"])
def test_rendering_matches_legacy_generator(name, split, config, legacy_dataset, monkeypatch):
    domains = domains_for(config, name, split)
    if domains is None:
        pytest.skip(f"{name} has no {split} range")
    tpl, rng = TEMPLATES[name], random.Random(f"parity|{name}|{split}")
    legacy_fn = getattr(legacy_dataset.TopicGenerators, name)
    for _ in range(200):
        ops = sample_operands(tpl, domains, rng)
        monkeypatch.setattr(legacy_dataset, "random", ScriptedRandom(ops[p] for p in tpl.params))
        assert tpl.render(ops) == legacy_fn(), ops


def test_subtraction_bug_is_fixed(config):
    """Audit F3: legacy turned ~51% of subtractions into x - x. Fixed: a >= b, P(a == b) = 1/29."""
    tpl, domains, rng = TEMPLATES["arith_integer"], domains_for(config, "arith_integer", "id"), random.Random(3)
    subs = [ops for ops in (sample_operands(tpl, domains, rng) for _ in range(150_000)) if ops["op"] == "-"]
    assert all(o["a"] >= o["b"] for o in subs)
    rate = sum(o["a"] == o["b"] for o in subs) / len(subs)
    assert abs(rate - 1 / 29) < 0.005


@pytest.mark.parametrize("name", LEGACY_NAMES)
@pytest.mark.parametrize("split", ["id", "ood_same", "ood_longer"])
def test_generator_and_oracle_agree(name, split, config):
    """Plan test 1 (random draws): the generator's answer equals the oracle's independent answer."""
    domains = domains_for(config, name, split)
    if domains is None:
        pytest.skip(f"{name} has no {split} range")
    tpl, rng = TEMPLATES[name], random.Random(f"agree|{name}|{split}")
    for _ in range(2000):
        q, _, answer = tpl.render(sample_operands(tpl, domains, rng))
        assert solve(q) == (answer, tpl.atype), q


def test_sampling_is_deterministic_and_ignores_global_random(config):
    """Plan test 3: generators draw only from the rng passed in."""
    tpl, domains = TEMPLATES["mat_mul"], domains_for(config, "mat_mul", "id")
    random.seed(1)
    first = [sample_operands(tpl, domains, random.Random(7)) for _ in range(3)]
    random.seed(999)
    second = [sample_operands(tpl, domains, random.Random(7)) for _ in range(3)]
    assert first == second


@pytest.mark.parametrize("name", sorted(AUDIT_SPACE_SIZES))
def test_space_size_matches_audit_and_keys_are_injective(name, config):
    tpl, domains = TEMPLATES[name], domains_for(config, name, "id")
    instances = enumerate_space(tpl, domains)
    assert len(instances) == AUDIT_SPACE_SIZES[name] == space_size(tpl, domains)
    questions = {tpl.render(ops)[0] for ops in instances}
    assert len(questions) == len(instances), "two instance keys render the same question"


def test_g_and_m_sets_match_preregistration(config):
    sizes = {n: space_size(TEMPLATES[n], domains_for(config, n, "id")) for n in TEMPLATES}
    m_set = {n for n, s in sizes.items() if s < config["gset_min_space"]}
    assert m_set == M_SET
    assert all(sizes[n] >= 500 for n in TEMPLATES if n not in M_SET)


def test_ood_operands_lie_outside_the_id_range(config):
    checked = 0
    for name, tpl in TEMPLATES.items():
        id_domains = domains_for(config, name, "id")
        for split in ("ood_same", "ood_longer"):
            domains = domains_for(config, name, split)
            if domains is None:
                continue
            varied = config["templates"][name][split]
            varied = tpl.params if "_default" in varied else list(varied)
            rng = random.Random(f"ood|{name}|{split}")
            for _ in range(200):
                ops = sample_operands(tpl, domains, rng)
                for p in varied:
                    assert ops[p] not in id_domains[p], (name, split, p, ops[p])
                    checked += 1
    assert checked > 10_000
