"""
Operand domains from configs/base.yaml: parsing, uniform sampling, enumeration.

A domain is the explicit list of allowed values for one parameter. Sampling draws each
parameter uniformly and independently, exactly like the legacy generators, then applies the
template's canonicalization.
"""

import itertools
import math
import random

from .templates import Ops, Template

ENUMERATE_LIMIT = 2_000_000   # larger spaces are counted, not enumerated


def parse_domain(spec: list) -> list:
    if all(isinstance(v, str) for v in spec):
        return list(spec)
    if len(spec) == 2 and all(isinstance(v, int) for v in spec):
        lo, hi = spec
        return list(range(lo, hi + 1))
    if all(isinstance(r, list) for r in spec):
        return [v for r in spec for v in parse_domain(r)]
    raise ValueError(f"bad domain spec {spec!r}")


def _overlay(params: tuple[str, ...], base: dict[str, list], spec: dict) -> dict[str, list]:
    out = dict(base)
    if "_default" in spec:
        out.update({p: parse_domain(spec["_default"]) for p in params})
    for p, s in spec.items():
        if p == "_default":
            continue
        if p not in params:
            raise ValueError(f"unknown parameter {p!r}")
        out[p] = parse_domain(s)
    return out


def domains_for(config: dict, name: str, split: str) -> dict[str, list] | None:
    """Domains for split 'id', 'ood_same' or 'ood_longer'; None if the template has no such split."""
    from .templates import TEMPLATES
    params, tcfg = TEMPLATES[name].params, config["templates"][name]
    domains = _overlay(params, {}, tcfg["id"])
    if missing := set(params) - set(domains):
        raise ValueError(f"{name}: no id domain for {sorted(missing)}")
    if split == "id":
        return domains
    if split not in tcfg:
        return None
    return _overlay(params, domains, tcfg[split])


def sample_operands(template: Template, domains: dict[str, list], rng: random.Random) -> Ops:
    return template.canon({p: rng.choice(domains[p]) for p in template.params})


def product_size(template: Template, domains: dict[str, list]) -> int:
    return math.prod(len(domains[p]) for p in template.params)


def enumerate_space(template: Template, domains: dict[str, list]) -> list[Ops]:
    """Every distinct canonical instance, in a deterministic order."""
    if product_size(template, domains) > ENUMERATE_LIMIT:
        raise ValueError(f"{template.name}: space too large to enumerate")
    seen: dict[str, Ops] = {}
    for values in itertools.product(*(domains[p] for p in template.params)):
        ops = template.canon(dict(zip(template.params, values)))
        seen.setdefault(template.key(ops), ops)
    return list(seen.values())


def space_size(template: Template, domains: dict[str, list]) -> int:
    if product_size(template, domains) <= ENUMERATE_LIMIT:
        return len(enumerate_space(template, domains))
    if template.canonicalizes:
        raise ValueError(f"{template.name}: cannot count a canonicalizing space without enumerating")
    return product_size(template, domains)
