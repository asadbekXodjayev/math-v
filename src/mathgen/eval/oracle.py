"""
Answer oracle for data v1: derives the correct answer from the question text alone, and
scores predicted answers against it.

- `solve(question)` re-derives the answer **independently of the generator code**: regexes
  over the rendered question plus exact arithmetic. It is the second of the two oracles that
  plan test 1 requires to agree. Fixes audit F4: exact rational division instead of floor
  division, and negative operands are parsed.
- `math_correct(pred, truth, atype)` compares by meaning under a restricted grammar per
  answer type (see RESEARCH_DESIGN.md §5). There is no sympy and no eval: a prediction
  that doesn't parse is wrong, so hostile strings fail fast and nothing needs a timeout.
- `exact_match(pred, truth)` is string equality after whitespace normalization.

Answer types: int, rational, decimal, surd, roots, vector, matrix, poly, antiderivative.
"""

import json
import math
import re
from fractions import Fraction

MAX_PRED_CHARS = 200          # longer predictions are wrong without parsing
DECIMAL_TOL = 5e-5            # half of the 4th decimal place

_INT = r"-?[0-9]+"


# ── polynomials: {power: coefficient} with exact Fractions ──────────────────────

_TERM = re.compile(r"^([0-9]+(?:/([0-9]+))?)?(?:\*?(x(?:(?:\^|\*\*)([0-9]+))?))?$")


def _parse_poly(s: str) -> dict[int, Fraction] | None:
    """Parse '12x^3 + 4x - 2' (also '*', '**', no spaces) into {power: coeff}."""
    s = s.replace("−", "-").replace(" ", "")
    if not s or len(s) > MAX_PRED_CHARS:
        return None
    if s[0] not in "+-":
        s = "+" + s
    terms = re.findall(r"[+-][^+-]+", s)
    if "".join(terms) != s:
        return None
    poly: dict[int, Fraction] = {}
    for t in terms:
        sign, body = (-1 if t[0] == "-" else 1), t[1:]
        m = _TERM.match(body)
        if not m or (m.group(1) is None and m.group(3) is None):
            return None
        if m.group(2) is not None and int(m.group(2)) == 0:
            return None
        coeff = Fraction(m.group(1)) if m.group(1) else Fraction(1)
        power = (int(m.group(4)) if m.group(4) else 1) if m.group(3) else 0
        if power > 50:
            return None
        poly[power] = poly.get(power, Fraction(0)) + sign * coeff
    return {p: c for p, c in poly.items() if c != 0}


def _format_poly(poly: dict[int, Fraction]) -> str | None:
    """Render like the generators: descending powers, '12x^3 + 4x - 2'. Integer coefficients only."""
    if any(c.denominator != 1 for c in poly.values()):
        return None
    parts = []
    for p in sorted(poly, reverse=True):
        c = int(poly[p])
        mag = abs(c)
        body = str(mag) if p == 0 else (("" if mag == 1 else str(mag)) + ("x" if p == 1 else f"x^{p}"))
        parts.append(("-" if c < 0 else "+", body))
    if not parts:
        return "0"
    out = parts[0][1] if parts[0][0] == "+" else "-" + parts[0][1]
    return out + "".join(f" {sign} {body}" for sign, body in parts[1:])


def _eval_poly(poly: dict[int, Fraction], x: Fraction) -> Fraction:
    return sum((c * x ** p for p, c in poly.items()), Fraction(0))


# ── rendering helpers for derived answers ───────────────────────────────────────

def _rational_str(r: Fraction) -> str:
    return str(r.numerator) if r.denominator == 1 else f"{r.numerator}/{r.denominator}"


def _mats(s: str) -> list[list[list[int]]]:
    return [json.loads(m) for m in re.findall(rf"\[\[{_INT},\s*{_INT}\],\s*\[{_INT},\s*{_INT}\]\]", s)]


def _vecs(s: str) -> list[list[int]]:
    return [[int(a), int(b)] for a, b in re.findall(rf"\[\s*({_INT})\s*,\s*({_INT})\s*\]", s)]


# ── solve(): independent derivation ─────────────────────────────────────────────

def solve(question: str) -> tuple[str, str] | None:
    q = question.strip()

    if m := re.fullmatch(rf"Evaluate: ({_INT}) × \(({_INT}) \+ ({_INT})\)", q):
        a, b, c = map(int, m.groups())
        return str(a * (b + c)), "int"

    if m := re.fullmatch(rf"Compute ({_INT}) ([+\-*]) ({_INT})\.", q):
        a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
        return str(a + b if op == "+" else a - b if op == "-" else a * b), "int"

    if m := re.fullmatch(rf"Add the fractions: ({_INT})/({_INT}) \+ ({_INT})/({_INT})", q):
        n1, d1, n2, d2 = map(int, m.groups())
        return _rational_str(Fraction(n1, d1) + Fraction(n2, d2)), "rational"

    if m := re.fullmatch(rf"Solve for x: ({_INT})x \+ ({_INT}) = ({_INT})", q):
        a, b, c = map(int, m.groups())
        return (_rational_str(Fraction(c - b, a)), "rational") if a != 0 else None

    if m := re.fullmatch(r"Solve: x² ([+-]) ([0-9]+)x ([+-]) ([0-9]+) = 0", q):
        B = int(m.group(2)) * (1 if m.group(1) == "+" else -1)
        C = int(m.group(4)) * (1 if m.group(3) == "+" else -1)
        disc = B * B - 4 * C
        root = math.isqrt(disc) if disc >= 0 else -1
        if root < 0 or root * root != disc or (B + root) % 2:
            return None
        return json.dumps(sorted([(-B - root) // 2, (-B + root) // 2])), "roots"

    if m := re.fullmatch(rf"Given f\(x\) = ({_INT})x² \+ ({_INT})x \+ ({_INT}), find f\(({_INT})\)\.", q):
        a, b, c, x = map(int, m.groups())
        return str(a * x * x + b * x + c), "int"

    if m := re.fullmatch(rf"Find \|\|v\|\| for v = \[({_INT}), ({_INT})\]\.", q):
        a, b = map(int, m.groups())
        sq = a * a + b * b
        r = math.isqrt(sq)
        return (str(r) if r * r == sq else f"sqrt({sq})"), "surd"

    if m := re.fullmatch(rf"Compute u·v for u = \[({_INT}), ({_INT})\] and v = \[({_INT}), ({_INT})\]\.", q):
        a, b, c, d = map(int, m.groups())
        return str(a * c + b * d), "int"

    if m := re.fullmatch(rf"Find cos\(θ\) between u = \[({_INT}), ({_INT})\] and v = \[({_INT}), ({_INT})\]\.", q):
        a, b, c, d = map(int, m.groups())
        return str(round((a * c + b * d) / (math.sqrt(a * a + b * b) * math.sqrt(c * c + d * d)), 4)), "decimal"

    if q.startswith("Add A = ") and len(ms := _mats(q)) == 2:
        A, B = ms
        return json.dumps([[A[i][j] + B[i][j] for j in range(2)] for i in range(2)]), "matrix"

    if (m := re.match(rf"Compute ({_INT})A for A = ", q)) and len(ms := _mats(q)) == 1:
        k, A = int(m.group(1)), ms[0]
        return json.dumps([[k * A[i][j] for j in range(2)] for i in range(2)]), "matrix"

    if q.startswith("Compute AB where") and len(ms := _mats(q)) == 2:
        A, B = ms
        return json.dumps([[A[i][0] * B[0][j] + A[i][1] * B[1][j] for j in range(2)] for i in range(2)]), "matrix"

    if q.startswith("Find det(A) for A = ") and len(ms := _mats(q)) == 1:
        A = ms[0]
        return str(A[0][0] * A[1][1] - A[0][1] * A[1][0]), "int"

    if q.startswith("Compute Aᵀ for A = ") and len(ms := _mats(q)) == 1:
        A = ms[0]
        return json.dumps([[A[j][i] for j in range(2)] for i in range(2)]), "matrix"

    if q.startswith("Apply the 90° CCW rotation matrix A = [[0,-1],[1,0]] to v = "):
        vs = _vecs(q.split(" to v = ", 1)[1])
        if len(vs) == 1:
            x, y = vs[0]
            return json.dumps([-y, x]), "vector"
        return None

    if m := re.fullmatch(r"Differentiate f\(x\) = (.+)\.", q):
        poly = _parse_poly(m.group(1))
        if poly is None:
            return None
        out = _format_poly({p - 1: c * p for p, c in poly.items() if p > 0})
        return (out, "poly") if out is not None else None

    if m := re.fullmatch(rf"Find the slope of f\(x\) = (.+) at x = ({_INT})\.", q):
        poly = _parse_poly(m.group(1))
        if poly is None:
            return None
        slope = _eval_poly({p - 1: c * p for p, c in poly.items() if p > 0}, Fraction(int(m.group(2))))
        return _rational_str(slope), "int"

    if m := re.fullmatch(r"Evaluate the indefinite integral: ∫ (.+) dx\.", q):
        poly = _parse_poly(m.group(1))
        if poly is None:
            return None
        out = _format_poly({p + 1: c / (p + 1) for p, c in poly.items()})
        return (out + " + C", "antiderivative") if out is not None else None

    if m := re.fullmatch(rf"Evaluate the definite integral: ∫ from ({_INT}) to ({_INT}) of (.+) dx\.", q):
        lo, hi, poly = Fraction(int(m.group(1))), Fraction(int(m.group(2))), _parse_poly(m.group(3))
        if poly is None:
            return None
        anti = {p + 1: c / (p + 1) for p, c in poly.items()}
        return _rational_str(_eval_poly(anti, hi) - _eval_poly(anti, lo)), "rational"

    if m := re.fullmatch(rf"Evaluate the limit: lim\(x→({_INT})\) \(x\^2 − ([0-9]+)\) / \(x − ({_INT})\)\.", q):
        c, sq, c2 = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return (str(2 * c), "int") if c == c2 and c * c == sq else None

    if m := re.fullmatch(rf"Evaluate the limit: lim\(x→∞\) \(({_INT})x\^([0-9]+) \+ 1\) / \(({_INT})x\^([0-9]+) \+ x\)\.", q):
        a, n, b, n2 = map(int, m.groups())
        return (_rational_str(Fraction(a, b)), "rational") if n == n2 and n >= 2 else None

    if m := re.fullmatch(rf"Evaluate the limit: lim\(x→({_INT})\) \((.+)\)\.", q):
        poly = _parse_poly(m.group(2))
        if poly is None:
            return None
        return _rational_str(_eval_poly(poly, Fraction(int(m.group(1))))), "int"

    return None


# ── scoring ──────────────────────────────────────────────────────────────────────

def _parse_rational(s: str) -> Fraction | None:
    s = s.replace(" ", "")
    if not re.fullmatch(r"-?[0-9]+(/-?[0-9]+)?", s):
        return None
    try:
        return Fraction(s)
    except (ValueError, ZeroDivisionError):
        return None


def _parse_surd_square(s: str) -> int | None:
    """Value² of 'k', 'sqrt(n)', 'ksqrt(n)' or 'k*sqrt(n)' (non-negative forms)."""
    s = s.replace(" ", "")
    if m := re.fullmatch(r"([0-9]+)?\*?sqrt\(([0-9]+)\)", s):
        k = int(m.group(1)) if m.group(1) else 1
        return k * k * int(m.group(2))
    if re.fullmatch(r"[0-9]+", s):
        return int(s) ** 2
    return None


def _parse_int_json(s: str, shape: str):
    s = s.replace(" ", "")
    pattern = {"vector": rf"\[{_INT},{_INT}\]",
               "matrix": rf"\[\[{_INT},{_INT}\],\[{_INT},{_INT}\]\]",
               "roots": rf"\[{_INT},{_INT}\]"}[shape]
    return json.loads(s) if re.fullmatch(pattern, s) else None


def _value(s: str, atype: str):
    if atype == "int":
        return int(s.strip()) if re.fullmatch(_INT, s.strip()) else None
    if atype == "rational":
        return _parse_rational(s)
    if atype == "decimal":
        t = s.strip()
        return float(t) if re.fullmatch(r"-?[0-9]+(\.[0-9]+)?", t) else None
    if atype == "surd":
        return _parse_surd_square(s)
    if atype in ("vector", "matrix"):
        return _parse_int_json(s, atype)
    if atype == "roots":
        v = _parse_int_json(s, "roots")
        return sorted(v) if v is not None else None
    if atype == "poly":
        return _parse_poly(s)
    if atype == "antiderivative":
        m = re.fullmatch(r"(.+?)\s*\+\s*C", s.strip())
        return _parse_poly(m.group(1)) if m else None
    raise ValueError(f"unknown answer type {atype!r}")


def math_correct(pred: str | None, truth: str, atype: str) -> bool:
    if not isinstance(pred, str) or not pred.strip() or len(pred) > MAX_PRED_CHARS:
        return False
    p, t = _value(pred, atype), _value(truth, atype)
    if p is None or t is None:
        return False
    if atype == "decimal":
        return abs(p - t) <= DECIMAL_TOL
    return p == t


def exact_match(pred: str | None, truth: str) -> bool:
    if not isinstance(pred, str):
        return False
    return " ".join(pred.split()) == " ".join(truth.split())
