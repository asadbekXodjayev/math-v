"""
Deterministic math verifier — the single source of truth for answer correctness.

Two public entry points:
  compute_true_answer(question) -> (answer_str, kind) | None
      Independently re-derives the correct answer from a question string,
      WITHOUT trusting any model output. kind is "exact" or "sym".
  smart_equal(a, b, kind="exact") -> bool
      Robust equality (numeric / fraction / json / sympy-equivalence).

Used by:
  - eval_accuracy.py  (measuring the model)
  - server.py         (verifier-in-the-loop: guarantee served answers are correct)

This logic is validated by eval_accuracy's generator control, which must score
100% (1200/1200) — that control is the regression test for this file.
"""

import re, json, math
from fractions import Fraction

import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication_application,
)

X = sp.Symbol("x")
_TRANSF = standard_transformations + (implicit_multiplication_application,)


# ── math-text normalisation ─────────────────────────────────────────────────────

def _norm_math(s: str) -> str:
    return (s.replace("×", "*").replace("·", "*").replace("−", "-")
             .replace("∞", "oo").replace("√", "sqrt")
             .replace("²", "**2").replace("³", "**3").replace("^", "**")
             .strip())


def _sym(s: str):
    return parse_expr(_norm_math(s), local_dict={"x": X}, transformations=_TRANSF)


def _mats(s: str):
    out = []
    for m in re.finditer(r"\[\[[^\]]*\],\s*\[[^\]]*\]\]", s):
        try:
            out.append(json.loads(m.group()))
        except Exception:
            pass
    return out


def _vec(s: str):
    m = re.search(r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]", s)
    return [int(m.group(1)), int(m.group(2))] if m else None


# ── independent answer derivation ───────────────────────────────────────────────

def compute_true_answer(q: str):
    qs = q.strip()

    m = re.search(r"Evaluate:\s*(\d+)\s*×\s*\(\s*(\d+)\s*\+\s*(\d+)\s*\)", qs)
    if m:
        a, b, c = map(int, m.groups()); return str(a * (b + c)), "exact"

    m = re.search(r"Compute\s+(\d+)\s*([+\-*])\s*(\d+)\.", qs)
    if m:
        a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
        return str(a + b if op == "+" else a - b if op == "-" else a * b), "exact"

    m = re.search(r"Add the fractions:\s*(\d+)/(\d+)\s*\+\s*(\d+)/(\d+)", qs)
    if m:
        n1, d1, n2, d2 = map(int, m.groups())
        r = Fraction(n1, d1) + Fraction(n2, d2)
        return (str(r.numerator) if r.denominator == 1 else f"{r.numerator}/{r.denominator}"), "exact"

    m = re.search(r"Solve for x:\s*(\d+)x\s*\+\s*(\d+)\s*=\s*(-?\d+)", qs)
    if m:
        a, b, c = map(int, m.groups()); return str((c - b) // a), "exact"

    m = re.search(r"Solve:\s*x²\s*([+\-])\s*(\d+)x\s*([+\-])\s*(\d+)\s*=\s*0", qs)
    if m:
        sb, b, sc, c = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))
        B = b if sb == "+" else -b
        C = c if sc == "+" else -c
        disc = B * B - 4 * C
        if disc < 0:
            return None
        sq = math.isqrt(disc)
        if sq * sq != disc:
            return None
        r1, r2 = (-B - sq) // 2, (-B + sq) // 2
        return json.dumps(sorted([r1, r2])), "exact"

    m = re.search(r"Given f\(x\)\s*=\s*(\d+)x²\s*\+\s*(\d+)x\s*\+\s*(\d+),\s*find f\((-?\d+)\)", qs)
    if m:
        a, b, c, k = map(int, m.groups()); return str(a * k * k + b * k + c), "exact"

    m = re.search(r"Find \|\|v\|\| for v = \[(\d+),\s*(\d+)\]", qs)
    if m:
        a, b = map(int, m.groups()); sq = a * a + b * b; r = math.isqrt(sq)
        return (str(r) if r * r == sq else f"sqrt({sq})"), "exact"

    m = re.search(r"Compute u\*v for u = \[(\d+),\s*(\d+)\] and v = \[(\d+),\s*(\d+)\]",
                  qs.replace("·", "*"))
    if m:
        a, b, c, d = map(int, m.groups()); return str(a * c + b * d), "exact"

    m = re.search(r"Find cos\(θ\) between u = \[(\d+),\s*(\d+)\] and v = \[(\d+),\s*(\d+)\]", qs)
    if m:
        a, b, c, d = map(int, m.groups())
        dot = a * c + b * d
        cos = round(dot / (math.sqrt(a * a + b * b) * math.sqrt(c * c + d * d)), 4)
        return str(cos), "exact"

    if qs.startswith("Add A ="):
        ms = _mats(qs)
        if len(ms) == 2:
            A, B = ms
            return json.dumps([[A[i][j] + B[i][j] for j in range(2)] for i in range(2)]), "exact"

    m = re.match(r"Compute (\d+)A for A = ", qs)
    if m:
        ms = _mats(qs)
        if ms:
            k = int(m.group(1)); A = ms[0]
            return json.dumps([[k * A[i][j] for j in range(2)] for i in range(2)]), "exact"

    if qs.startswith("Compute AB where"):
        ms = _mats(qs)
        if len(ms) == 2:
            A, B = ms
            return json.dumps([[A[i][0] * B[0][j] + A[i][1] * B[1][j] for j in range(2)] for i in range(2)]), "exact"

    if qs.startswith("Find det(A)"):
        ms = _mats(qs)
        if ms:
            A = ms[0]; return str(A[0][0] * A[1][1] - A[0][1] * A[1][0]), "exact"

    if qs.startswith("Compute A") and "for A =" in qs and ("ᵀ" in qs or "^T" in qs):
        ms = _mats(qs)
        if ms:
            A = ms[0]; return json.dumps([[A[j][i] for j in range(2)] for i in range(2)]), "exact"

    if "rotation matrix" in qs:
        v = _vec(qs.split("to v")[-1]) if "to v" in qs else None
        if v:
            A = [[0, -1], [1, 0]]
            return json.dumps([A[0][0] * v[0] + A[0][1] * v[1], A[1][0] * v[0] + A[1][1] * v[1]]), "exact"

    m = re.search(r"Find the slope of f\(x\)\s*=\s*(.+?)\s*at x = (-?\d+)", qs)
    if m:
        try:
            d = sp.diff(_sym(m.group(1)), X)
            return str(int(d.subs(X, int(m.group(2))))), "exact"
        except Exception:
            return None

    m = re.search(r"Differentiate f\(x\)\s*=\s*(.+?)\.?\s*$", qs)
    if m:
        try:
            return str(sp.diff(_sym(m.group(1)), X)), "sym"
        except Exception:
            return None

    m = re.search(r"from 0 to (\d+) of\s*(.+?)\s*dx", qs)
    if m:
        try:
            b = int(m.group(1)); val = sp.integrate(_sym(m.group(2)), (X, 0, b))
            return (str(int(val)) if val == int(val) else str(val)), "exact"
        except Exception:
            return None

    m = re.search(r"indefinite integral:\s*∫\s*(.+?)\s*dx", qs)
    if m:
        try:
            return str(sp.integrate(_sym(m.group(1)), X)), "sym"
        except Exception:
            return None

    m = re.search(r"lim\(x→(.+?)\)\s+(.+?)\.?\s*$", qs)
    if m:
        try:
            pt = sp.oo if "∞" in m.group(1) else int(m.group(1))
            return str(sp.limit(_sym(m.group(2)), X, pt)), "exact"
        except Exception:
            return None

    return None


# ── robust equality ──────────────────────────────────────────────────────────────

def _eq_exact(a, b) -> bool:
    a, b = str(a).strip(), str(b).strip()
    if a == b or a.replace(" ", "") == b.replace(" ", ""):
        return True
    try:
        if abs(float(a) - float(b)) < 1e-4:
            return True
    except Exception:
        pass
    try:
        if Fraction(a) == Fraction(b):
            return True
    except Exception:
        pass
    try:
        if json.loads(a) == json.loads(b):
            return True
    except Exception:
        pass
    return False


def _eq_sym(a, b) -> bool:
    try:
        diff = sp.simplify(_sym(re.sub(r"\+\s*C\b", "", str(a))) - _sym(str(b)))
        return diff == 0
    except Exception:
        return _eq_exact(a, b)


def smart_equal(a, b, kind: str = "exact") -> bool:
    if a is None or b is None:
        return False
    return _eq_sym(a, b) if kind == "sym" else _eq_exact(a, b)


def answer_correct(model_ans, truth: str, kind: str) -> bool:
    return smart_equal(model_ans, truth, kind)
