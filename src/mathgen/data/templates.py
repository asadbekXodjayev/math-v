"""
The 23 data-v1 templates, ported from the legacy `TopicGenerators` (`src/dataset.py` at
`legacy-v0`).

Each template is a pure function of its operands; the operand *ranges* live in
configs/base.yaml. Parameters are declared in the legacy generator's draw order, which lets
tests/test_templates.py check byte-for-byte parity with the legacy code.

Changes from legacy, both canonicalizations of the operands:
- arith_integer subtraction orders the operands (max, min). This fixes the audit F3 bug
  (`a = max(a, b); b = min(a, b)` read the updated `a`).
- algebra_quadratic sorts its roots, so one question always has the same hints.
"""

import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from math import gcd

Ops = dict[str, int | str]


@dataclass(frozen=True)
class Template:
    name: str
    params: tuple[str, ...]
    atype: str
    render: Callable[[Ops], tuple[str, list[str], str]]
    canon: Callable[[Ops], Ops] = field(default=lambda ops: ops)
    canonicalizes: bool = False

    def key(self, ops: Ops) -> str:
        return self.name + "|" + ",".join(f"{p}={ops[p]}" for p in self.params)


# ── legacy helpers (verbatim ports) ──────────────────────────────────────────────

def _poly_str(terms: list[tuple[int, int]]) -> str:
    """Format [(coeff, power), ...] as a polynomial string, e.g. '12x^3 + 4x - 2'."""
    parts: list[tuple[str, str]] = []
    for coeff, power in terms:
        if coeff == 0:
            continue
        mag = abs(coeff)
        if power == 0:
            term = f"{mag}"
        elif power == 1:
            term = "x" if mag == 1 else f"{mag}x"
        else:
            term = f"x^{power}" if mag == 1 else f"{mag}x^{power}"
        parts.append(("-" if coeff < 0 else "+", term))
    if not parts:
        return "0"
    out = parts[0][1] if parts[0][0] == "+" else f"-{parts[0][1]}"
    for sign, term in parts[1:]:
        out += f" {sign} {term}"
    return out


def _mat_mul(A: list[list[int]], B: list[list[int]]) -> list[list[int]]:
    return [[A[i][0] * B[0][j] + A[i][1] * B[1][j] for j in range(2)] for i in range(2)]


def _mat(o: Ops, p: str) -> list[list[int]]:
    return [[o[f"{p}11"], o[f"{p}12"]], [o[f"{p}21"], o[f"{p}22"]]]


_A = ("a11", "a12", "a21", "a22")
_B = ("b11", "b12", "b21", "b22")


# ── renderers ────────────────────────────────────────────────────────────────────

def _arith_pemdas(o):
    a, b, c = o["a"], o["b"], o["c"]
    ans = a * (b + c)
    return (f"Evaluate: {a} × ({b} + {c})",
            [f"Parentheses first: {b} + {c} = {b+c}", f"Then multiply: {a} × {b+c} = {ans}"],
            str(ans))


def _arith_integer(o):
    a, b, op = o["a"], o["b"], o["op"]
    if op == "+":
        ans = a + b; s = f"{a} + {b} = {ans}"
    elif op == "-":
        ans = a - b; s = f"{a} - {b} = {ans}"
    else:
        ans = a * b; s = f"{a} × {b} = {ans}"
    return f"Compute {a} {op} {b}.", [s], str(ans)


def _arith_integer_canon(o):
    if o["op"] == "-" and o["a"] < o["b"]:
        return {**o, "a": o["b"], "b": o["a"]}
    return o


def _arith_fractions(o):
    n1, d1, n2, d2 = o["n1"], o["d1"], o["n2"], o["d2"]
    lcd = d1 * d2 // gcd(d1, d2)
    num = n1 * (lcd // d1) + n2 * (lcd // d2)
    g = gcd(num, lcd)
    rn, rd = num // g, lcd // g
    ans = f"{rn}/{rd}" if rd != 1 else str(rn)
    return (f"Add the fractions: {n1}/{d1} + {n2}/{d2}",
            [f"LCD = {lcd}", f"= {n1*(lcd//d1)}/{lcd} + {n2*(lcd//d2)}/{lcd} = {num}/{lcd} = {ans}"],
            ans)


def _algebra_linear(o):
    a, x, b = o["a"], o["x"], o["b"]
    c = a * x + b
    return (f"Solve for x: {a}x + {b} = {c}",
            [f"Subtract {b}: {a}x = {c - b}", f"Divide by {a}: x = {x}"],
            str(x))


def _algebra_quadratic(o):
    r1, r2 = o["r1"], o["r2"]
    b, c = -(r1 + r2), r1 * r2
    sign_b = f"+ {b}" if b >= 0 else f"- {-b}"
    sign_c = f"+ {c}" if c >= 0 else f"- {-c}"
    return (f"Solve: x² {sign_b}x {sign_c} = 0",
            [f"Factor: (x - {r1})(x - {r2}) = 0", f"x = {r1} or x = {r2}"],
            json.dumps(sorted([r1, r2])))


def _algebra_quadratic_canon(o):
    return o if o["r1"] <= o["r2"] else {**o, "r1": o["r2"], "r2": o["r1"]}


def _func_evaluate(o):
    a, b, c, x = o["a"], o["b"], o["c"], o["x"]
    ans = a * x**2 + b * x + c
    return (f"Given f(x) = {a}x² + {b}x + {c}, find f({x}).",
            [f"f({x}) = {a}({x})² + {b}({x}) + {c}", f"= {a*x**2} + {b*x} + {c} = {ans}"],
            str(ans))


def _vec_magnitude(o):
    v = [o["v0"], o["v1"]]
    sq = v[0]**2 + v[1]**2
    mag = math.sqrt(sq)
    ans = str(int(mag)) if mag == int(mag) else f"sqrt({sq})"
    return (f"Find ||v|| for v = [{v[0]}, {v[1]}].",
            [f"||v|| = sqrt({v[0]}² + {v[1]}²) = sqrt({v[0]**2} + {v[1]**2})", f"= {ans}"],
            ans)


def _vec_dot(o):
    u, v = [o["u0"], o["u1"]], [o["v0"], o["v1"]]
    dot = u[0]*v[0] + u[1]*v[1]
    return (f"Compute u·v for u = [{u[0]}, {u[1]}] and v = [{v[0]}, {v[1]}].",
            [f"u·v = {u[0]}·{v[0]} + {u[1]}·{v[1]} = {u[0]*v[0]} + {u[1]*v[1]}"],
            str(dot))


def _vec_angle(o):
    u, v = [o["u0"], o["u1"]], [o["v0"], o["v1"]]
    dot = u[0]*v[0] + u[1]*v[1]
    mag_u = math.sqrt(u[0]**2 + u[1]**2)
    mag_v = math.sqrt(v[0]**2 + v[1]**2)
    cos_theta = round(dot / (mag_u * mag_v), 4)
    return (f"Find cos(θ) between u = [{u[0]}, {u[1]}] and v = [{v[0]}, {v[1]}].",
            ["cos(θ) = u·v / (||u|| ||v||)",
             f"= {dot} / ({round(mag_u,3)} × {round(mag_v,3)}) ≈ {cos_theta}"],
            str(cos_theta))


def _mat_add(o):
    A, B = _mat(o, "a"), _mat(o, "b")
    C = [[A[i][j] + B[i][j] for j in range(2)] for i in range(2)]
    return (f"Add A = {A} and B = {B}.",
            ["Add entry by entry: (A+B)ᵢⱼ = aᵢⱼ + bᵢⱼ", f"Result = {C}"],
            json.dumps(C))


def _mat_scalar(o):
    A, k = _mat(o, "a"), o["k"]
    C = [[k * A[i][j] for j in range(2)] for i in range(2)]
    return (f"Compute {k}A for A = {A}.",
            [f"Multiply every entry by {k}", f"Result = {C}"],
            json.dumps(C))


def _mat_mul_tpl(o):
    A, B = _mat(o, "a"), _mat(o, "b")
    C = _mat_mul(A, B)
    return (f"Compute AB where A = {A} and B = {B}.",
            [f"c₁₁ = {A[0][0]}·{B[0][0]} + {A[0][1]}·{B[1][0]} = {C[0][0]}",
             f"c₁₂ = {A[0][0]}·{B[0][1]} + {A[0][1]}·{B[1][1]} = {C[0][1]}",
             f"Full result = {C}"],
            json.dumps(C))


def _mat_det(o):
    A = _mat(o, "a")
    det = A[0][0] * A[1][1] - A[0][1] * A[1][0]
    return (f"Find det(A) for A = {A}.",
            ["det = a₁₁a₂₂ − a₁₂a₂₁", f"= {A[0][0]}·{A[1][1]} − {A[0][1]}·{A[1][0]} = {det}"],
            str(det))


def _mat_transpose(o):
    A = _mat(o, "a")
    T = [[A[j][i] for j in range(2)] for i in range(2)]
    return (f"Compute Aᵀ for A = {A}.",
            ["Transpose flips rows and columns: (Aᵀ)ᵢⱼ = Aⱼᵢ", f"Aᵀ = {T}"],
            json.dumps(T))


def _linear_transform(o):
    A = [[0, -1], [1, 0]]
    v = [o["v0"], o["v1"]]
    result = [A[0][0]*v[0] + A[0][1]*v[1], A[1][0]*v[0] + A[1][1]*v[1]]
    return (f"Apply the 90° CCW rotation matrix A = [[0,-1],[1,0]] to v = {v}.",
            ["T(v) = Av",
             f"= [{A[0][0]}·{v[0]} + {A[0][1]}·{v[1]}, {A[1][0]}·{v[0]} + {A[1][1]}·{v[1]}] = {result}"],
            json.dumps(result))


def _deriv_power(o):
    a, n = o["a"], o["n"]
    ans = _poly_str([(a * n, n - 1)])
    return (f"Differentiate f(x) = {a}x^{n}.",
            ["Power Rule: d/dx[a·x^n] = a·n·x^(n-1)", f"f'(x) = {a}·{n}·x^{n-1} = {ans}"],
            ans)


def _deriv_polynomial(o):
    a, n, b, m, c = o["a"], o["n"], o["b"], o["m"], o["c"]
    fx = _poly_str([(a, n), (b, m), (c, 0)])
    ans = _poly_str([(a * n, n - 1), (b * m, m - 1)])
    return (f"Differentiate f(x) = {fx}.",
            ["Differentiate term by term with the Power Rule",
             f"The constant {c} differentiates to 0",
             f"f'(x) = {ans}"],
            ans)


def _deriv_at_point(o):
    a, b, x0 = o["a"], o["b"], o["x0"]
    slope = 2 * a * x0 + b
    fx = _poly_str([(a, 2), (b, 1)])
    return (f"Find the slope of f(x) = {fx} at x = {x0}.",
            [f"f'(x) = {2*a}x + {b}", f"f'({x0}) = {2*a}·{x0} + {b} = {slope}"],
            str(slope))


def _integral_power(o):
    n, coeff = o["n"], o["coeff"]
    n1 = n + 1
    a = coeff * n1
    ans = _poly_str([(coeff, n1)]) + " + C"
    return (f"Evaluate the indefinite integral: ∫ {_poly_str([(a, n)])} dx.",
            ["Power Rule: ∫ a·x^n dx = a/(n+1)·x^(n+1) + C", f"= {a}/{n1}·x^{n1} + C = {ans}"],
            ans)


def _integral_definite(o):
    n, coeff, b = o["n"], o["coeff"], o["b"]
    n1 = n + 1
    a = coeff * n1
    val = coeff * (b ** n1)
    return (f"Evaluate the definite integral: ∫ from 0 to {b} of {_poly_str([(a, n)])} dx.",
            [f"Antiderivative: {a}/{n1}·x^{n1} = {coeff}x^{n1}", f"Evaluate at {b}: {coeff}·{b}^{n1} = {val}"],
            str(val))


def _limit_substitution(o):
    a, b, c, x0 = o["a"], o["b"], o["c"], o["x0"]
    val = a * x0**2 + b * x0 + c
    fx = _poly_str([(a, 2), (b, 1), (c, 0)])
    return (f"Evaluate the limit: lim(x→{x0}) ({fx}).",
            [f"The polynomial is continuous — substitute x = {x0}", f"= {a}·{x0}^2 + {b}·{x0} + {c} = {val}"],
            str(val))


def _limit_factor(o):
    c = o["c"]
    val = 2 * c
    return (f"Evaluate the limit: lim(x→{c}) (x^2 − {c*c}) / (x − {c}).",
            [f"Factor the numerator: x^2 − {c*c} = (x − {c})(x + {c})",
             f"Cancel (x − {c}): lim(x→{c}) (x + {c}) = {val}"],
            str(val))


def _limit_infinity(o):
    a, b, n = o["a"], o["b"], o["n"]
    g = gcd(a, b)
    ra, rb = a // g, b // g
    ans = f"{ra}/{rb}" if rb != 1 else str(ra)
    return (f"Evaluate the limit: lim(x→∞) ({a}x^{n} + 1) / ({b}x^{n} + x).",
            [f"Divide numerator and denominator by x^{n}", f"Lower-order terms vanish → {a}/{b} = {ans}"],
            ans)


# ── registry ─────────────────────────────────────────────────────────────────────

_SPECS = [
    Template("arith_pemdas", ("a", "b", "c"), "int", _arith_pemdas),
    Template("arith_integer", ("a", "b", "op"), "int", _arith_integer, _arith_integer_canon, True),
    Template("arith_fractions", ("n1", "d1", "n2", "d2"), "rational", _arith_fractions),
    Template("algebra_linear", ("a", "x", "b"), "rational", _algebra_linear),
    Template("algebra_quadratic", ("r1", "r2"), "roots", _algebra_quadratic, _algebra_quadratic_canon, True),
    Template("func_evaluate", ("a", "b", "c", "x"), "int", _func_evaluate),
    Template("vec_magnitude", ("v0", "v1"), "surd", _vec_magnitude),
    Template("vec_dot", ("u0", "u1", "v0", "v1"), "int", _vec_dot),
    Template("vec_angle", ("u0", "u1", "v0", "v1"), "decimal", _vec_angle),
    Template("mat_add", _A + _B, "matrix", _mat_add),
    Template("mat_scalar", _A + ("k",), "matrix", _mat_scalar),
    Template("mat_mul", _A + _B, "matrix", _mat_mul_tpl),
    Template("mat_det", _A, "int", _mat_det),
    Template("mat_transpose", _A, "matrix", _mat_transpose),
    Template("linear_transform", ("v0", "v1"), "vector", _linear_transform),
    Template("deriv_power", ("a", "n"), "poly", _deriv_power),
    Template("deriv_polynomial", ("a", "n", "b", "m", "c"), "poly", _deriv_polynomial),
    Template("deriv_at_point", ("a", "b", "x0"), "int", _deriv_at_point),
    Template("integral_power", ("n", "coeff"), "antiderivative", _integral_power),
    Template("integral_definite", ("n", "coeff", "b"), "rational", _integral_definite),
    Template("limit_substitution", ("a", "b", "c", "x0"), "int", _limit_substitution),
    Template("limit_factor", ("c",), "int", _limit_factor),
    Template("limit_infinity", ("a", "b", "n"), "rational", _limit_infinity),
]

TEMPLATES: dict[str, Template] = {t.name: t for t in _SPECS}
