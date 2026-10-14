"""
Multi-task training data for MathGen v2.

Three task types:
  [GEN]    → generate problem JSON
  [GRADE]  → grade student answer → feedback JSON
  [ASSESS] → assess skill from history → profile JSON

Data sources (in order of quality):
  1. GSM8K          — 8.5K real grade-school problems (HuggingFace)
  2. MATH dataset   — 12.5K competition problems (HuggingFace)
  3. Synthetic      — generated from topic templates (infinite)
  4. Grading        — auto-derived from synthetic problems
  5. Assessment     — synthetic answer history sequences
"""

import json
import math
import random
from pathlib import Path
from collections import defaultdict
from typing import Iterator

import torch
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────────────────────────────────────────────────────────
# Task 1 — GENERATE
# ─────────────────────────────────────────────────────────────────────────────

def _r(lo: int = 1, hi: int = 9) -> int:
    return random.randint(lo, hi)


def _vec2() -> list[int]:
    return [_r(1, 12), _r(1, 12)]


def _mat2() -> list[list[int]]:
    return [[_r(1, 9), _r(1, 9)], [_r(1, 9), _r(1, 9)]]


def _mat_mul(A: list[list[int]], B: list[list[int]]) -> list[list[int]]:
    return [
        [A[i][0]*B[0][j] + A[i][1]*B[1][j] for j in range(2)]
        for i in range(2)
    ]


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


# ── Per-topic generators ──────────────────────────────────────────────────────

class TopicGenerators:
    """Each method returns a (question_text, hints, answer_key) tuple."""

    # Basic math ───────────────────────────────────────────────────────────────

    @staticmethod
    def arith_pemdas() -> tuple[str, list[str], str]:
        a, b, c = _r(2, 9), _r(2, 9), _r(2, 9)
        ans = a * (b + c)
        return (
            f"Evaluate: {a} × ({b} + {c})",
            [f"Parentheses first: {b} + {c} = {b+c}", f"Then multiply: {a} × {b+c} = {ans}"],
            str(ans),
        )

    @staticmethod
    def arith_integer() -> tuple[str, list[str], str]:
        a, b = _r(2, 30), _r(2, 30)
        op = random.choice(["+", "-", "*"])
        if op == "+":  ans = a + b; s = f"{a} + {b} = {ans}"
        elif op == "-": a = max(a, b); b = min(a, b); ans = a - b; s = f"{a} - {b} = {ans}"
        else:          ans = a * b; s = f"{a} × {b} = {ans}"
        return (f"Compute {a} {op} {b}.", [s], str(ans))

    @staticmethod
    def arith_fractions() -> tuple[str, list[str], str]:
        n1, d1 = _r(1, 5), _r(2, 8)
        n2, d2 = _r(1, 5), _r(2, 8)
        from math import gcd
        lcd = d1 * d2 // gcd(d1, d2)
        num = n1 * (lcd // d1) + n2 * (lcd // d2)
        g = gcd(num, lcd)
        rn, rd = num // g, lcd // g
        ans = f"{rn}/{rd}" if rd != 1 else str(rn)
        return (
            f"Add the fractions: {n1}/{d1} + {n2}/{d2}",
            [f"LCD = {lcd}", f"= {n1*(lcd//d1)}/{lcd} + {n2*(lcd//d2)}/{lcd} = {num}/{lcd} = {ans}"],
            ans,
        )

    @staticmethod
    def algebra_linear() -> tuple[str, list[str], str]:
        a = _r(2, 8)
        x_val = _r(1, 10)
        b = _r(0, 15)
        c = a * x_val + b
        return (
            f"Solve for x: {a}x + {b} = {c}",
            [f"Subtract {b}: {a}x = {c - b}", f"Divide by {a}: x = {x_val}"],
            str(x_val),
        )

    @staticmethod
    def algebra_quadratic() -> tuple[str, list[str], str]:
        r1, r2 = _r(-6, 6), _r(-6, 6)
        b = -(r1 + r2)
        c = r1 * r2
        sign_b = f"+ {b}" if b >= 0 else f"- {-b}"
        sign_c = f"+ {c}" if c >= 0 else f"- {-c}"
        q = f"Solve: x² {sign_b}x {sign_c} = 0"
        ans = json.dumps(sorted([r1, r2]))
        return (
            q,
            [f"Factor: (x - {r1})(x - {r2}) = 0", f"x = {r1} or x = {r2}"],
            ans,
        )

    @staticmethod
    def func_evaluate() -> tuple[str, list[str], str]:
        a, b, c = _r(1, 5), _r(0, 8), _r(0, 10)
        x_val = _r(-4, 4)
        ans = a * x_val**2 + b * x_val + c
        return (
            f"Given f(x) = {a}x² + {b}x + {c}, find f({x_val}).",
            [f"f({x_val}) = {a}({x_val})² + {b}({x_val}) + {c}", f"= {a*x_val**2} + {b*x_val} + {c} = {ans}"],
            str(ans),
        )

    # Linear algebra ───────────────────────────────────────────────────────────

    @staticmethod
    def vec_magnitude() -> tuple[str, list[str], str]:
        v = _vec2()
        sq = v[0]**2 + v[1]**2
        mag = math.sqrt(sq)
        ans = str(int(mag)) if mag == int(mag) else f"sqrt({sq})"
        return (
            f"Find ||v|| for v = [{v[0]}, {v[1]}].",
            [f"||v|| = sqrt({v[0]}² + {v[1]}²) = sqrt({v[0]**2} + {v[1]**2})", f"= {ans}"],
            ans,
        )

    @staticmethod
    def vec_dot() -> tuple[str, list[str], str]:
        u, v = _vec2(), _vec2()
        dot = u[0]*v[0] + u[1]*v[1]
        return (
            f"Compute u·v for u = [{u[0]}, {u[1]}] and v = [{v[0]}, {v[1]}].",
            [f"u·v = {u[0]}·{v[0]} + {u[1]}·{v[1]} = {u[0]*v[0]} + {u[1]*v[1]}"],
            str(dot),
        )

    @staticmethod
    def vec_angle() -> tuple[str, list[str], str]:
        u, v = [_r(1, 5), _r(1, 5)], [_r(1, 5), _r(1, 5)]
        dot = u[0]*v[0] + u[1]*v[1]
        mag_u = math.sqrt(u[0]**2 + u[1]**2)
        mag_v = math.sqrt(v[0]**2 + v[1]**2)
        cos_theta = round(dot / (mag_u * mag_v), 4)
        return (
            f"Find cos(θ) between u = [{u[0]}, {u[1]}] and v = [{v[0]}, {v[1]}].",
            [f"cos(θ) = u·v / (||u|| ||v||)", f"= {dot} / ({round(mag_u,3)} × {round(mag_v,3)}) ≈ {cos_theta}"],
            str(cos_theta),
        )

    @staticmethod
    def mat_add() -> tuple[str, list[str], str]:
        A, B = _mat2(), _mat2()
        C = [[A[i][j] + B[i][j] for j in range(2)] for i in range(2)]
        return (
            f"Add A = {A} and B = {B}.",
            [f"Add entry by entry: (A+B)ᵢⱼ = aᵢⱼ + bᵢⱼ", f"Result = {C}"],
            json.dumps(C),
        )

    @staticmethod
    def mat_scalar() -> tuple[str, list[str], str]:
        A, k = _mat2(), _r(2, 5)
        C = [[k * A[i][j] for j in range(2)] for i in range(2)]
        return (
            f"Compute {k}A for A = {A}.",
            [f"Multiply every entry by {k}", f"Result = {C}"],
            json.dumps(C),
        )

    @staticmethod
    def mat_mul() -> tuple[str, list[str], str]:
        A, B = _mat2(), _mat2()
        C = _mat_mul(A, B)
        return (
            f"Compute AB where A = {A} and B = {B}.",
            [
                f"c₁₁ = {A[0][0]}·{B[0][0]} + {A[0][1]}·{B[1][0]} = {C[0][0]}",
                f"c₁₂ = {A[0][0]}·{B[0][1]} + {A[0][1]}·{B[1][1]} = {C[0][1]}",
                f"Full result = {C}",
            ],
            json.dumps(C),
        )

    @staticmethod
    def mat_det() -> tuple[str, list[str], str]:
        A = _mat2()
        det = A[0][0] * A[1][1] - A[0][1] * A[1][0]
        return (
            f"Find det(A) for A = {A}.",
            [f"det = a₁₁a₂₂ − a₁₂a₂₁", f"= {A[0][0]}·{A[1][1]} − {A[0][1]}·{A[1][0]} = {det}"],
            str(det),
        )

    @staticmethod
    def mat_transpose() -> tuple[str, list[str], str]:
        A = _mat2()
        T = [[A[j][i] for j in range(2)] for i in range(2)]
        return (
            f"Compute Aᵀ for A = {A}.",
            [f"Transpose flips rows and columns: (Aᵀ)ᵢⱼ = Aⱼᵢ", f"Aᵀ = {T}"],
            json.dumps(T),
        )

    @staticmethod
    def linear_transform() -> tuple[str, list[str], str]:
        A = [[0, -1], [1, 0]]   # 90° rotation
        v = _vec2()
        result = [A[0][0]*v[0] + A[0][1]*v[1], A[1][0]*v[0] + A[1][1]*v[1]]
        return (
            f"Apply the 90° CCW rotation matrix A = [[0,-1],[1,0]] to v = {v}.",
            [f"T(v) = Av", f"= [{A[0][0]}·{v[0]} + {A[0][1]}·{v[1]}, {A[1][0]}·{v[0]} + {A[1][1]}·{v[1]}] = {result}"],
            json.dumps(result),
        )

    # Calculus ───────────────────────────────────────────────────────────────

    @staticmethod
    def deriv_power() -> tuple[str, list[str], str]:
        a, n = _r(2, 6), _r(2, 5)
        ans = _poly_str([(a * n, n - 1)])
        return (
            f"Differentiate f(x) = {a}x^{n}.",
            [f"Power Rule: d/dx[a·x^n] = a·n·x^(n-1)",
             f"f'(x) = {a}·{n}·x^{n-1} = {ans}"],
            ans,
        )

    @staticmethod
    def deriv_polynomial() -> tuple[str, list[str], str]:
        a, n = _r(2, 5), _r(3, 5)
        b, m = _r(2, 6), _r(1, 2)
        c = _r(1, 9)
        fx  = _poly_str([(a, n), (b, m), (c, 0)])
        ans = _poly_str([(a * n, n - 1), (b * m, m - 1)])
        return (
            f"Differentiate f(x) = {fx}.",
            [f"Differentiate term by term with the Power Rule",
             f"The constant {c} differentiates to 0",
             f"f'(x) = {ans}"],
            ans,
        )

    @staticmethod
    def deriv_at_point() -> tuple[str, list[str], str]:
        a, b = _r(2, 5), _r(2, 6)
        x0 = _r(1, 4)
        slope = 2 * a * x0 + b
        fx = _poly_str([(a, 2), (b, 1)])
        return (
            f"Find the slope of f(x) = {fx} at x = {x0}.",
            [f"f'(x) = {2*a}x + {b}",
             f"f'({x0}) = {2*a}·{x0} + {b} = {slope}"],
            str(slope),
        )

    @staticmethod
    def integral_power() -> tuple[str, list[str], str]:
        n = _r(1, 4)
        n1 = n + 1
        coeff = _r(1, 3)
        a = coeff * n1                      # keep the integrated coefficient an integer
        ans = _poly_str([(coeff, n1)]) + " + C"
        return (
            f"Evaluate the indefinite integral: ∫ {_poly_str([(a, n)])} dx.",
            [f"Power Rule: ∫ a·x^n dx = a/(n+1)·x^(n+1) + C",
             f"= {a}/{n1}·x^{n1} + C = {ans}"],
            ans,
        )

    @staticmethod
    def integral_definite() -> tuple[str, list[str], str]:
        n = _r(1, 3)
        n1 = n + 1
        coeff = _r(1, 3)
        a = coeff * n1
        b = _r(2, 4)
        val = coeff * (b ** n1)
        return (
            f"Evaluate the definite integral: ∫ from 0 to {b} of {_poly_str([(a, n)])} dx.",
            [f"Antiderivative: {a}/{n1}·x^{n1} = {coeff}x^{n1}",
             f"Evaluate at {b}: {coeff}·{b}^{n1} = {val}"],
            str(val),
        )

    @staticmethod
    def limit_substitution() -> tuple[str, list[str], str]:
        a, b, c = _r(1, 4), _r(1, 6), _r(1, 9)
        x0 = _r(1, 4)
        val = a * x0**2 + b * x0 + c
        fx = _poly_str([(a, 2), (b, 1), (c, 0)])
        return (
            f"Evaluate the limit: lim(x→{x0}) ({fx}).",
            [f"The polynomial is continuous — substitute x = {x0}",
             f"= {a}·{x0}^2 + {b}·{x0} + {c} = {val}"],
            str(val),
        )

    @staticmethod
    def limit_factor() -> tuple[str, list[str], str]:
        c = _r(2, 6)
        val = 2 * c
        return (
            f"Evaluate the limit: lim(x→{c}) (x^2 − {c*c}) / (x − {c}).",
            [f"Factor the numerator: x^2 − {c*c} = (x − {c})(x + {c})",
             f"Cancel (x − {c}): lim(x→{c}) (x + {c}) = {val}"],
            str(val),
        )

    @staticmethod
    def limit_infinity() -> tuple[str, list[str], str]:
        from math import gcd
        a, b = _r(2, 6), _r(2, 6)
        n = _r(2, 3)
        g = gcd(a, b)
        ra, rb = a // g, b // g
        ans = f"{ra}/{rb}" if rb != 1 else str(ra)
        return (
            f"Evaluate the limit: lim(x→∞) ({a}x^{n} + 1) / ({b}x^{n} + x).",
            [f"Divide numerator and denominator by x^{n}",
             f"Lower-order terms vanish → {a}/{b} = {ans}"],
            ans,
        )


# Map: topic_id → list of generator methods per difficulty
TOPIC_GENERATORS: dict[str, dict[str, list]] = {
    "basic-math/arithmetic": {
        "easy":   [TopicGenerators.arith_integer],
        "medium": [TopicGenerators.arith_pemdas, TopicGenerators.arith_integer],
        "hard":   [TopicGenerators.arith_pemdas, TopicGenerators.arith_fractions],
    },
    "basic-math/algebra-basics": {
        "easy":   [TopicGenerators.algebra_linear],
        "medium": [TopicGenerators.algebra_linear, TopicGenerators.func_evaluate],
        "hard":   [TopicGenerators.algebra_quadratic],
    },
    "basic-math/functions": {
        "easy":   [TopicGenerators.func_evaluate],
        "medium": [TopicGenerators.func_evaluate, TopicGenerators.algebra_linear],
        "hard":   [TopicGenerators.algebra_quadratic, TopicGenerators.func_evaluate],
    },
    "linear-algebra/vectors": {
        "easy":   [TopicGenerators.vec_magnitude, TopicGenerators.vec_dot],
        "medium": [TopicGenerators.vec_dot, TopicGenerators.vec_angle],
        "hard":   [TopicGenerators.vec_angle, TopicGenerators.vec_dot],
    },
    "linear-algebra/matrices": {
        "easy":   [TopicGenerators.mat_add, TopicGenerators.mat_scalar],
        "medium": [TopicGenerators.mat_add, TopicGenerators.mat_transpose],
        "hard":   [TopicGenerators.mat_det, TopicGenerators.mat_transpose],
    },
    "linear-algebra/matrix-multiplication": {
        "easy":   [TopicGenerators.mat_mul],
        "medium": [TopicGenerators.mat_mul, TopicGenerators.mat_det],
        "hard":   [TopicGenerators.mat_mul, TopicGenerators.linear_transform],
    },
    "linear-algebra/linear-transformations": {
        "easy":   [TopicGenerators.linear_transform, TopicGenerators.mat_mul],
        "medium": [TopicGenerators.linear_transform, TopicGenerators.mat_det],
        "hard":   [TopicGenerators.linear_transform, TopicGenerators.mat_mul],
    },
    "calculus/limits": {
        "easy":   [TopicGenerators.limit_substitution],
        "medium": [TopicGenerators.limit_substitution, TopicGenerators.limit_factor],
        "hard":   [TopicGenerators.limit_factor, TopicGenerators.limit_infinity],
    },
    "calculus/derivatives": {
        "easy":   [TopicGenerators.deriv_power],
        "medium": [TopicGenerators.deriv_polynomial, TopicGenerators.deriv_at_point],
        "hard":   [TopicGenerators.deriv_polynomial, TopicGenerators.deriv_at_point],
    },
    "calculus/integrals": {
        "easy":   [TopicGenerators.integral_power],
        "medium": [TopicGenerators.integral_power, TopicGenerators.integral_definite],
        "hard":   [TopicGenerators.integral_definite],
    },
}


def _gen_problem(topic_id: str, difficulty: str) -> dict | None:
    gens = TOPIC_GENERATORS.get(topic_id, {}).get(difficulty, [])
    if not gens:
        return None
    fn = random.choice(gens)
    q, hints, ans = fn()
    return {"questionText": q, "hints": hints, "answerKey": ans}


def format_generate(topic_id: str, difficulty: str, problem: dict) -> str:
    prompt = f"[GEN] topic:{topic_id} difficulty:{difficulty} <SEP><JSON>"
    out    = json.dumps({
        "questionText": problem["questionText"],
        "hints":        problem["hints"],
        "answerKey":    problem["answerKey"],
    }) + "<EOS>"
    return prompt + out


# ─────────────────────────────────────────────────────────────────────────────
# Task 2 — GRADE
# ─────────────────────────────────────────────────────────────────────────────

def _perturb_answer(ans: str) -> str:
    """Generate a plausible wrong answer."""
    try:
        n = float(ans)
        return str(int(n + random.choice([-2, -1, 1, 2, 3])))
    except (ValueError, TypeError):
        return ans + "X"   # slightly wrong


GRADE_EXPLANATIONS = {
    True: [
        "Correct! You applied the method accurately.",
        "Perfect — your computation is right.",
        "Excellent work. The answer is exactly right.",
        "That's correct! Great understanding of the concept.",
    ],
    False: [
        "Not quite — check your arithmetic in the last step.",
        "Incorrect. Review the formula and try again.",
        "Close, but there's an error. Re-read the definition.",
        "That's not right. Look at the hints for guidance.",
    ],
}


def format_grade(topic_id: str, question: str, student_ans: str, correct_ans: str) -> str:
    is_correct = student_ans.strip().replace(" ", "") == correct_ans.strip().replace(" ", "")
    explanation = random.choice(GRADE_EXPLANATIONS[is_correct])
    score = 100 if is_correct else random.choice([0, 10, 20])
    hint  = "" if is_correct else "Review the step-by-step solution and try again."

    prompt = (
        f"[GRADE] topic:{topic_id} "
        f"question:{question[:80]} "
        f"student_answer:{student_ans} "
        f"correct_answer:{correct_ans} <SEP><JSON>"
    )
    out = json.dumps({
        "correct":     is_correct,
        "score":       score,
        "explanation": explanation,
        "hint":        hint,
    }) + "<EOS>"
    return prompt + out


def generate_grading_samples(n: int = 3000) -> list[str]:
    samples: list[str] = []
    topics = list(TOPIC_GENERATORS.keys())
    diffs  = ["easy", "medium", "hard"]

    for _ in range(n):
        tid  = random.choice(topics)
        diff = random.choice(diffs)
        prob = _gen_problem(tid, diff)
        if prob is None:
            continue
        # 60% correct, 40% wrong
        if random.random() < 0.6:
            student_ans = prob["answerKey"]
        else:
            student_ans = _perturb_answer(prob["answerKey"])
        samples.append(format_grade(tid, prob["questionText"], student_ans, prob["answerKey"]))

    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Task 3 — ASSESS
# ─────────────────────────────────────────────────────────────────────────────

SKILL_MESSAGES = {
    range(0, 3):   ("Beginner", "easy",   "Keep practicing the basics — you're building foundations!"),
    range(3, 5):   ("Developing", "easy", "Good progress! Focus on getting the easy problems solid."),
    range(5, 7):   ("Intermediate", "medium", "You're on track. Challenge yourself with medium problems."),
    range(7, 9):   ("Advanced", "medium", "Impressive — you've mastered the core. Try harder problems!"),
    range(9, 11):  ("Expert", "hard",   "Outstanding mastery. You're ready for competition-level problems."),
}


def _skill_from_history(history: list[dict]) -> tuple[int, str]:
    """Compute skill level 0-10 from recent answer history."""
    if not history:
        return 3, "easy"
    correct = [h["correct"] for h in history[-10:]]
    diff_weights = {"easy": 1, "medium": 2, "hard": 3}
    weighted_score = sum(
        diff_weights.get(h["difficulty"], 1) * h["correct"]
        for h in history[-10:]
    )
    max_score = sum(diff_weights.get(h["difficulty"], 1) for h in history[-10:])
    skill = min(10, round((weighted_score / max(max_score, 1)) * 10))
    next_diff = "easy"
    for r, (_, nd, _) in SKILL_MESSAGES.items():
        if skill in r:
            next_diff = nd
            break
    return skill, next_diff


def format_assess(history: list[dict]) -> str:
    skill, next_diff = _skill_from_history(history)
    label, _, msg = next(
        (v for r, v in SKILL_MESSAGES.items() if skill in r),
        ("Intermediate", "medium", "Keep going!")
    )
    recent_wrong = [h for h in history[-5:] if not h["correct"]]
    weaknesses   = list({h["topic"] for h in recent_wrong})[:2] if recent_wrong else []

    history_str = json.dumps([
        {"topic": h["topic"], "correct": h["correct"], "difficulty": h["difficulty"]}
        for h in history[-8:]
    ])
    prompt = f"[ASSESS] history:{history_str} <SEP><JSON>"
    out    = json.dumps({
        "skill_level":       skill,
        "label":             label,
        "next_difficulty":   next_diff,
        "weaknesses":        weaknesses,
        "message":           msg,
    }) + "<EOS>"
    return prompt + out


def generate_assessment_samples(n: int = 2000) -> list[str]:
    topics = list(TOPIC_GENERATORS.keys())
    diffs  = ["easy", "medium", "hard"]
    samples: list[str] = []

    for _ in range(n):
        # Simulate a student's session history
        skill_bias = random.uniform(0.2, 0.9)   # student's true skill
        history = []
        for _ in range(random.randint(5, 15)):
            diff = random.choice(diffs)
            p_correct = skill_bias * (0.9 if diff == "easy" else 0.65 if diff == "medium" else 0.4)
            history.append({
                "topic":      random.choice(topics),
                "difficulty": diff,
                "correct":    random.random() < p_correct,
            })
        samples.append(format_assess(history))

    return samples


# ─────────────────────────────────────────────────────────────────────────────
# External dataset loading (GSM8K + MATH via HuggingFace)
# ─────────────────────────────────────────────────────────────────────────────

def load_gsm8k() -> list[str]:
    """Load GSM8K grade-school math problems and format as [GEN] samples."""
    try:
        from datasets import load_dataset
        ds = load_dataset("gsm8k", "main", split="train")
        samples = []
        for ex in ds:
            q = ex["question"]
            a = ex["answer"].split("####")[-1].strip()
            steps = [s.strip() for s in ex["answer"].split("\n") if s.strip() and "####" not in s]
            problem = {"questionText": q, "hints": steps[:3], "answerKey": a}
            samples.append(format_generate("basic-math/arithmetic", "medium", problem))
        print(f"GSM8K: loaded {len(samples)} problems")
        return samples
    except Exception as e:
        print(f"GSM8K load failed ({e}), skipping")
        return []


def load_math_dataset() -> list[str]:
    """Load MATH competition dataset (Hendrycks et al.) and format as [GEN] samples."""
    try:
        from datasets import load_dataset
        ds = load_dataset("lighteval/MATH", "all", split="train", trust_remote_code=True)
        topic_map = {
            "algebra":          "basic-math/algebra-basics",
            "prealgebra":       "basic-math/arithmetic",
            "number_theory":    "basic-math/arithmetic",
            "precalculus":      "basic-math/functions",
            "intermediate_algebra": "basic-math/algebra-basics",
            "linear_algebra":   "linear-algebra/matrices",
            "geometry":         "basic-math/functions",
        }
        diff_map = {"1": "easy", "2": "easy", "3": "medium", "4": "hard", "5": "hard"}
        samples = []
        for ex in ds:
            topic    = topic_map.get(ex.get("type", ""), "basic-math/arithmetic")
            diff     = diff_map.get(str(ex.get("level", 3)), "medium")
            problem  = {
                "questionText": ex["problem"],
                "hints":        [ex["solution"][:200]],
                "answerKey":    ex["solution"].split("\\boxed{")[-1].split("}")[0] if "\\boxed{" in ex["solution"] else ex["solution"][-30:],
            }
            samples.append(format_generate(topic, diff, problem))
        print(f"MATH dataset: loaded {len(samples)} problems")
        return samples
    except Exception as e:
        print(f"MATH dataset load failed ({e}), skipping")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Main dataset builder
# ─────────────────────────────────────────────────────────────────────────────

def build_all_samples(n_synthetic_per_topic: int = 2000) -> list[str]:
    samples: list[str] = []

    # Task 1: Synthetic generation samples
    print("Generating synthetic [GEN] samples...")
    for tid in TOPIC_GENERATORS:
        for diff in ["easy", "medium", "hard"]:
            for _ in range(n_synthetic_per_topic):
                prob = _gen_problem(tid, diff)
                if prob:
                    samples.append(format_generate(tid, diff, prob))

    print(f"  Synthetic [GEN]: {len(samples)}")

    gen_count = len(samples)
    print(f"  Total [GEN]: {gen_count}  (HuggingFace datasets skipped for offline training)")

    # Task 2: Grading samples
    print("Generating [GRADE] samples...")
    grade = generate_grading_samples(n=max(3000, gen_count // 2))
    samples.extend(grade)
    print(f"  [GRADE]: {len(grade)}")

    # Task 3: Assessment samples
    print("Generating [ASSESS] samples...")
    assess = generate_assessment_samples(n=2000)
    samples.extend(assess)
    print(f"  [ASSESS]: {len(assess)}")

    random.shuffle(samples)
    print(f"\nTotal training samples: {len(samples)}")
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────────

class MathDataset(Dataset):
    def __init__(self, tokens: list[list[int]], context_len: int, pad_id: int):
        self.data       = tokens
        self.ctx        = context_len
        self.pad_id     = pad_id

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        t = self.data[idx][:self.ctx + 1]
        if len(t) < self.ctx + 1:
            t = t + [self.pad_id] * (self.ctx + 1 - len(t))
        x = torch.tensor(t[:-1], dtype=torch.long)
        y = torch.tensor(t[1:],  dtype=torch.long)
        y[y == self.pad_id] = -1
        return x, y


def build_dataloader(
    samples:     list[str],
    tokenizer,
    context_len: int = 512,
    batch_size:  int = 8,
    shuffle:     bool = True,
    n_workers:   int = 0,
) -> DataLoader:
    print("Tokenizing dataset...")
    tokens = [tokenizer.encode(s) for s in samples]
    tokens = [t for t in tokens if len(t) > 4]   # drop near-empty
    ds = MathDataset(tokens, context_len, tokenizer.pad_id)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=n_workers, pin_memory=False)
