"""Oracle tests: independent answer derivation (fixes audit F4) and answer scoring (plan test 9)."""

import time

import pytest

from mathgen.eval.oracle import exact_match, math_correct, solve


# ── solve(): independent derivation from the question text ─────────────────────

@pytest.mark.parametrize("question, answer, atype", [
    ("Compute 7 + 5.", "12", "int"),
    ("Compute 30 - 2.", "28", "int"),
    ("Compute 12 * 3.", "36", "int"),
    ("Evaluate: 4 × (3 + 5)", "32", "int"),
    ("Add the fractions: 1/2 + 1/3", "5/6", "rational"),
    ("Add the fractions: 1/2 + 1/2", "1", "rational"),
    ("Solve for x: 3x + 4 = 19", "5", "rational"),
    ("Given f(x) = 2x² + 3x + 1, find f(-2).", "3", "int"),
    ("Find ||v|| for v = [3, 4].", "5", "surd"),
    ("Find ||v|| for v = [1, 2].", "sqrt(5)", "surd"),
    ("Compute u·v for u = [1, 2] and v = [3, 4].", "11", "int"),
    ("Find cos(θ) between u = [1, 0] and v = [1, 1].", "0.7071", "decimal"),
    ("Add A = [[1, 2], [3, 4]] and B = [[5, 6], [7, 8]].", "[[6, 8], [10, 12]]", "matrix"),
    ("Compute 3A for A = [[1, 2], [3, 4]].", "[[3, 6], [9, 12]]", "matrix"),
    ("Compute AB where A = [[1, 2], [3, 4]] and B = [[5, 6], [7, 8]].", "[[19, 22], [43, 50]]", "matrix"),
    ("Find det(A) for A = [[1, 2], [3, 4]].", "-2", "int"),
    ("Compute Aᵀ for A = [[1, 2], [3, 4]].", "[[1, 3], [2, 4]]", "matrix"),
    ("Apply the 90° CCW rotation matrix A = [[0,-1],[1,0]] to v = [3, 5].", "[-5, 3]", "vector"),
    ("Solve: x² - 1x - 6 = 0", "[-2, 3]", "roots"),
    ("Differentiate f(x) = 3x^4.", "12x^3", "poly"),
    ("Differentiate f(x) = 2x^3 + 5x + 7.", "6x^2 + 5", "poly"),
    ("Find the slope of f(x) = 3x^2 + 2x at x = 2.", "14", "int"),
    ("Evaluate the indefinite integral: ∫ 6x^2 dx.", "2x^3 + C", "antiderivative"),
    ("Evaluate the definite integral: ∫ from 0 to 2 of 6x^2 dx.", "16", "rational"),
    ("Evaluate the limit: lim(x→2) (x^2 + 3x + 1).", "11", "int"),
    ("Evaluate the limit: lim(x→3) (x^2 − 9) / (x − 3).", "6", "int"),
    ("Evaluate the limit: lim(x→∞) (4x^2 + 1) / (6x^2 + x).", "2/3", "rational"),
])
def test_solve_derives_answer_from_question(question, answer, atype):
    assert solve(question) == (answer, atype)


def test_solve_linear_uses_exact_division_not_floor():
    # Audit F4: the legacy verifier returned 8 (floor division).
    assert solve("Solve for x: 7x + 11 = 72") == ("61/7", "rational")


@pytest.mark.parametrize("question, answer", [
    ("Compute -3 + 5.", "2"),
    ("Find ||v|| for v = [-3, 4].", "5"),
    ("Compute u·v for u = [-1, 2] and v = [3, -4].", "-11"),
])
def test_solve_supports_negative_operands(question, answer):
    # Audit F4: the legacy verifier's (\d+) patterns silently returned None here.
    assert solve(question)[0] == answer


def test_solve_returns_none_for_unknown_question():
    assert solve("What is the meaning of life?") is None


def test_solve_returns_none_when_no_unique_solution():
    assert solve("Solve for x: 0x + 1 = 5") is None


# ── math_correct / exact_match ──────────────────────────────────────────────────

def test_unreduced_fraction_is_math_correct_but_not_exact():
    assert math_correct("10/12", "5/6", "rational")
    assert not exact_match("10/12", "5/6")


def test_integer_answer_equals_rational_truth():
    assert math_correct("1", "1", "rational")
    assert math_correct("2/2", "1", "rational")


def test_antiderivative_requires_plus_c():
    assert math_correct("2x^3 + C", "2x^3 + C", "antiderivative")
    assert not math_correct("2x^3", "2x^3 + C", "antiderivative")


def test_polynomial_compared_by_terms_not_text():
    assert math_correct("5 + 6x^2", "6x^2 + 5", "poly")
    assert math_correct("6*x**2 + 5", "6x^2 + 5", "poly")
    assert not math_correct("6x^2 + 4", "6x^2 + 5", "poly")


def test_decimal_tolerance_is_half_of_last_place():
    assert math_correct("0.70711", "0.7071", "decimal")
    assert not math_correct("0.7072", "0.7071", "decimal")


def test_surd_forms_are_equivalent():
    assert math_correct("sqrt(20)", "sqrt(20)", "surd")
    assert math_correct("2sqrt(5)", "sqrt(20)", "surd")
    assert not math_correct("sqrt(21)", "sqrt(20)", "surd")


def test_roots_compared_as_multisets():
    assert math_correct("[3, -2]", "[-2, 3]", "roots")
    assert not math_correct("[3, 3]", "[-2, 3]", "roots")


def test_matrix_and_vector_compared_elementwise():
    assert math_correct("[[6,8],[10,12]]", "[[6, 8], [10, 12]]", "matrix")
    assert not math_correct("[[6, 8], [10, 13]]", "[[6, 8], [10, 12]]", "matrix")
    assert math_correct("[-5,3]", "[-5, 3]", "vector")


@pytest.mark.parametrize("pred", [None, "", "abc", "[[1, 2]", "1/0", "nan", "inf", "x" * 10_000],
                         ids=["none", "empty", "word", "unclosed", "div0", "nan", "inf", "huge"])
def test_failures_count_as_wrong(pred):
    for atype in ("int", "rational", "decimal", "surd", "roots", "vector", "matrix", "poly", "antiderivative"):
        assert not math_correct(pred, "1", atype)


@pytest.mark.parametrize("pred, atype", [
    ("1/00x", "poly"), ("2/000x^2 + C", "antiderivative"),   # zero denominators must not crash
    ("3*", "poly"), ("12x^3 + 4*", "poly"),                  # dangling '*' is malformed
    ("٥", "int"), ("５", "int"), ("٥/١", "rational"),          # non-ASCII digits are not our grammar
])
def test_malformed_answers_are_wrong_not_errors(pred, atype):
    truth = {"poly": "3", "antiderivative": "x^3 + C", "int": "5", "rational": "5"}[atype]
    assert not math_correct(pred, truth, atype)


def test_pathological_input_is_rejected_fast():
    hostile = "9" * 5000 + "**" + "9" * 5000
    start = time.perf_counter()
    for atype in ("int", "rational", "poly", "antiderivative"):
        assert not math_correct(hostile, "1", atype)
    assert time.perf_counter() - start < 0.5
