"""
Prompt/target serialization for the conditional-solving task (RESEARCH_DESIGN.md §2).
JSON is written with ensure_ascii=False, so there are no `\\uXXXX` escapes (audit F8).
"""

import json

SPECIALS = ("[SOLVE]", "<SEP>", "<JSON>", "<EOS>")
ORDERS = ("hints_first", "answer_first")


def prompt_text(question: str) -> str:
    return f"[SOLVE] {question} <SEP><JSON>"


def target_text(hints: list[str], answer: str, order: str = "hints_first") -> str:
    if order == "hints_first":
        obj = {"hints": hints, "answerKey": answer}
    elif order == "answer_first":
        obj = {"answerKey": answer, "hints": hints}
    else:
        raise ValueError(f"unknown order {order!r}")
    return json.dumps(obj, ensure_ascii=False) + "<EOS>"


def tchar_len(text: str) -> int:
    """Sequence length under the character tokenizer: one token per character, one per special."""
    return len(text) - sum(text.count(s) * (len(s) - 1) for s in SPECIALS)
