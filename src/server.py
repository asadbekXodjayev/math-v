"""
MathGen inference server — three endpoints:
  POST /generate  → AI-generated math problem
  POST /grade     → grade student answer + explanation
  POST /assess    → skill level assessment from answer history
"""

import json
import os
import re
import random
from pathlib import Path

# ── WMI hang workaround (Python 3.14 + Windows on this machine) ──────────────────
# torch's import calls platform.machine(); on Py3.14/Win that routes through
# platform._wmi_query(), and this box's WMI service hangs indefinitely (cold),
# stalling `import torch` forever. Pre-fill the uname cache and neutralise the WMI
# helpers so uname()/machine()/platform() resolve from constants instead of WMI.
import platform as _platform
try:
    _uc = _platform.uname_result("Windows", "localhost", "10", "10.0", "AMD64")
    _uc.__dict__["processor"] = ""  # prime lazy cached_property -> no WMI on access
    _platform._uname_cache = _uc
except Exception:
    pass
_platform.win32_ver = lambda *a, **k: ("10", "10.0", "", "")
_platform._wmi_query = lambda *a, **k: None
# ────────────────────────────────────────────────────────────────────────────────

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model import MathGen, MathGenConfig
from tokenizer import BPETokenizer
from dataset import (
    format_generate, format_grade, format_assess, _gen_problem,
    _skill_from_history, SKILL_MESSAGES, GRADE_EXPLANATIONS,
)
from verifier import compute_true_answer, smart_equal

CKPT_DIR   = Path(__file__).parent.parent / "checkpoints"
BEST_PT    = CKPT_DIR / "best.pt"
TOK_PATH   = str(CKPT_DIR / "tokenizer.json")

app = FastAPI(title="MathGen API v2")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model:     MathGen | None     = None
tokenizer: BPETokenizer | None = None
# Inference is forced to CPU: GPU execution on this RTX 3050 throws cudaErrorUnknown,
# and generation is light enough that CPU latency is fine for the local demo loop.
device = torch.device("cpu")


@app.on_event("startup")
async def load() -> None:
    global model, tokenizer
    if not BEST_PT.exists():
        print(f"No model at {BEST_PT} — run train.py first. Server will respond with 503 until trained.")
        return

    tokenizer = BPETokenizer()
    tokenizer.load(TOK_PATH)

    cfg = MathGenConfig(vocab_size=len(tokenizer), use_grad_ckpt=False)
    model = MathGen(cfg).to(device)
    ckpt = torch.load(BEST_PT, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"MathGen loaded | {model.num_params()/1e6:.1f}M params | {device} | val={ckpt.get('val_loss',0):.4f}")


def _infer(prompt: str, max_new: int = 400, temperature: float = 0.7) -> str:
    if model is None or tokenizer is None:
        raise HTTPException(503, "Model not loaded. Run train.py first.")
    ids = tokenizer.encode(prompt)
    inp = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        out = model.generate(inp, max_new=max_new, temperature=temperature)
    gen_ids = out[0, len(ids):].tolist()
    return tokenizer.decode(gen_ids)


def _try_extract_json(text: str) -> dict | None:
    """Best-effort: return parsed JSON object, or None if the output isn't valid JSON."""
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return None


# ── Schemas ───────────────────────────────────────────────────────────────────

class GenerateReq(BaseModel):
    topic_id:     str
    difficulty:   str = "easy"
    problem_type: str = "auto"
    temperature:  float = 0.75

class GenerateRes(BaseModel):
    questionText: str
    hints:        list[str]
    answerKey:    str | int | list

class GradeReq(BaseModel):
    topic_id:      str
    question:      str
    student_answer: str
    correct_answer: str

class GradeRes(BaseModel):
    correct:     bool
    score:       int
    explanation: str
    hint:        str

class HistoryEntry(BaseModel):
    topic:      str
    difficulty: str
    correct:    bool

class AssessReq(BaseModel):
    history: list[HistoryEntry]

class AssessRes(BaseModel):
    skill_level:     int
    label:           str
    next_difficulty: str
    weaknesses:      list[str]
    message:         str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model": model is not None, "device": str(device)}


# Problem source: "generator" (deterministic, verified-correct templates — DEFAULT)
# or "model" (the neural net, experimental). At 36.7M params the model learned the
# output FORMAT but not reliable arithmetic (e.g. it emits "10 + 17" → 25), which is
# unsafe to grade a student against — so generation defaults to the verified generators
# in dataset.py (the very templates the model was trained to imitate). Flip with
# MATHGEN_SOURCE=model to use the neural net.
GEN_SOURCE = os.environ.get("MATHGEN_SOURCE", "generator").lower()


@app.post("/generate", response_model=GenerateRes)
def generate(req: GenerateReq):
    # Default path: deterministic, guaranteed-correct generator.
    if GEN_SOURCE != "model":
        prob = _gen_problem(req.topic_id, req.difficulty)
        if prob:
            return GenerateRes(questionText=prob["questionText"], hints=prob["hints"], answerKey=prob["answerKey"])
        # no template for this topic/difficulty → fall through to the model

    # Neural model path (experimental) — VERIFIER-IN-THE-LOOP: only serve a
    # model-generated problem if its answer can be INDEPENDENTLY verified, and
    # replace the model's (often wrong) answerKey with the verified-correct one.
    # This guarantees every served answer is correct even when the net miscomputes.
    prompt = f"[GEN] topic:{req.topic_id} difficulty:{req.difficulty} <SEP><JSON>"
    for _ in range(3):
        text = _infer(prompt, max_new=350, temperature=req.temperature)
        data = _try_extract_json(text)
        if data and data.get("questionText"):
            truth = compute_true_answer(data["questionText"])
            if truth is not None:
                return GenerateRes(
                    questionText=data["questionText"],
                    hints       =data.get("hints", []),
                    answerKey   =truth[0],          # verified-correct, not model-guessed
                )
    # Fall back to the verified generator.
    prob = _gen_problem(req.topic_id, req.difficulty)
    if prob:
        return GenerateRes(questionText=prob["questionText"], hints=prob["hints"], answerKey=prob["answerKey"])
    raise HTTPException(503, "Could not produce a valid problem")


@app.post("/grade", response_model=GradeRes)
def grade(req: GradeReq):
    # Correctness is decided DETERMINISTICALLY (robust numeric/fraction/json/symbolic
    # compare) — never by the neural net, which mis-grades. The LM is not the source
    # of truth for whether a student is right.
    is_correct = smart_equal(req.student_answer, req.correct_answer)
    explanation = random.choice(GRADE_EXPLANATIONS[is_correct])
    hint = "" if is_correct else "Review the step-by-step solution and try again."
    return GradeRes(
        correct    =is_correct,
        score      =100 if is_correct else 0,
        explanation=explanation,
        hint       =hint,
    )


@app.post("/assess", response_model=AssessRes)
def assess(req: AssessReq):
    # Skill assessment is a DETERMINISTIC function of answer history (weighted by
    # difficulty) — computed directly, not hallucinated by the net.
    history = [h.model_dump() for h in req.history]
    skill, next_diff = _skill_from_history(history)
    label, _, msg = next(
        (v for r, v in SKILL_MESSAGES.items() if skill in r),
        ("Intermediate", "medium", "Keep going!"),
    )
    recent_wrong = [h for h in history[-5:] if not h["correct"]]
    weaknesses = list({h["topic"] for h in recent_wrong})[:2] if recent_wrong else []
    return AssessRes(
        skill_level    =skill,
        label          =label,
        next_difficulty=next_diff,
        weaknesses     =weaknesses,
        message        =msg,
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("MATHGEN_PORT", "8787"))
    print(f"MathGen API starting on :{port}")
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False, workers=1)
