"""scripts/pilots.py: stages that must survive being stopped and re-launched."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def pilots(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("pilots_under_test", ROOT / "scripts" / "pilots.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    monkeypatch.setattr(mod, "ART", tmp_path / "artifacts")
    mod.ART.mkdir()
    yield mod
    sys.modules.pop("pilots_under_test", None)


def write_log(run: Path, epochs: list[int]) -> None:
    run.mkdir(parents=True, exist_ok=True)
    (run / "train_log.jsonl").write_text("".join(json.dumps({"epoch": e}) + "\n" for e in epochs), encoding="utf-8")


def epochs_in(path: Path) -> list[int]:
    return [json.loads(l)["epoch"] for l in path.read_text(encoding="utf-8").splitlines()]


def test_a_complete_single_attempt_counts_as_trained(pilots, tmp_path):
    run = tmp_path / "run"
    write_log(run, list(range(1, 11)))
    (run / "best.pt").write_bytes(b"")
    assert pilots.legacy_train_done(run)


def test_two_interrupted_attempts_adding_up_to_ten_log_lines_do_not_count(pilots, tmp_path):
    """The legacy --fresh run appends to train_log.jsonl; 6 + 5 epochs of two stopped attempts is 11 lines."""
    run = tmp_path / "run"
    write_log(run, list(range(1, 7)) + list(range(1, 6)))
    (run / "best.pt").write_bytes(b"")
    assert not pilots.legacy_train_done(run)


def test_a_full_log_without_best_pt_does_not_count(pilots, tmp_path):
    run = tmp_path / "run"
    write_log(run, list(range(1, 11)))
    assert not pilots.legacy_train_done(run)


def test_p1_archives_a_stopped_attempt_before_training_fresh(pilots, tmp_path, monkeypatch):
    run = tmp_path / "runs" / "p1-legacy-repro-s0"
    write_log(run, [1, 2, 3, 4, 5, 6])
    (run / "best.pt").write_bytes(b"")
    calls = []

    def fake_sh(cmd, cwd=None):
        calls.append(cmd)
        if "src/train.py" in cmd:
            assert not (run / "train_log.jsonl").exists(), "the fresh attempt must start with an empty log"
            write_log(run, list(range(1, 11)))
        if "src/eval_accuracy.py" in cmd:
            (run / "accuracy_report.json").write_text(json.dumps({"model": {"correct": 1, "total": 2}}), encoding="utf-8")

    monkeypatch.setattr(pilots, "sh", fake_sh)
    pilots.p1()
    assert sum("src/train.py" in c for c in calls) == 1
    assert epochs_in(run / "train_log.jsonl") == list(range(1, 11))
    assert epochs_in(run / "train_log.attempt-1.jsonl") == [1, 2, 3, 4, 5, 6]
    record = json.loads((pilots.ART / "p1_legacy_repro.json").read_text(encoding="utf-8"))
    assert [e["epoch"] for e in record["train_log"]] == list(range(1, 11))
    assert record["stopped_attempts"] == ["train_log.attempt-1.jsonl"]


def test_p1_skips_training_once_a_complete_attempt_exists(pilots, tmp_path, monkeypatch):
    run = tmp_path / "runs" / "p1-legacy-repro-s0"
    write_log(run, list(range(1, 11)))
    (run / "best.pt").write_bytes(b"")
    calls = []

    def fake_sh(cmd, cwd=None):
        calls.append(cmd)
        if "src/eval_accuracy.py" in cmd:
            (run / "accuracy_report.json").write_text(json.dumps({}), encoding="utf-8")

    monkeypatch.setattr(pilots, "sh", fake_sh)
    pilots.p1()
    assert not any("src/train.py" in c for c in calls)
