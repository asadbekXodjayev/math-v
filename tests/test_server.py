"""Plan test 12: the FastAPI contract mathvis-app depends on, and the served generator's F3 fix."""

import importlib
import os
import random
import re

import pytest


@pytest.fixture(scope="module")
def client(legacy_dataset):
    from fastapi.testclient import TestClient
    os.environ["MATHGEN_SOURCE"] = "generator"
    server = importlib.reload(importlib.import_module("server"))   # GEN_SOURCE is read at import time
    with TestClient(server.app) as c:
        yield c


def test_generate_returns_a_valid_problem_for_all_30_topic_difficulty_pairs(client, legacy_dataset):
    pairs = [(t, d) for t in legacy_dataset.TOPIC_GENERATORS for d in ("easy", "medium", "hard")]
    assert len(pairs) == 30
    for topic, diff in pairs:
        r = client.post("/generate", json={"topic_id": topic, "difficulty": diff})
        body = r.json()
        assert r.status_code == 200, (topic, diff, r.text[:200])
        assert isinstance(body["questionText"], str) and isinstance(body["hints"], list) and "answerKey" in body


def test_health(client):
    assert client.get("/health").status_code == 200


def test_served_subtraction_is_not_biased_to_x_minus_x(legacy_dataset):
    """Audit F3 also affected the deployed /generate: ~51% of subtractions were x - x."""
    random.seed(0)
    same = total = 0
    for _ in range(6000):
        q = legacy_dataset._gen_problem("basic-math/arithmetic", "easy")["questionText"]
        if m := re.fullmatch(r"Compute (\d+) - (\d+)\.", q):
            total += 1
            same += m.group(1) == m.group(2)
            assert int(m.group(1)) >= int(m.group(2))
    assert total > 1000 and same / total < 0.08   # expected 1/29 = 3.4%
