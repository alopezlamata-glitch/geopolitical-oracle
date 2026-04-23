from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from evaluation.walkforward import evaluate_walkforward


def _dataset(n: int = 40):
    base = datetime(2023, 1, 1, tzinfo=timezone.utc)
    data = []
    for i in range(n):
        data.append(
            {
                "timestamp": (base + timedelta(days=i)).isoformat(),
                "outcome": int(i % 2 == 0),
                "horizon": 30 if i < n // 2 else 60,
                "predicate": "resign" if i % 3 else "coup",
                "event_family": "political" if i % 3 else "conflict",
            }
        )
    return data


def test_evaluate_walkforward_persists_output(tmp_path):
    def predictor_fn(train_rows, test_rows):
        _ = train_rows
        return [0.7] * len(test_rows)

    out = tmp_path / "eval.json"
    result = evaluate_walkforward(
        _dataset(),
        predictor_fn,
        min_train_size=20,
        test_size=10,
        step_size=10,
        min_breakdown_support=5,
        output_path=out,
    )

    assert out.exists()
    loaded = json.loads(out.read_text())
    assert loaded["metadata"]["version"] == "baseline_v1"
    assert loaded["aggregate"]["n_examples"] == 20
    assert result["breakdown"]["predicate"]
    assert result["breakdown"]["event_family"]


def test_evaluate_walkforward_anti_leakage_check():
    bad = _dataset()
    bad[20]["timestamp"] = bad[19]["timestamp"]

    def predictor_fn(train_rows, test_rows):
        _ = train_rows, test_rows
        return [0.5] * len(test_rows)

    try:
        evaluate_walkforward(bad, predictor_fn, min_train_size=20, test_size=10)
        raise AssertionError("Expected ValueError due to leakage")
    except ValueError as exc:
        assert "Temporal leakage detected" in str(exc)
