from __future__ import annotations

import numpy as np
import pytest

import scripts.backtest as backtest


def _example(ts: str, outcome: int = 0) -> dict:
    return {"ts": ts, "outcome": outcome, "features": np.array([0.1], dtype=np.float32)}


def test_run_folds_asserts_no_temporal_leakage():
    # intentionally leaking: test starts before latest train date
    folds = [(
        "leaky",
        [_example("2022-01-10", 0), _example("2022-01-11", 1)],
        [_example("2022-01-09", 1), _example("2022-01-12", 0)],
    )]

    with pytest.raises(AssertionError, match="Temporal leakage"):
        backtest._run_folds(folds)


def test_run_folds_emits_required_metrics(monkeypatch):
    folds = [(
        "ok",
        [_example("2022-01-01", 0), _example("2022-01-02", 1), _example("2022-01-03", 0), _example("2022-01-04", 1), _example("2022-01-05", 0), _example("2022-01-06", 1), _example("2022-01-07", 0), _example("2022-01-08", 1), _example("2022-01-09", 0), _example("2022-01-10", 1), _example("2022-01-11", 0), _example("2022-01-12", 1)],
        [_example("2022-02-01", 0), _example("2022-02-02", 1), _example("2022-02-03", 0), _example("2022-02-04", 1), _example("2022-02-05", 1)],
    )]

    monkeypatch.setattr(backtest, "_train_fold", lambda *args, **kwargs: (object(), object(), "platt"))

    def fake_predict(_m, _c, _method, X):
        p = np.full((X.shape[0],), 0.6, dtype=float)
        return p, p

    monkeypatch.setattr(backtest, "_predict_fold", fake_predict)

    results = backtest._run_folds(folds)
    assert len(results) == 1

    row = results[0]
    for key in ("brier", "log_loss", "ece", "n_test"):
        assert key in row

    # roc_auc is conditional: can be NaN only if test fold has one class.
    assert "roc_auc" in row
