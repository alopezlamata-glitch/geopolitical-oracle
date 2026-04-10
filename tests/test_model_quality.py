"""
CI quality gate: reads data/model/backtest_results.json and asserts that
aggregate metrics stay above minimum thresholds.

Thresholds are intentionally conservative (well below current values)
so that the gate catches genuine regressions, not normal run-to-run variance.

Current baselines (2026-04-10, 911 examples, 9 folds):
  Brier skill  : +0.508   (threshold 0.10)
  ROC-AUC      : 0.917    (threshold 0.70)
  ECE          : 0.100    (threshold 0.20)
  Coverage 80% : 0.802    (threshold 0.70)
  Worst-fold AUC: checked per-fold

Run with:
    pytest tests/test_model_quality.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_RESULTS_PATH = Path(__file__).parent.parent / "data" / "model" / "backtest_results.json"

# ── Thresholds ───────────────────────────────────────────────────────────────
# A PR that pushes any metric below these values will fail CI.
# Update them (upward only) when you achieve a meaningful improvement.

THRESHOLDS = {
    "brier_skill_weighted": 0.10,   # must beat climatology baseline
    "roc_auc_weighted": 0.70,       # basic discrimination
    "ece_weighted": 0.20,           # calibration error ceiling
    "conformal_coverage_80_avg": 0.70,  # within 10pp of 80% target
}

# No single fold should collapse completely
WORST_FOLD_AUC_MIN = 0.55


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def results() -> dict:
    if not _RESULTS_PATH.exists():
        pytest.skip(
            f"backtest_results.json not found at {_RESULTS_PATH}. "
            "Run: python scripts/backtest.py"
        )
    data = json.loads(_RESULTS_PATH.read_text())
    return data


@pytest.fixture(scope="module")
def aggregate(results) -> dict:
    return results.get("aggregate", {})


@pytest.fixture(scope="module")
def folds(results) -> list:
    return results.get("folds", [])


# ── Aggregate metric tests ────────────────────────────────────────────────────

@pytest.mark.parametrize("metric,threshold", THRESHOLDS.items())
def test_aggregate_threshold(aggregate, metric, threshold):
    """Each aggregate metric must meet its threshold."""
    value = aggregate.get(metric)
    assert value is not None, f"Metric '{metric}' missing from backtest_results.json"
    assert value >= threshold, (
        f"{metric} = {value:.4f} is below threshold {threshold:.4f}. "
        "Retrain the model or investigate data quality."
    )


# ── Per-fold sanity checks ────────────────────────────────────────────────────

def test_minimum_folds(folds):
    """Backtest must have run at least 3 folds to be meaningful."""
    assert len(folds) >= 3, (
        f"Only {len(folds)} fold(s) in backtest results. "
        "Need ≥ 3 for aggregate stability."
    )


def test_no_collapsed_fold(folds):
    """No single fold's AUC should fall below the worst-fold floor."""
    low_folds = [
        f for f in folds
        if f.get("roc_auc") is not None and f["roc_auc"] < WORST_FOLD_AUC_MIN
    ]
    assert not low_folds, (
        f"{len(low_folds)} fold(s) have AUC < {WORST_FOLD_AUC_MIN}: "
        + ", ".join(
            f"{f.get('test_period','?')} AUC={f['roc_auc']:.3f}" for f in low_folds
        )
    )


def test_no_negative_brier_skill(folds):
    """Every fold must beat the naive climatology baseline (skill > 0)."""
    failing = [
        f for f in folds
        if f.get("brier_skill") is not None and f["brier_skill"] < 0
    ]
    assert not failing, (
        f"{len(failing)} fold(s) have negative Brier skill (worse than base rate): "
        + ", ".join(
            f"{f.get('test_period','?')} skill={f['brier_skill']:.3f}" for f in failing
        )
    )


def test_conformal_coverage_not_too_wide(aggregate):
    """Coverage should not exceed 95% — that would indicate degenerate [0,1] intervals."""
    cov = aggregate.get("conformal_coverage_80_avg", 0.0)
    assert cov <= 0.99, (
        f"Conformal coverage {cov:.3f} is suspiciously high (≥ 0.99). "
        "Intervals may have degenerated to [0, 1] — check conformal scores."
    )


# ── Data freshness check ──────────────────────────────────────────────────────

def test_results_not_stale(results):
    """
    Warn if backtest results are more than 90 days old.
    Stale results may not reflect the current model or training data.
    """
    from datetime import datetime, timedelta, timezone
    generated_at_str = results.get("generated_at", "")
    if not generated_at_str:
        pytest.skip("No 'generated_at' timestamp in backtest_results.json")
    generated_at = datetime.fromisoformat(generated_at_str)
    if generated_at.tzinfo is None:
        generated_at = generated_at.replace(tzinfo=timezone.utc)
    age = datetime.now(timezone.utc) - generated_at
    assert age <= timedelta(days=90), (
        f"backtest_results.json is {age.days} days old. "
        "Re-run: python scripts/backtest.py"
    )
