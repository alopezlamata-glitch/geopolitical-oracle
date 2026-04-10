"""
CI quality gate: reads data/model/backtest_results.json and asserts that
aggregate metrics stay above minimum thresholds.

Thresholds are set to be genuinely demanding — a PR that regresses calibration
or temporal generalisation meaningfully will fail.

Current baselines (2026-04-10, 911 examples, 9 folds):
  Brier skill  : +0.508   (threshold > 0.20)
  ROC-AUC      : 0.917    (threshold > 0.75)
  ECE          : 0.100    (threshold < 0.12)
  Coverage 80% : 80.2%    (threshold |cov - 0.80| < 0.05)
  Cross-era fold brier_skill: all > 0

Run with:
    pytest tests/test_model_quality.py -v
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

_RESULTS_PATH = Path(__file__).parent.parent / "data" / "model" / "backtest_results.json"

# ── Aggregate thresholds ──────────────────────────────────────────────────────
# Update upward only when a meaningful improvement is confirmed.

THRESHOLDS = {
    "brier_skill_weighted": 0.20,   # clearly beats climatology
    "roc_auc_weighted": 0.75,       # solid discrimination
    "ece_weighted": 0.12,           # calibration error ceiling (note: lower = better; assert value < threshold)
}

# Conformal coverage must stay within ±5pp of the 80% target
CONFORMAL_TARGET = 0.80
CONFORMAL_TOL = 0.05

# Per-fold floors
WORST_FOLD_AUC_MIN = 0.55
CROSS_ERA_BRIER_SKILL_MIN = 0.0   # every cross-era fold must beat base rate


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def results() -> dict:
    if not _RESULTS_PATH.exists():
        pytest.skip(
            f"backtest_results.json not found at {_RESULTS_PATH}. "
            "Run: python scripts/backtest.py"
        )
    return json.loads(_RESULTS_PATH.read_text())


@pytest.fixture(scope="module")
def aggregate(results) -> dict:
    return results.get("aggregate", {})


@pytest.fixture(scope="module")
def within_folds(results) -> list:
    return results.get("within_era", [])


@pytest.fixture(scope="module")
def cross_folds(results) -> list:
    return results.get("cross_era", [])


@pytest.fixture(scope="module")
def all_folds(within_folds, cross_folds) -> list:
    return within_folds + cross_folds


# ── Aggregate metric tests ────────────────────────────────────────────────────

def test_brier_skill(aggregate):
    """Model must clearly beat climatology baseline."""
    v = aggregate.get("brier_skill_weighted")
    assert v is not None, "brier_skill_weighted missing from backtest_results.json"
    assert v > THRESHOLDS["brier_skill_weighted"], (
        f"Brier skill {v:.4f} <= {THRESHOLDS['brier_skill_weighted']:.2f}. "
        "Model is too close to the base-rate predictor."
    )


def test_roc_auc(aggregate):
    """Model must maintain solid discrimination."""
    v = aggregate.get("roc_auc_weighted")
    assert v is not None, "roc_auc_weighted missing from backtest_results.json"
    assert not math.isnan(v), "roc_auc_weighted is NaN"
    assert v > THRESHOLDS["roc_auc_weighted"], (
        f"ROC-AUC {v:.4f} <= {THRESHOLDS['roc_auc_weighted']:.2f}."
    )


def test_ece(aggregate):
    """Calibration error must stay below ceiling."""
    v = aggregate.get("ece_weighted")
    assert v is not None, "ece_weighted missing from backtest_results.json"
    assert v < THRESHOLDS["ece_weighted"], (
        f"ECE {v:.4f} >= {THRESHOLDS['ece_weighted']:.2f}. "
        "Calibration has regressed — check training data distribution or calibrator."
    )


def test_conformal_coverage_on_target(aggregate):
    """80% conformal CI must achieve coverage within ±5pp of 80%."""
    cov = aggregate.get("conformal_coverage_80_avg")
    assert cov is not None, "conformal_coverage_80_avg missing"
    deviation = abs(cov - CONFORMAL_TARGET)
    assert deviation <= CONFORMAL_TOL, (
        f"Conformal coverage {cov:.1%} deviates {deviation:.1%} from "
        f"{CONFORMAL_TARGET:.0%} target (tolerance ±{CONFORMAL_TOL:.0%}). "
        "Distribution shift detected — retrain or recalibrate."
    )


def test_conformal_coverage_not_degenerate(aggregate):
    """Coverage must not be trivially 100% (degenerate [0,1] intervals)."""
    cov = aggregate.get("conformal_coverage_80_avg", 0.0)
    assert cov <= 0.99, (
        f"Conformal coverage {cov:.1%} is pathologically high. "
        "Conformal scores may have degenerated — check conformal_scores.json."
    )


# ── Range-ECE tests (require backtest to emit ece_by_range) ──────────────────

def test_ece_low_range(aggregate):
    """
    ECE in the 0–0.2 range (rare events) must not exceed 0.15.
    This is the known weak spot — under-prediction at low probabilities.
    Failing here indicates the low-frequency tail is getting worse.
    """
    ece_ranges = aggregate.get("ece_by_range", {})
    if "low" not in ece_ranges:
        pytest.skip("ece_by_range.low not in results — re-run scripts/backtest.py")
    v = ece_ranges["low"]
    assert v < 0.15, (
        f"ECE in [0,0.2] range = {v:.4f} >= 0.15. "
        "Low-probability calibration has regressed."
    )


def test_ece_mid_range(aggregate):
    """ECE in the 0.2–0.8 range (contested predictions) must stay below 0.12."""
    ece_ranges = aggregate.get("ece_by_range", {})
    if "mid" not in ece_ranges:
        pytest.skip("ece_by_range.mid not in results — re-run scripts/backtest.py")
    v = ece_ranges["mid"]
    assert v < 0.12, (
        f"ECE in [0.2,0.8] range = {v:.4f} >= 0.12."
    )


# ── Per-fold sanity checks ────────────────────────────────────────────────────

def test_minimum_folds(all_folds):
    """Backtest must have produced at least 3 folds total."""
    assert len(all_folds) >= 3, (
        f"Only {len(all_folds)} fold(s). Need >= 3 for aggregate stability."
    )


def test_no_collapsed_fold(all_folds):
    """No single fold's AUC should fall below the worst-fold floor."""
    low = [
        f for f in all_folds
        if f.get("roc_auc") is not None
        and not math.isnan(f["roc_auc"])
        and f["roc_auc"] < WORST_FOLD_AUC_MIN
    ]
    assert not low, (
        f"{len(low)} fold(s) with AUC < {WORST_FOLD_AUC_MIN}: "
        + ", ".join(f"{f.get('fold','?')} AUC={f['roc_auc']:.3f}" for f in low)
    )


def test_no_negative_brier_skill_within_era(within_folds):
    """
    Every within-era fold with n_test >= 25 must beat the naive climatology baseline.

    Folds smaller than 25 examples are excluded: with n < 25, a single cluster
    of wrong predictions can produce Brier skill < 0 by chance (SE ≈ 0.2+),
    making the metric statistically unreliable. Boundary folds at the edge of
    the training era (e.g. 2013-12..2014-03) commonly fall below this threshold.
    """
    _MIN_N = 25
    eligible = [f for f in within_folds if f.get("n_test", 0) >= _MIN_N]
    failing = [
        f for f in eligible
        if f.get("brier_skill") is not None and f["brier_skill"] < 0
    ]
    assert not failing, (
        f"{len(failing)} within-era fold(s) (n>={_MIN_N}) have negative Brier skill: "
        + ", ".join(f"{f.get('fold','?')} n={f.get('n_test')} skill={f['brier_skill']:.3f}"
                    for f in failing)
    )


def test_no_negative_brier_skill_cross_era(cross_folds):
    """
    Every cross-era fold must beat the naive climatology baseline.

    Cross-era folds reveal temporal generalisation across the 2014-2020 gap.
    A negative skill score here means the model learned patterns that are
    era-specific and do not transfer — the most serious failure mode.
    """
    if not cross_folds:
        pytest.skip("No cross-era folds in results.")
    failing = [
        f for f in cross_folds
        if f.get("brier_skill") is not None and f["brier_skill"] < CROSS_ERA_BRIER_SKILL_MIN
    ]
    assert not failing, (
        f"{len(failing)} cross-era fold(s) have Brier skill < {CROSS_ERA_BRIER_SKILL_MIN}: "
        + ", ".join(
            f"{f.get('fold','?')} skill={f['brier_skill']:.3f}" for f in failing
        )
        + "\nThe model may be overfitting to the training era. "
        "Consider adding data from the 2014-2020 gap."
    )


def test_cross_era_calibrator_logged(cross_folds):
    """
    Cross-era folds should log which calibrator was selected.
    Absence of this field indicates the backtest is outdated.
    """
    if not cross_folds:
        pytest.skip("No cross-era folds in results.")
    missing = [
        f for f in cross_folds if "calibrator_method" not in f
    ]
    if missing:
        pytest.skip(
            f"{len(missing)} cross-era fold(s) lack 'calibrator_method'. "
            "Re-run scripts/backtest.py to get updated results."
        )


# ── Data freshness check ──────────────────────────────────────────────────────

def test_results_not_stale(results):
    """Results more than 90 days old may not reflect the current model."""
    from datetime import datetime, timedelta, timezone
    ts = results.get("generated_at", "")
    if not ts:
        pytest.skip("No 'generated_at' in backtest_results.json")
    generated_at = datetime.fromisoformat(ts)
    if generated_at.tzinfo is None:
        generated_at = generated_at.replace(tzinfo=timezone.utc)
    age = datetime.now(timezone.utc) - generated_at
    assert age <= timedelta(days=90), (
        f"backtest_results.json is {age.days} days old. "
        "Re-run: python scripts/backtest.py"
    )
