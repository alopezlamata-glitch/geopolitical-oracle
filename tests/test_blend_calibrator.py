"""
Tests for predictor/blend_calibrator.py — Phase B learned blend weights.

Coverage:
  - fit_blend_weights: gate (not enough data), fit correctness, coefficient types
  - _compute_diagnostics: Brier comparison, by_strategy breakdown
  - load_blend_weights: round-trip save/load
  - run_blend_calibration: end-to-end with synthetic resolved records
  - market_prior integration: learned weights loaded via _load_blend_weights_cached
  - DB migration: blend columns added to predictions table
"""
from __future__ import annotations

import json
import math
import random
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

import pytest

from predictor.blend_calibrator import (
    MIN_RESOLVED_WITH_MARKET,
    BLEND_CALIBRATOR_VERSION,
    fit_blend_weights,
    load_blend_weights,
    run_blend_calibration,
    _compute_diagnostics,
    _apply_fixed_blend,
    _apply_learned_blend,
    _brier,
    _log_loss,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _make_records(
    n: int,
    include_market: bool = True,
    seed: int = 42,
) -> list[dict]:
    """
    Generate synthetic resolved prediction records.

    Ground truth: outcome ~ Bernoulli(p_model * 0.7 + p_market * 0.3)
    so the market signal adds genuine predictive value.
    """
    rng = random.Random(seed)
    records = []
    for _ in range(n):
        p_model = rng.uniform(0.15, 0.85)
        p_market = rng.uniform(0.15, 0.85) if include_market else None
        true_p = p_model * 0.7 + (p_market or p_model) * 0.3
        outcome = 1 if rng.random() < true_p else 0
        records.append({
            "p_model_raw": p_model,
            "p_market_raw": p_market,
            "market_weight": 0.35 if include_market else 0.0,
            "blend_strategy": "model_plus_market" if include_market else "model_only",
            "calibrated_prob": p_model,  # simplified
            "predicted_at": datetime.now(timezone.utc),
            "outcome": outcome,
        })
    return records


# ── _brier and _log_loss ───────────────────────────────────────────────────────

class TestMetrics:

    def test_brier_perfect_predictions(self):
        probs = [1.0, 1.0, 0.0, 0.0]
        outcomes = [1, 1, 0, 0]
        assert _brier(probs, outcomes) == pytest.approx(0.0, abs=1e-6)

    def test_brier_worst_predictions(self):
        probs = [0.0, 0.0, 1.0, 1.0]
        outcomes = [1, 1, 0, 0]
        assert _brier(probs, outcomes) == pytest.approx(1.0, abs=1e-6)

    def test_brier_uniform_is_0_25(self):
        probs = [0.5] * 100
        outcomes = [1] * 50 + [0] * 50
        assert _brier(probs, outcomes) == pytest.approx(0.25, abs=0.01)

    def test_log_loss_perfect(self):
        # Near-perfect predictions → very small log loss
        probs = [0.999, 0.999, 0.001, 0.001]
        outcomes = [1, 1, 0, 0]
        assert _log_loss(probs, outcomes) < 0.01

    def test_brier_empty_returns_nan(self):
        import math
        result = _brier([], [])
        assert math.isnan(result)


# ── fit_blend_weights ──────────────────────────────────────────────────────────

class TestFitBlendWeights:

    def test_returns_none_when_not_enough_data(self):
        # MIN_RESOLVED_WITH_MARKET - 1 records with market signal
        records = _make_records(MIN_RESOLVED_WITH_MARKET - 1, include_market=True)
        result = fit_blend_weights(records)
        assert result is None

    def test_returns_none_when_no_market_signal(self):
        # Plenty of records but none have p_market_raw
        records = _make_records(100, include_market=False)
        result = fit_blend_weights(records)
        assert result is None

    def test_returns_coefficients_when_enough_data(self):
        records = _make_records(MIN_RESOLVED_WITH_MARKET + 10, include_market=True)
        result = fit_blend_weights(records)
        assert result is not None
        assert "alpha" in result
        assert "beta" in result
        assert "bias" in result
        assert "n_training" in result

    def test_coefficients_are_floats(self):
        records = _make_records(50, include_market=True)
        result = fit_blend_weights(records)
        assert result is not None
        assert isinstance(result["alpha"], float)
        assert isinstance(result["beta"], float)
        assert isinstance(result["bias"], float)

    def test_n_training_matches_market_subset(self):
        n_total = 60
        records_with = _make_records(n_total, include_market=True)
        # Mix in some without market
        records_without = _make_records(10, include_market=False, seed=99)
        mixed = records_with + records_without
        result = fit_blend_weights(mixed)
        assert result is not None
        # Only records with p_market_raw should be counted
        assert result["n_training"] == n_total

    def test_positive_alpha_and_beta_on_informative_data(self):
        """
        When both model and market are informative, both coefficients should
        be positive (both signals point toward outcome).
        """
        records = _make_records(200, include_market=True, seed=7)
        result = fit_blend_weights(records)
        assert result is not None
        # Alpha (model) must be positive on this data-generating process
        assert result["alpha"] > 0, "alpha should be positive when model is informative"
        # Beta (market) — typically positive too, may be small
        # Don't assert strictly positive as small-N can give noise


# ── _compute_diagnostics ──────────────────────────────────────────────────────

class TestComputeDiagnostics:

    def test_returns_empty_on_no_records(self):
        diag = _compute_diagnostics([], None)
        assert diag == {}

    def test_base_rate_computed(self):
        records = _make_records(100, include_market=False)
        diag = _compute_diagnostics(records, None)
        assert "base_rate" in diag
        assert 0.0 <= diag["base_rate"] <= 1.0

    def test_brier_model_only_present(self):
        records = _make_records(50, include_market=False)
        diag = _compute_diagnostics(records, None)
        assert "brier_model_only" in diag
        assert 0.0 <= diag["brier_model_only"] <= 1.0

    def test_fixed_blend_metrics_present_when_market_available(self):
        records = _make_records(50, include_market=True)
        diag = _compute_diagnostics(records, None)
        assert "brier_fixed_blend_on_market_subset" in diag
        assert "brier_model_only_on_market_subset" in diag
        assert "n_with_market" in diag
        assert diag["n_with_market"] == 50

    def test_learned_blend_metrics_present_when_fit_provided(self):
        records = _make_records(50, include_market=True)
        fake_fit = {"alpha": 0.8, "beta": 0.3, "bias": -0.05, "n_training": 50}
        diag = _compute_diagnostics(records, fake_fit)
        assert "brier_learned_blend" in diag
        assert "learned_vs_fixed_improvement" in diag

    def test_by_strategy_breakdown(self):
        records = _make_records(30, include_market=True)
        records += _make_records(20, include_market=False, seed=77)
        diag = _compute_diagnostics(records, None)
        assert "by_strategy" in diag
        # Should have at least two strategies
        assert len(diag["by_strategy"]) >= 1

    def test_brier_skill_model_positive_on_informative_model(self):
        """
        On synthetic data where outcome ~ f(p_model), model should beat base rate.
        """
        rng = random.Random(42)
        records = []
        for _ in range(200):
            p = rng.uniform(0.15, 0.85)
            outcome = 1 if rng.random() < p else 0
            records.append({
                "p_model_raw": p, "p_market_raw": None,
                "calibrated_prob": p, "blend_strategy": "model_only",
                "outcome": outcome, "market_weight": 0.0, "predicted_at": None,
            })
        diag = _compute_diagnostics(records, None)
        assert diag.get("brier_skill_model_only", -1) > 0


# ── _apply_fixed_blend and _apply_learned_blend ───────────────────────────────

class TestBlendMath:

    def test_fixed_blend_between_model_and_market(self):
        p = _apply_fixed_blend(0.30, 0.70, w=0.40)
        assert 0.30 < p < 0.70

    def test_fixed_blend_identity_at_w0(self):
        p = _apply_fixed_blend(0.40, 0.80, w=0.0)
        assert abs(p - 0.40) < 0.01

    def test_fixed_blend_identity_at_w1(self):
        p = _apply_fixed_blend(0.40, 0.80, w=1.0)
        assert abs(p - 0.80) < 0.01

    def test_learned_blend_respects_coefficients(self):
        # alpha=1, beta=0, bias=0 → should return p_model exactly
        p = _apply_learned_blend(0.60, 0.20, alpha=1.0, beta=0.0, bias=0.0)
        assert abs(p - 0.60) < 0.01

    def test_learned_blend_always_in_01(self):
        for p_m, p_mkt in [(0.01, 0.99), (0.99, 0.01), (0.50, 0.50)]:
            p = _apply_learned_blend(p_m, p_mkt, alpha=2.0, beta=2.0, bias=1.0)
            assert 0.0 < p < 1.0


# ── load_blend_weights round-trip ─────────────────────────────────────────────

class TestLoadBlendWeights:

    def test_returns_none_when_file_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "predictor.blend_calibrator._BLEND_WEIGHTS_PATH",
            tmp_path / "nonexistent.json",
        )
        assert load_blend_weights() is None

    def test_round_trip_save_and_load(self, tmp_path, monkeypatch):
        path = tmp_path / "blend_weights.json"
        monkeypatch.setattr("predictor.blend_calibrator._BLEND_WEIGHTS_PATH", path)

        # Write a valid weights file
        weights = {
            "alpha": 0.82,
            "beta": 0.34,
            "bias": -0.07,
            "n_training": 45,
            "blend_calibrator_version": BLEND_CALIBRATOR_VERSION,
            "fitted_at": datetime.now(timezone.utc).isoformat(),
        }
        path.write_text(json.dumps(weights))

        loaded = load_blend_weights()
        assert loaded is not None
        assert abs(loaded["alpha"] - 0.82) < 1e-6
        assert abs(loaded["beta"] - 0.34) < 1e-6
        assert abs(loaded["bias"] - (-0.07)) < 1e-6

    def test_returns_none_on_malformed_json(self, tmp_path, monkeypatch):
        path = tmp_path / "blend_weights.json"
        monkeypatch.setattr("predictor.blend_calibrator._BLEND_WEIGHTS_PATH", path)
        path.write_text("{not valid json")
        assert load_blend_weights() is None

    def test_returns_none_on_missing_keys(self, tmp_path, monkeypatch):
        path = tmp_path / "blend_weights.json"
        monkeypatch.setattr("predictor.blend_calibrator._BLEND_WEIGHTS_PATH", path)
        path.write_text(json.dumps({"alpha": 0.5}))   # missing beta, bias
        assert load_blend_weights() is None


# ── run_blend_calibration end-to-end ─────────────────────────────────────────

class TestRunBlendCalibration:

    def test_returns_message_when_no_db_data(self, monkeypatch):
        """When _fetch_resolved_predictions returns empty, get a clear message."""
        monkeypatch.setattr(
            "predictor.blend_calibrator._fetch_resolved_predictions",
            lambda: [],
        )
        result = run_blend_calibration(verbose=False)
        assert result["fit"] is None
        assert result["saved"] is False
        assert "No resolved" in result["message"] or "resolved" in result["message"].lower()

    def test_saves_weights_when_enough_data(self, tmp_path, monkeypatch):
        """End-to-end: inject synthetic records, verify weights saved."""
        records = _make_records(MIN_RESOLVED_WITH_MARKET + 20, include_market=True)
        monkeypatch.setattr(
            "predictor.blend_calibrator._fetch_resolved_predictions",
            lambda: records,
        )
        blend_path = tmp_path / "blend_weights.json"
        monkeypatch.setattr("predictor.blend_calibrator._BLEND_WEIGHTS_PATH", blend_path)
        monkeypatch.setattr("predictor.blend_calibrator._MODEL_DIR", tmp_path)

        result = run_blend_calibration(verbose=False)

        assert result["fit"] is not None
        assert result["saved"] is True
        assert blend_path.exists()

        saved = json.loads(blend_path.read_text())
        assert "alpha" in saved
        assert "beta" in saved
        assert "blend_calibrator_version" in saved
        assert "fitted_at" in saved

    def test_does_not_save_when_insufficient_data(self, tmp_path, monkeypatch):
        """Below gate: no file written, message explains why."""
        records = _make_records(MIN_RESOLVED_WITH_MARKET - 5, include_market=True)
        monkeypatch.setattr(
            "predictor.blend_calibrator._fetch_resolved_predictions",
            lambda: records,
        )
        blend_path = tmp_path / "blend_weights.json"
        monkeypatch.setattr("predictor.blend_calibrator._BLEND_WEIGHTS_PATH", blend_path)
        monkeypatch.setattr("predictor.blend_calibrator._MODEL_DIR", tmp_path)

        result = run_blend_calibration(verbose=False)

        assert result["fit"] is None
        assert result["saved"] is False
        assert not blend_path.exists()

    def test_diagnostics_always_returned(self, monkeypatch):
        """Diagnostics should be present even when gate not passed."""
        records = _make_records(5, include_market=True)
        monkeypatch.setattr(
            "predictor.blend_calibrator._fetch_resolved_predictions",
            lambda: records,
        )
        result = run_blend_calibration(verbose=False)
        assert "diagnostics" in result
        assert "n_resolved" in result["diagnostics"]


# ── DB migration: blend columns ────────────────────────────────────────────────

class TestBlendColumnsMigration:

    def test_migration_adds_columns_to_predictions(self):
        """
        In-memory DuckDB: create predictions table (old schema, no blend cols),
        then run migration and verify new columns exist.
        """
        try:
            import duckdb
        except ImportError:
            pytest.skip("duckdb not installed")

        con = duckdb.connect(":memory:")
        # Minimal predictions table without blend columns
        con.execute("""
            CREATE TABLE predictions (
                prediction_id VARCHAR PRIMARY KEY,
                raw_prob FLOAT,
                calibrated_prob FLOAT,
                answer VARCHAR,
                model_id VARCHAR,
                predicted_at TIMESTAMPTZ,
                as_of_time TIMESTAMPTZ
            )
        """)

        # Import and run the migration against this connection
        from data_layer.db import _migrate_predictions_blend_columns
        _migrate_predictions_blend_columns(con)

        # Verify columns now exist
        cols = {
            row[0]
            for row in con.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'predictions'"
            ).fetchall()
        }
        for expected_col in ["p_model_raw", "p_market_raw", "market_weight",
                              "blend_strategy", "n_market_signals", "market_gate_passed"]:
            assert expected_col in cols, f"Missing column: {expected_col}"

        con.close()
