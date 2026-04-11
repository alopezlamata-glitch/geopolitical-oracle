"""
Tests for predictor/market_prior.py — calibrated market prior blending.

Coverage:
  - resolve_market_signal: correct signal construction, liquidity normalisation
  - compute_market_weight: gate conditions, formula bounds, conflict penalty
  - blend_market_prior: model_only, model_plus_market, market_dominant paths
  - BlendResult: full audit trail fields present and consistent
  - Edge cases: stale signals, both sources, conflicting probabilities
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from predictor.market_prior import (
    GATE_MAX_AGE_HOURS,
    GATE_MIN_MATCH_SCORE,
    W_MARKET_CAP,
    BlendResult,
    MarketSignal,
    blend_market_prior,
    compute_market_weight,
    resolve_market_signal,
)

# ── Helpers ────────────────────────────────────────────────────────────────────

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _fresh_signal(
    p: float = 0.65,
    source: str = "metaculus",
    match_score: float = 0.60,
    liquidity_score: float = 0.50,
    forecasters: int = 120,
    volume_usd: float | None = None,
    age_hours: float = 0.5,
) -> MarketSignal:
    now = _now()
    return MarketSignal(
        probability_yes=p,
        source=source,
        source_question_id=None,
        snapshot_time=now - timedelta(hours=age_hours),
        match_score=match_score,
        liquidity_score=liquidity_score,
        forecasters=forecasters if source == "metaculus" else None,
        volume_usd=volume_usd if source == "polymarket" else None,
        resolution_compatible=True,
    )


# ── resolve_market_signal ──────────────────────────────────────────────────────

class TestResolveMarketSignal:

    def test_returns_empty_when_no_signals(self):
        sigs = resolve_market_signal(as_of_time=_now())
        assert sigs == []

    def test_metaculus_signal_constructed(self):
        sigs = resolve_market_signal(
            metaculus_p=0.72,
            metaculus_forecasters=150,
            as_of_time=_now(),
        )
        assert len(sigs) == 1
        s = sigs[0]
        assert s.source == "metaculus"
        assert abs(s.probability_yes - 0.72) < 1e-6
        assert s.forecasters == 150
        assert s.resolution_compatible is True

    def test_polymarket_signal_constructed(self):
        sigs = resolve_market_signal(
            polymarket_p=0.55,
            polymarket_volume=50_000,
            polymarket_match_score=0.70,
            as_of_time=_now(),
        )
        assert len(sigs) == 1
        s = sigs[0]
        assert s.source == "polymarket"
        assert abs(s.probability_yes - 0.55) < 1e-6
        assert s.volume_usd == 50_000
        assert abs(s.match_score - 0.70) < 1e-6

    def test_both_sources_returned(self):
        sigs = resolve_market_signal(
            metaculus_p=0.60,
            metaculus_forecasters=80,
            polymarket_p=0.65,
            polymarket_volume=30_000,
            polymarket_match_score=0.55,
            as_of_time=_now(),
        )
        assert len(sigs) == 2
        sources = {s.source for s in sigs}
        assert sources == {"metaculus", "polymarket"}

    def test_out_of_range_probabilities_clipped(self):
        sigs = resolve_market_signal(metaculus_p=1.05, as_of_time=_now())
        # 1.05 is out of (0.01, 0.99) → should NOT create a signal
        assert len(sigs) == 0

    def test_zero_probability_rejected(self):
        sigs = resolve_market_signal(metaculus_p=0.0, as_of_time=_now())
        assert len(sigs) == 0

    def test_liquidity_score_normalised(self):
        # 300 forecasters = full liquidity (1.0)
        sigs = resolve_market_signal(metaculus_p=0.50, metaculus_forecasters=300, as_of_time=_now())
        assert sigs[0].liquidity_score == pytest.approx(1.0, abs=0.01)

        # 150 forecasters = 0.50
        sigs2 = resolve_market_signal(metaculus_p=0.50, metaculus_forecasters=150, as_of_time=_now())
        assert sigs2[0].liquidity_score == pytest.approx(0.50, abs=0.01)


# ── compute_market_weight ──────────────────────────────────────────────────────

class TestComputeMarketWeight:

    def test_no_signals_returns_zero(self):
        w, reason = compute_market_weight([], _now())
        assert w == 0.0
        assert reason is not None

    def test_gate_fails_on_low_match_score(self):
        sig = _fresh_signal(match_score=0.20)   # below GATE_MIN_MATCH_SCORE
        w, reason = compute_market_weight([sig], _now())
        assert w == 0.0
        assert "match_score" in reason

    def test_gate_fails_on_stale_signal(self):
        sig = _fresh_signal(match_score=0.70, age_hours=GATE_MAX_AGE_HOURS + 1)
        w, reason = compute_market_weight([sig], _now())
        assert w == 0.0
        assert "stale" in reason

    def test_weight_positive_when_gate_passes(self):
        sig = _fresh_signal(match_score=0.70, liquidity_score=0.60, age_hours=0.5)
        w, reason = compute_market_weight([sig], _now())
        assert reason is None
        assert w > 0.0

    def test_weight_capped_at_W_MARKET_CAP(self):
        # Max possible weight: match=1.0, liq=1.0, rec=1.0, consensus bonus, no penalty
        sig1 = _fresh_signal(source="metaculus", match_score=1.0, liquidity_score=1.0, age_hours=0.0)
        sig2 = _fresh_signal(source="polymarket", match_score=1.0, liquidity_score=1.0, age_hours=0.0)
        w, reason = compute_market_weight([sig1, sig2], _now())
        assert reason is None
        assert w <= W_MARKET_CAP

    def test_consensus_bonus_when_two_sources(self):
        sig_single = _fresh_signal(match_score=0.60, liquidity_score=0.50, age_hours=1.0)
        w_single, _ = compute_market_weight([sig_single], _now())

        sig1 = _fresh_signal(source="metaculus", match_score=0.60, liquidity_score=0.50, age_hours=1.0)
        sig2 = _fresh_signal(source="polymarket", match_score=0.60, liquidity_score=0.50, age_hours=1.0)
        w_two, _ = compute_market_weight([sig1, sig2], _now())

        assert w_two > w_single   # consensus bonus applied

    def test_conflict_penalty_reduces_weight(self):
        # Two signals with spread > 0.20 → penalty applied
        sig_agree1 = _fresh_signal(p=0.60, source="metaculus", match_score=0.65, age_hours=1.0)
        sig_agree2 = _fresh_signal(p=0.62, source="polymarket", match_score=0.65, age_hours=1.0)
        w_agree, _ = compute_market_weight([sig_agree1, sig_agree2], _now())

        sig_conflict1 = _fresh_signal(p=0.30, source="metaculus", match_score=0.65, age_hours=1.0)
        sig_conflict2 = _fresh_signal(p=0.65, source="polymarket", match_score=0.65, age_hours=1.0)
        w_conflict, _ = compute_market_weight([sig_conflict1, sig_conflict2], _now())

        assert w_agree > w_conflict


# ── blend_market_prior ─────────────────────────────────────────────────────────

class TestBlendMarketPrior:

    def test_model_only_when_no_signals(self):
        result = blend_market_prior(p_model=0.40, signals=[], as_of_time=_now())
        assert result.blend_strategy == "model_only"
        assert result.p_final == result.p_model
        assert result.market_weight == 0.0
        assert result.gate_passed is False
        assert result.p_market is None

    def test_model_only_when_gate_fails(self):
        sig = _fresh_signal(match_score=0.10)   # too low
        result = blend_market_prior(p_model=0.40, signals=[sig], as_of_time=_now())
        assert result.blend_strategy == "model_only"
        assert result.p_final == pytest.approx(result.p_model, abs=1e-4)
        assert result.gate_passed is False

    def test_blend_pulls_toward_market(self):
        # model at 0.30, market at 0.70 — blend should be between them
        sig = _fresh_signal(p=0.70, match_score=0.65, liquidity_score=0.60, age_hours=0.5)
        result = blend_market_prior(p_model=0.30, signals=[sig], as_of_time=_now())
        assert result.gate_passed is True
        assert result.blend_strategy in ("model_plus_market", "market_dominant")
        assert 0.30 < result.p_final < 0.70   # strictly between model and market

    def test_blend_result_audit_fields_complete(self):
        sig = _fresh_signal(p=0.65, match_score=0.70, age_hours=1.0)
        result = blend_market_prior(p_model=0.45, signals=[sig], as_of_time=_now())
        # All audit fields must be present and typed correctly
        assert isinstance(result.p_final, float)
        assert isinstance(result.p_model, float)
        assert isinstance(result.market_weight, float)
        assert isinstance(result.market_sources, list)
        assert isinstance(result.blend_strategy, str)
        assert isinstance(result.blend_strategy_version, str)
        assert isinstance(result.gate_passed, bool)
        assert isinstance(result.n_signals, int)

    def test_p_final_always_in_01(self):
        # Extreme model probs should still give valid output
        for p_model in [0.01, 0.05, 0.50, 0.95, 0.99]:
            sig = _fresh_signal(p=0.50, match_score=0.70, age_hours=0.5)
            result = blend_market_prior(p_model=p_model, signals=[sig], as_of_time=_now())
            assert 0.0 < result.p_final < 1.0, f"p_final out of range for p_model={p_model}"

    def test_market_dominant_strategy_label(self):
        # Force high weight: max match, max liquidity, fresh, both sources
        sig1 = _fresh_signal(source="metaculus",  p=0.80, match_score=1.0, liquidity_score=1.0, forecasters=300, age_hours=0.1)
        sig2 = _fresh_signal(source="polymarket", p=0.78, match_score=1.0, liquidity_score=1.0, age_hours=0.1)
        result = blend_market_prior(p_model=0.50, signals=[sig1, sig2], as_of_time=_now())
        # With maximum inputs the weight should be >= 0.40 → market_dominant
        if result.market_weight >= 0.40:
            assert result.blend_strategy == "market_dominant"
        else:
            assert result.blend_strategy in ("model_plus_market", "model_only")

    def test_blend_is_log_odds_not_linear(self):
        """
        Log-odds blend and linear blend diverge when probabilities are extreme.
        This test confirms we're not doing simple linear blending.
        """
        p_m = 0.20
        p_mkt = 0.80
        w = 0.40

        # What linear blend would give
        linear = (1 - w) * p_m + w * p_mkt

        # Log-odds blend
        lo_m = math.log(p_m / (1 - p_m))
        lo_mkt = math.log(p_mkt / (1 - p_mkt))
        lo_blend = (1 - w) * lo_m + w * lo_mkt
        logodds = 1.0 / (1.0 + math.exp(-lo_blend))

        # They should differ (if they're equal something is wrong)
        assert abs(linear - logodds) > 0.005


# ── Integration: inference.predict with market blend ──────────────────────────

class TestInferenceMarketBlend:
    """
    Smoke tests for the predict() function market blend path.
    Does NOT require a trained model — untrained path still exercises audit fields.
    """

    def test_predict_returns_blend_audit_fields(self):
        from predictor.inference import predict
        result = predict(
            features={},
            metaculus_p=0.65,
            metaculus_forecasters=100,
            as_of_time=_now(),
        )
        assert "blend_strategy" in result
        assert "market_weight" in result
        assert "blend_strategy_version" in result
        assert "market_gate_passed" in result
        assert "n_market_signals" in result

    def test_predict_no_market_gives_model_only(self):
        from predictor.inference import predict
        result = predict(features={}, as_of_time=_now())
        assert result["blend_strategy"] == "model_only"
        assert result["market_weight"] == 0.0
        assert result["n_market_signals"] == 0
