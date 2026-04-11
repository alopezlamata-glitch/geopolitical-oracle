"""
Calibrated market prior blending for geopolitical oracle predictions.

Design principles (Phase A — logodds_v1):
  - Never override the model; blend in log-odds space with a bounded weight.
  - Weight is dynamic: it rises with match quality, liquidity, and freshness,
    and falls when sources conflict or the gate is not passed.
  - Ceiling: w_market <= 0.70 (model always retains at least 30% weight).
  - Gate: minimum match_score and recency thresholds must pass, or the market
    layer is skipped entirely and p_final = p_model.
  - Full audit trail returned in BlendResult for every call.

Blend strategy labels:
  model_only          — gate failed, no market signal used
  model_plus_market   — market weight < 0.40
  market_dominant     — market weight >= 0.40 (model still at 30%+ minimum)

Phase B will add Platt recalibration of the blend weights from resolved history.
Phase C will learn alpha/beta/bias from a logistic regression on resolved pairs.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Gate thresholds ────────────────────────────────────────────────────────────
# Keyword overlap is coarser than semantic matching, so threshold is moderate.
# Raise to 0.60+ once we have semantic embeddings.
GATE_MIN_MATCH_SCORE: float = 0.40
GATE_MAX_AGE_HOURS: float = 48.0      # stale after 48h

# ── Blend limits ───────────────────────────────────────────────────────────────
W_MARKET_MIN: float = 0.0
W_MARKET_CAP: float = 0.70            # model always retains >= 30%

# Liquidity normalisation denominators
_META_FORECASTERS_FULL: int = 300     # ≥300 forecasters → liquidity_score = 1.0
_POLY_VOLUME_FULL: float = 100_000.0  # ≥$100k volume   → liquidity_score = 1.0

BLEND_STRATEGY_VERSION: str = "logodds_v1"


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class MarketSignal:
    """
    Enriched market observation from a single source.

    All quality scores are in [0, 1]; probability_yes is clipped to [0.01, 0.99].
    """
    probability_yes: float          # community prediction / YES price, clipped
    source: str                     # "metaculus" | "polymarket"
    source_question_id: Optional[str]
    snapshot_time: datetime         # when the collector ran (upper bound on staleness)
    match_score: float              # 0–1 keyword overlap between question and market title
    liquidity_score: float          # 0–1 normalised forecasters / volume
    forecasters: Optional[int]      # Metaculus only
    volume_usd: Optional[float]     # Polymarket only
    resolution_compatible: bool = True  # v1: always True; v2 will parse resolution clauses


@dataclass
class BlendResult:
    """
    Complete audit record of one blend operation.

    Every field here is written to the predictions table so the blend can be
    re-examined, recalibrated, or rolled back from audit data alone.
    """
    p_final: float
    p_model: float
    p_market: Optional[float]       # geometric mean of all market signals (None if unused)
    market_weight: float            # actual w_market applied (0.0 if gate failed)
    market_sources: list[str]       # e.g. ["metaculus", "polymarket"]
    best_match_score: Optional[float]
    blend_strategy: str             # "model_only" | "model_plus_market" | "market_dominant"
    blend_strategy_version: str
    gate_passed: bool
    gate_reason: Optional[str]      # None when gate passes; explanation when it fails
    n_signals: int                  # number of market signals used


# ── Internal helpers ───────────────────────────────────────────────────────────

def _clip(p: float) -> float:
    return max(0.01, min(0.99, float(p)))


def _logit(p: float) -> float:
    p = _clip(p)
    return math.log(p / (1.0 - p))


def _sigmoid(lo: float) -> float:
    return 1.0 / (1.0 + math.exp(-lo))


def _liquidity_score(sig: MarketSignal) -> float:
    """Normalise liquidity to [0, 1]."""
    if sig.source == "metaculus" and sig.forecasters is not None:
        return min(sig.forecasters / _META_FORECASTERS_FULL, 1.0)
    if sig.source == "polymarket" and sig.volume_usd is not None:
        return min(sig.volume_usd / _POLY_VOLUME_FULL, 1.0)
    return sig.liquidity_score   # caller-supplied fallback


def _recency_score(sig: MarketSignal, as_of_time: datetime) -> float:
    """
    Freshness of the market snapshot.

    Assumes snapshot_time <= as_of_time (collection always precedes inference).
    Returns 1.0 for snapshots taken within 1 hour, decaying to 0.10 at 48 h.
    """
    delta_h = max(
        (as_of_time - sig.snapshot_time).total_seconds() / 3600.0,
        0.0,
    )
    if delta_h <= 1.0:
        return 1.0
    if delta_h <= 6.0:
        return 0.90
    if delta_h <= 12.0:
        return 0.75
    if delta_h <= 24.0:
        return 0.55
    if delta_h <= 48.0:
        return 0.30
    return 0.10   # very stale — but still allowed if gate says ok


def _conflict_penalty(signals: list[MarketSignal]) -> float:
    """
    Penalty when two sources disagree strongly (|Δp| > 0.20).

    Keeps multi-source blends honest when Metaculus and Polymarket diverge.
    """
    if len(signals) < 2:
        return 0.0
    probs = [s.probability_yes for s in signals]
    max_spread = max(probs) - min(probs)
    if max_spread > 0.20:
        return 0.15
    return 0.0


# ── Public API ─────────────────────────────────────────────────────────────────

def resolve_market_signal(
    *,
    metaculus_p: Optional[float] = None,
    metaculus_forecasters: Optional[int] = None,
    polymarket_p: Optional[float] = None,
    polymarket_volume: Optional[float] = None,
    polymarket_match_score: Optional[float] = None,
    as_of_time: Optional[datetime] = None,
) -> list[MarketSignal]:
    """
    Build a list of MarketSignal objects from raw collector outputs.

    Accepts raw floats so callers don't need to construct dataclasses.
    Returns an empty list if both sources are unavailable.
    """
    if as_of_time is None:
        as_of_time = datetime.now(timezone.utc)

    signals: list[MarketSignal] = []

    if metaculus_p is not None and 0.01 < metaculus_p < 0.99:
        forecasters = metaculus_forecasters or 0
        liq = min(forecasters / _META_FORECASTERS_FULL, 1.0) if forecasters else 0.0
        # Metaculus API search returns by relevance; treat 30+ forecasters as minimum
        # match quality. We don't have an explicit text score, so derive from forecasters.
        # Low baseline (0.40): keyword search results are assumed relevant but not exact.
        implied_match = min(0.40 + 0.20 * liq, 0.70)   # range: [0.40, 0.70]
        signals.append(MarketSignal(
            probability_yes=_clip(metaculus_p),
            source="metaculus",
            source_question_id=None,
            snapshot_time=as_of_time,
            match_score=implied_match,
            liquidity_score=liq,
            forecasters=forecasters if forecasters else None,
            volume_usd=None,
            resolution_compatible=True,
        ))

    if polymarket_p is not None and 0.01 < polymarket_p < 0.99:
        vol = polymarket_volume or 0.0
        liq = min(vol / _POLY_VOLUME_FULL, 1.0)
        match = float(polymarket_match_score) if polymarket_match_score is not None else 0.40
        signals.append(MarketSignal(
            probability_yes=_clip(polymarket_p),
            source="polymarket",
            source_question_id=None,
            snapshot_time=as_of_time,
            match_score=match,
            liquidity_score=liq,
            forecasters=None,
            volume_usd=vol if vol else None,
            resolution_compatible=True,
        ))

    return signals


def compute_market_weight(
    signals: list[MarketSignal],
    as_of_time: datetime,
) -> tuple[float, Optional[str]]:
    """
    Compute dynamic market weight and check activation gate.

    Returns (weight, gate_reason):
      - weight = 0.0 and gate_reason is set when the gate fails.
      - weight in (0, W_MARKET_CAP] and gate_reason is None when gate passes.

    Gate conditions (all must pass):
      1. At least one signal with match_score >= GATE_MIN_MATCH_SCORE
      2. At least one signal fresher than GATE_MAX_AGE_HOURS
      3. At least one signal marked resolution_compatible=True

    Weight formula (in [0, W_MARKET_CAP]):
      base     = 0.10
      + 0.30 * best_match_score
      + 0.20 * best_liquidity_score
      + 0.15 * best_recency_score
      + 0.10 * consensus_bonus  (both sources present)
      - conflict_penalty
    """
    if not signals:
        return 0.0, "no market signals available"

    # ── Gate checks ───────────────────────────────────────────────────────────
    best_match = max(s.match_score for s in signals)
    if best_match < GATE_MIN_MATCH_SCORE:
        return 0.0, (
            f"match_score={best_match:.2f} below gate threshold "
            f"({GATE_MIN_MATCH_SCORE:.2f})"
        )

    ages_h = [
        max((as_of_time - s.snapshot_time).total_seconds() / 3600.0, 0.0)
        for s in signals
    ]
    if min(ages_h) > GATE_MAX_AGE_HOURS:
        oldest_h = min(ages_h)
        return 0.0, (
            f"all signals stale (freshest is {oldest_h:.1f}h old, "
            f"gate max {GATE_MAX_AGE_HOURS:.0f}h)"
        )

    if not any(s.resolution_compatible for s in signals):
        return 0.0, "no signal with resolution_compatible=True"

    # ── Dynamic weight ────────────────────────────────────────────────────────
    best_liq = max(_liquidity_score(s) for s in signals)
    best_rec = max(_recency_score(s, as_of_time) for s in signals)
    consensus_bonus = 0.10 if len(signals) >= 2 else 0.0
    penalty = _conflict_penalty(signals)

    raw_weight = (
        0.10
        + 0.30 * best_match
        + 0.20 * best_liq
        + 0.15 * best_rec
        + consensus_bonus
        - penalty
    )
    weight = max(W_MARKET_MIN, min(W_MARKET_CAP, raw_weight))

    logger.debug(
        "market_weight: match=%.2f liq=%.2f rec=%.2f consensus=%.2f penalty=%.2f"
        " → raw=%.3f clamped=%.3f",
        best_match, best_liq, best_rec, consensus_bonus, penalty, raw_weight, weight,
    )
    return weight, None


def blend_market_prior(
    p_model: float,
    signals: list[MarketSignal],
    as_of_time: datetime,
) -> BlendResult:
    """
    Blend model probability with market prior in log-odds space.

    When the gate fails → BlendResult.p_final == p_model, market_weight = 0.0.
    When the gate passes → log-odds weighted blend of p_model and p_market.

    Phase A (logodds_v1):
        logit(p_final) = (1 - w) * logit(p_model) + w * logit(p_market)
        w = dynamic weight from compute_market_weight(), capped at W_MARKET_CAP.

    Phase B (logodds_learned_v1), active when blend_weights.json exists:
        logit(p_final) = alpha * logit(p_model) + beta * w_scale * logit(p_market) + bias
        where alpha/beta/bias are fitted from resolved history, and
        w_scale = (w_market / W_MARKET_CAP) scales beta by market quality.

    The quality gate still applies in both phases: gate fail → model_only.
    """
    p_model = _clip(p_model)
    weight, gate_reason = compute_market_weight(signals, as_of_time)
    gate_passed = gate_reason is None

    p_market: Optional[float] = None
    market_sources: list[str] = []
    best_match: Optional[float] = None

    if gate_passed and signals:
        # Geometric mean in log-odds space
        lo_sum = sum(_logit(s.probability_yes) for s in signals)
        p_market = _clip(_sigmoid(lo_sum / len(signals)))
        market_sources = [s.source for s in signals]
        best_match = max(s.match_score for s in signals)

    if not gate_passed or p_market is None:
        return BlendResult(
            p_final=round(p_model, 4),
            p_model=round(p_model, 4),
            p_market=None,
            market_weight=0.0,
            market_sources=[],
            best_match_score=None,
            blend_strategy="model_only",
            blend_strategy_version=BLEND_STRATEGY_VERSION,
            gate_passed=gate_passed,
            gate_reason=gate_reason,
            n_signals=len(signals),
        )

    # ── Try Phase B: learned weights ──────────────────────────────────────────
    learned = _load_blend_weights_cached()
    if learned is not None:
        alpha = learned["alpha"]
        beta  = learned["beta"]
        bias  = learned["bias"]
        # Scale beta by normalised market quality (w_market / cap)
        w_scale = weight / W_MARKET_CAP
        lo_final = alpha * _logit(p_model) + beta * w_scale * _logit(p_market) + bias
        strategy_version = learned.get("blend_calibrator_version", "logodds_learned_v1")
        logger.info(
            "market blend [learned]: p_model=%.3f p_market=%.3f "
            "alpha=%.3f beta=%.3f w_scale=%.2f bias=%.3f → lo=%.3f",
            p_model, p_market, alpha, beta, w_scale, bias, lo_final,
        )
    else:
        # ── Phase A: fixed dynamic weight ────────────────────────────────────
        lo_final = (1.0 - weight) * _logit(p_model) + weight * _logit(p_market)
        strategy_version = BLEND_STRATEGY_VERSION
        logger.info(
            "market blend [fixed]: p_model=%.3f p_market=%.3f w=%.2f → lo=%.3f",
            p_model, p_market, weight, lo_final,
        )

    p_final = _clip(_sigmoid(lo_final))

    if weight >= 0.40:
        blend_strategy = "market_dominant"
    else:
        blend_strategy = "model_plus_market"

    logger.info(
        "market blend result: p_final=%.3f  strategy=%s  sources=%s",
        p_final, blend_strategy, market_sources,
    )

    return BlendResult(
        p_final=round(p_final, 4),
        p_model=round(p_model, 4),
        p_market=round(p_market, 4),
        market_weight=round(weight, 4),
        market_sources=market_sources,
        best_match_score=round(best_match, 4) if best_match is not None else None,
        blend_strategy=blend_strategy,
        blend_strategy_version=strategy_version,
        gate_passed=gate_passed,
        gate_reason=gate_reason,
        n_signals=len(signals),
    )


# ── Blend weight cache (avoid disk read on every predict call) ────────────────

_blend_weights_cache: Optional[dict] = None
_blend_weights_mtime: float = 0.0

_BLEND_WEIGHTS_FILE = Path(__file__).parent.parent / "data" / "model" / "blend_weights.json"


def _load_blend_weights_cached() -> Optional[dict]:
    """
    Load blend_weights.json with a simple mtime cache.

    Re-reads the file if it was modified since last load.
    Returns None if the file doesn't exist (triggers Phase A fallback).
    """
    global _blend_weights_cache, _blend_weights_mtime

    _BLEND_WEIGHTS_PATH = _BLEND_WEIGHTS_FILE
    if not _BLEND_WEIGHTS_PATH.exists():
        return None

    try:
        mtime = _BLEND_WEIGHTS_PATH.stat().st_mtime
        if _blend_weights_cache is not None and mtime == _blend_weights_mtime:
            return _blend_weights_cache

        from predictor.blend_calibrator import load_blend_weights
        data = load_blend_weights()
        _blend_weights_cache = data
        _blend_weights_mtime = mtime
        if data:
            logger.debug(
                "blend weights loaded: alpha=%.4f beta=%.4f bias=%.4f",
                data["alpha"], data["beta"], data["bias"],
            )
        return data
    except Exception as e:
        logger.debug("blend weights cache miss: %s", e)
        return None
