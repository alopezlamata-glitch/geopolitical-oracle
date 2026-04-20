"""
Ollama-based reasoning predictor for domains without a trained XGBoost model.

Used for: political, economic, legal, entertainment questions.
Approach: superforecaster prompt with chain-of-thought reasoning.

The probability output is blended with market signals through the existing
blend layer (predictor/market_prior.py), keeping the audit trail consistent.

Non-fatal: if Ollama is unavailable, falls back to market prior or 0.5.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

_PREDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "probability": {"type": "number"},
        "confidence": {"type": "number"},
        "reasoning": {"type": "string"},
        "key_factors_for": {"type": "array", "items": {"type": "string"}},
        "key_factors_against": {"type": "array", "items": {"type": "string"}},
        "analogous_cases": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["probability", "confidence", "reasoning"],
}

_SYSTEM_PROMPT = """\
You are a superforecaster trained in calibrated probabilistic reasoning.
Your task: estimate the probability that a specific event will occur by a deadline.

Guidelines:
- Use the outside view: what fraction of similar historical cases resolved YES?
- Adjust for inside view: what specific evidence pushes above/below the base rate?
- Be calibrated: if you say 70%, events like this should happen ~70% of the time.
- Avoid extremes (< 0.05 or > 0.95) unless evidence is overwhelming.
- Output only valid JSON. Do not add keys beyond those requested."""

_PROMPT_TEMPLATE = """\
Binary forecasting question: {question}
Resolution deadline: {deadline}
Domain: {event_family}

{wiki_section}Evidence from recent news ({n} headlines):
{headlines}

{market_section}Rate the probability that this event will occur by the deadline.

Respond with JSON:
{{
  "probability": <float 0.0-1.0>,
  "confidence": <your confidence in this estimate, 0.0-1.0>,
  "reasoning": "<2-3 sentence reasoning chain>",
  "key_factors_for": ["<factor 1>", "<factor 2>"],
  "key_factors_against": ["<factor 1>", "<factor 2>"],
  "analogous_cases": ["<historical case>"]
}}"""


def _clip(v, lo=0.01, hi=0.99) -> float:
    try:
        return max(lo, min(hi, float(v)))
    except (TypeError, ValueError):
        return 0.5


def _logit(p: float) -> float:
    p = max(0.001, min(0.999, p))
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


async def _predict_async(
    question: str,
    deadline: str,
    event_family: str,
    headlines: list[str],
    wiki_context: str = "",
    market_signals: Optional[dict] = None,
) -> dict:
    """Core async prediction via Ollama reasoning."""
    from llm.client import get_client
    client = get_client()

    # Check Ollama availability
    if not await client.is_available():
        logger.warning("ollama_predictor: Ollama unavailable")
        return {"error": "ollama_unavailable"}

    # Build headline block (most recent first, truncated)
    truncated = [h[:150] for h in headlines[:25]]
    headline_block = "\n".join(f"- {h}" for h in truncated)

    wiki_section = (
        f"Background context:\n{wiki_context[:1500]}\n\n"
        if wiki_context.strip() else ""
    )

    market_section = ""
    if market_signals:
        parts = []
        if market_signals.get("metaculus_p") is not None:
            parts.append(f"Metaculus crowd forecast: {market_signals['metaculus_p']:.0%}")
        if market_signals.get("polymarket_p") is not None:
            parts.append(f"Polymarket (real-money): {market_signals['polymarket_p']:.0%}")
        if parts:
            market_section = "Prediction market signals:\n" + "\n".join(f"- {p}" for p in parts) + "\n\n"

    prompt = _PROMPT_TEMPLATE.format(
        question=question[:300],
        deadline=deadline,
        event_family=event_family,
        wiki_section=wiki_section,
        n=len(truncated),
        headlines=headline_block,
        market_section=market_section,
    )

    raw = await client.generate_json(
        prompt=prompt,
        system=_SYSTEM_PROMPT,
        temperature=0.1,   # small temperature for some diversity in reasoning
        schema=_PREDICT_SCHEMA,
    )

    if not isinstance(raw, dict):
        return {"error": "invalid_response", "raw": str(raw)[:200]}

    prob = _clip(raw.get("probability", 0.5))
    confidence = _clip(raw.get("confidence", 0.5), 0.0, 1.0)
    reasoning = raw.get("reasoning", "")
    factors_for = raw.get("key_factors_for", [])
    factors_against = raw.get("key_factors_against", [])
    analogous = raw.get("analogous_cases", [])

    return {
        "probability": prob,
        "confidence": confidence,
        "reasoning": reasoning,
        "key_factors_for": factors_for,
        "key_factors_against": factors_against,
        "analogous_cases": analogous,
        "model_used": client.text_model,
    }


def predict_ollama(
    question: str,
    event_family: str,
    headlines: list[str],
    deadline: Optional[datetime] = None,
    wiki_context: str = "",
    metaculus_p: Optional[float] = None,
    polymarket_p: Optional[float] = None,
    metaculus_forecasters: Optional[int] = None,
    polymarket_volume: Optional[float] = None,
    polymarket_match_score: Optional[float] = None,
    as_of_time: Optional[datetime] = None,
) -> dict:
    """
    Synchronous wrapper. Returns prediction dict compatible with inference.predict().

    The returned dict has the same keys as inference.predict() so callers
    don't need to special-case the predictor type.

    Falls back to market prior or 0.5 if Ollama unavailable.
    """
    import scipy.stats
    from predictor.market_prior import (
        BlendResult, resolve_market_signal, blend_market_prior,
    )

    if as_of_time is None:
        as_of_time = datetime.now(timezone.utc)

    deadline_str = deadline.strftime("%Y-%m-%d") if deadline else "unspecified"

    market_signals = {}
    if metaculus_p is not None:
        market_signals["metaculus_p"] = metaculus_p
    if polymarket_p is not None:
        market_signals["polymarket_p"] = polymarket_p

    t0 = time.monotonic()

    # ── 1. Ollama reasoning ───────────────────────────────────────────────────
    ollama_result: dict = {}
    try:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            logger.warning("ollama_predictor: called from running loop — skipping LLM")
        else:
            ollama_result = asyncio.run(_predict_async(
                question=question,
                deadline=deadline_str,
                event_family=event_family,
                headlines=headlines,
                wiki_context=wiki_context,
                market_signals=market_signals or None,
            ))
    except Exception as e:
        logger.warning("ollama_predictor: reasoning failed: %s", e)

    latency_ms = (time.monotonic() - t0) * 1000

    # ── 2. Extract raw probability ────────────────────────────────────────────
    ollama_available = "error" not in ollama_result and "probability" in ollama_result

    if ollama_available:
        raw_prob = ollama_result["probability"]
    else:
        # Fall back to market prior or 0.5
        valid_markets = [p for p in [metaculus_p, polymarket_p] if p is not None and 0 < p < 1]
        raw_prob = float(sum(valid_markets) / len(valid_markets)) if valid_markets else 0.5
        logger.info("ollama_predictor: using market/prior fallback p=%.3f", raw_prob)

    calibrated_prob = max(0.01, min(0.99, raw_prob))

    # ── 3. Market blend (same as XGBoost path) ────────────────────────────────
    signals = resolve_market_signal(
        metaculus_p=metaculus_p,
        metaculus_forecasters=metaculus_forecasters,
        polymarket_p=polymarket_p,
        polymarket_volume=polymarket_volume,
        polymarket_match_score=polymarket_match_score,
        as_of_time=as_of_time,
    )
    blend = blend_market_prior(calibrated_prob, signals, as_of_time)
    final_prob = max(0.01, min(0.99, blend.p_final))

    # ── 4. Confidence interval ────────────────────────────────────────────────
    # Use beta heuristic width scaled by Ollama confidence (if available)
    ollama_confidence = ollama_result.get("confidence", 0.5) if ollama_available else 0.3
    eff_n = max(5, int(ollama_confidence * 50))   # more confidence → narrower CI
    alpha_p = final_prob * eff_n
    beta_p = (1 - final_prob) * eff_n
    lo, hi = scipy.stats.beta.interval(0.80, max(0.1, alpha_p), max(0.1, beta_p))

    answer = "YES" if final_prob >= 0.5 else "NO"

    # ── 5. Attribution (reasoning chain as "features") ────────────────────────
    attribution = {}
    if ollama_available:
        attribution = {
            "reasoning": ollama_result.get("reasoning", ""),
            "key_factors_for": ollama_result.get("key_factors_for", []),
            "key_factors_against": ollama_result.get("key_factors_against", []),
            "analogous_cases": ollama_result.get("analogous_cases", []),
            "model_used": ollama_result.get("model_used", ""),
        }

    logger.info(
        "ollama_predictor[%s]: p=%.3f answer=%s llm_ok=%s latency=%.0fms",
        event_family, final_prob, answer, ollama_available, latency_ms,
    )

    return {
        # Core prediction (same keys as inference.predict())
        "raw_prob": round(raw_prob, 4),
        "calibrated_prob": round(final_prob, 4),
        "ci_lo": round(float(lo), 4),
        "ci_hi": round(float(hi), 4),
        "ci_method": "ollama_confidence",
        "answer": answer,
        "untrained": False,
        # Market blend audit trail
        "p_model_raw": round(blend.p_model, 4),
        "p_market_raw": blend.p_market,
        "market_weight": blend.market_weight,
        "market_sources": blend.market_sources,
        "market_match_score": blend.best_match_score,
        "blend_strategy": blend.blend_strategy,
        "blend_strategy_version": blend.blend_strategy_version,
        "market_gate_passed": blend.gate_passed,
        "market_gate_reason": blend.gate_reason,
        "n_market_signals": blend.n_signals,
        "market_override": blend.gate_passed and blend.n_signals > 0,
        # Ollama-specific
        "predictor": "ollama_reasoning_v1",
        "ollama_available": ollama_available,
        "ollama_latency_ms": round(latency_ms, 0),
        "attribution": attribution,
    }
