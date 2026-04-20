"""
Domain-specific LLM feature extractor dispatcher.

Extends the core conflict features (llm/text_features.py) with domain-specific
indicators for political, economic, and legal questions.

All extractors:
- Use Ollama local LLM (free, no API cost)
- Are non-fatal (return zeros if Ollama unavailable)
- Store results in feature_snapshots.explicit_features under domain-namespaced keys
- Are NOT used by the v3 XGBoost model (stored for future v4 domain models)

Domain feature sets:
  conflict:  llm_threat_level, llm_escalation, llm_deescalation, ...  (existing)
  political: pol_approval_pressure, pol_coalition_stability, pol_resignation_signals,
             pol_electoral_proximity, pol_judicial_pressure
  economic:  eco_rate_change_prob, eco_gdp_momentum, eco_debt_stress,
             eco_market_volatility, eco_policy_uncertainty
  legal:     leg_arrest_probability, leg_extradition_risk, leg_evidence_strength,
             leg_jurisdictional_support, leg_precedent_match
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from llm.client import OllamaClient, get_client

logger = logging.getLogger(__name__)

MAX_HEADLINES = 20
MAX_HEADLINE_LEN = 120

# ── JSON Schema for structured output (Ollama >= 0.4) ─────────────────────────

_POLITICAL_SCHEMA = {
    "type": "object",
    "properties": {
        "approval_pressure": {"type": "number"},
        "coalition_stability": {"type": "number"},
        "resignation_signals": {"type": "number"},
        "electoral_proximity": {"type": "number"},
        "judicial_pressure": {"type": "number"},
    },
    "required": [
        "approval_pressure", "coalition_stability", "resignation_signals",
        "electoral_proximity", "judicial_pressure",
    ],
}

_ECONOMIC_SCHEMA = {
    "type": "object",
    "properties": {
        "rate_change_prob": {"type": "number"},
        "gdp_momentum": {"type": "number"},
        "debt_stress": {"type": "number"},
        "market_volatility": {"type": "number"},
        "policy_uncertainty": {"type": "number"},
    },
    "required": [
        "rate_change_prob", "gdp_momentum", "debt_stress",
        "market_volatility", "policy_uncertainty",
    ],
}

_LEGAL_SCHEMA = {
    "type": "object",
    "properties": {
        "arrest_probability": {"type": "number"},
        "extradition_risk": {"type": "number"},
        "evidence_strength": {"type": "number"},
        "jurisdictional_support": {"type": "number"},
        "precedent_match": {"type": "number"},
    },
    "required": [
        "arrest_probability", "extradition_risk", "evidence_strength",
        "jurisdictional_support", "precedent_match",
    ],
}

# ── System prompts ─────────────────────────────────────────────────────────────

_POLITICAL_SYSTEM = (
    "You are a political analyst specializing in governmental stability. "
    "Extract numerical indicators from news about political leaders and governments. "
    "Each value must be between 0.0 and 1.0. Output only valid JSON."
)

_ECONOMIC_SYSTEM = (
    "You are a macroeconomic analyst. "
    "Extract numerical risk indicators from economic news. "
    "Each value must be between 0.0 and 1.0. Output only valid JSON."
)

_LEGAL_SYSTEM = (
    "You are a legal analyst specializing in international law and criminal proceedings. "
    "Extract numerical probability indicators from legal news. "
    "Each value must be between 0.0 and 1.0. Output only valid JSON."
)

# ── Prompt templates with few-shot calibration examples ───────────────────────

_POLITICAL_PROMPT = """\
Question: {question}

Recent political news ({n} headlines):
{headlines}

Rate each indicator from 0.0 to 1.0:
- approval_pressure: How much public/parliamentary pressure is the leader under? \
(0=strong support, 1=imminent collapse)
- coalition_stability: How stable is the governing coalition? \
(0=fragile/broken, 1=solid majority)
- resignation_signals: Are there direct signals of resignation/departure? \
(0=none, 1=official announcement)
- electoral_proximity: How soon is the next election or vote of no confidence? \
(0=years away, 1=imminent within weeks)
- judicial_pressure: How much legal/judicial pressure is on the leader? \
(0=none, 1=criminal charges/impeachment)

Few-shot examples:
- "PM survives confidence vote 240-180" -> coalition_stability=0.55, resignation_signals=0.1
- "Leader submits resignation letter" -> resignation_signals=0.95, approval_pressure=0.9
- "Opposition files no-confidence motion" -> electoral_proximity=0.8, approval_pressure=0.7

Respond with JSON only:
{{"approval_pressure": 0.0, "coalition_stability": 0.0, "resignation_signals": 0.0, \
"electoral_proximity": 0.0, "judicial_pressure": 0.0}}"""

_ECONOMIC_PROMPT = """\
Question: {question}

Recent economic news ({n} headlines):
{headlines}

Rate each indicator from 0.0 to 1.0:
- rate_change_prob: Probability of an interest rate change at next meeting? \
(0=rates unchanged, 1=rate change certain)
- gdp_momentum: Momentum of economic growth? \
(0=sharp contraction, 1=strong expansion)
- debt_stress: Level of sovereign debt stress? \
(0=no stress, 1=imminent default)
- market_volatility: Financial market volatility level? \
(0=calm, 1=extreme volatility/crisis)
- policy_uncertainty: Level of economic policy uncertainty? \
(0=clear stable policy, 1=total uncertainty)

Few-shot examples:
- "Fed signals rate hold, inflation cooling" -> rate_change_prob=0.15, policy_uncertainty=0.2
- "IMF warns of default risk, bond yields spike" -> debt_stress=0.8, market_volatility=0.75
- "GDP growth beats forecast at 3.2%" -> gdp_momentum=0.8, market_volatility=0.2

Respond with JSON only:
{{"rate_change_prob": 0.0, "gdp_momentum": 0.0, "debt_stress": 0.0, \
"market_volatility": 0.0, "policy_uncertainty": 0.0}}"""

_LEGAL_PROMPT = """\
Question: {question}

Recent legal/judicial news ({n} headlines):
{headlines}

Rate each indicator from 0.0 to 1.0:
- arrest_probability: Probability of imminent arrest based on news signals? \
(0=no indication, 1=arrest warrant issued/imminent)
- extradition_risk: Risk of extradition proceeding? \
(0=no request, 1=active extradition in progress)
- evidence_strength: How strong is the evidence based on reporting? \
(0=speculation only, 1=confirmed documentary evidence)
- jurisdictional_support: How supportive are jurisdictions of the legal action? \
(0=no cooperation, 1=full multilateral cooperation)
- precedent_match: How well does this match historical precedent for this outcome? \
(0=unprecedented, 1=clear historical precedent)

Few-shot examples:
- "Interpol issues red notice" -> arrest_probability=0.7, extradition_risk=0.6
- "Country refuses extradition request" -> extradition_risk=0.1, jurisdictional_support=0.1
- "Prosecutors present key witness testimony" -> evidence_strength=0.75

Respond with JSON only:
{{"arrest_probability": 0.0, "extradition_risk": 0.0, "evidence_strength": 0.0, \
"jurisdictional_support": 0.0, "precedent_match": 0.0}}"""


def _clip(v) -> float:
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return 0.0


def _build_headline_block(headlines: list[str]) -> tuple[str, int]:
    truncated = [h[:MAX_HEADLINE_LEN] for h in headlines[:MAX_HEADLINES]]
    return "\n".join(f"- {h}" for h in truncated), len(truncated)


# ── Political features ─────────────────────────────────────────────────────────

async def _extract_political_async(
    question: str,
    headlines: list[str],
    client: OllamaClient,
) -> dict[str, float]:
    block, n = _build_headline_block(headlines)
    prompt = _POLITICAL_PROMPT.format(question=question[:200], headlines=block, n=n)
    raw = await client.generate_json(
        prompt=prompt,
        system=_POLITICAL_SYSTEM,
        temperature=0.0,
        schema=_POLITICAL_SCHEMA,
    )
    if not isinstance(raw, dict):
        return _political_zeros()
    return {
        "pol_approval_pressure":   _clip(raw.get("approval_pressure", 0.0)),
        "pol_coalition_stability": _clip(raw.get("coalition_stability", 0.5)),
        "pol_resignation_signals": _clip(raw.get("resignation_signals", 0.0)),
        "pol_electoral_proximity": _clip(raw.get("electoral_proximity", 0.0)),
        "pol_judicial_pressure":   _clip(raw.get("judicial_pressure", 0.0)),
        "pol_llm_available": 1.0,
    }


def _political_zeros() -> dict[str, float]:
    return {
        "pol_approval_pressure": 0.0,
        "pol_coalition_stability": 0.0,
        "pol_resignation_signals": 0.0,
        "pol_electoral_proximity": 0.0,
        "pol_judicial_pressure": 0.0,
        "pol_llm_available": 0.0,
    }


# ── Economic features ──────────────────────────────────────────────────────────

async def _extract_economic_async(
    question: str,
    headlines: list[str],
    client: OllamaClient,
) -> dict[str, float]:
    block, n = _build_headline_block(headlines)
    prompt = _ECONOMIC_PROMPT.format(question=question[:200], headlines=block, n=n)
    raw = await client.generate_json(
        prompt=prompt,
        system=_ECONOMIC_SYSTEM,
        temperature=0.0,
        schema=_ECONOMIC_SCHEMA,
    )
    if not isinstance(raw, dict):
        return _economic_zeros()
    return {
        "eco_rate_change_prob":  _clip(raw.get("rate_change_prob", 0.0)),
        "eco_gdp_momentum":      _clip(raw.get("gdp_momentum", 0.5)),
        "eco_debt_stress":       _clip(raw.get("debt_stress", 0.0)),
        "eco_market_volatility": _clip(raw.get("market_volatility", 0.0)),
        "eco_policy_uncertainty": _clip(raw.get("policy_uncertainty", 0.0)),
        "eco_llm_available": 1.0,
    }


def _economic_zeros() -> dict[str, float]:
    return {
        "eco_rate_change_prob": 0.0,
        "eco_gdp_momentum": 0.0,
        "eco_debt_stress": 0.0,
        "eco_market_volatility": 0.0,
        "eco_policy_uncertainty": 0.0,
        "eco_llm_available": 0.0,
    }


# ── Legal features ─────────────────────────────────────────────────────────────

async def _extract_legal_async(
    question: str,
    headlines: list[str],
    client: OllamaClient,
) -> dict[str, float]:
    block, n = _build_headline_block(headlines)
    prompt = _LEGAL_PROMPT.format(question=question[:200], headlines=block, n=n)
    raw = await client.generate_json(
        prompt=prompt,
        system=_LEGAL_SYSTEM,
        temperature=0.0,
        schema=_LEGAL_SCHEMA,
    )
    if not isinstance(raw, dict):
        return _legal_zeros()
    return {
        "leg_arrest_probability":     _clip(raw.get("arrest_probability", 0.0)),
        "leg_extradition_risk":       _clip(raw.get("extradition_risk", 0.0)),
        "leg_evidence_strength":      _clip(raw.get("evidence_strength", 0.0)),
        "leg_jurisdictional_support": _clip(raw.get("jurisdictional_support", 0.0)),
        "leg_precedent_match":        _clip(raw.get("precedent_match", 0.0)),
        "leg_llm_available": 1.0,
    }


def _legal_zeros() -> dict[str, float]:
    return {
        "leg_arrest_probability": 0.0,
        "leg_extradition_risk": 0.0,
        "leg_evidence_strength": 0.0,
        "leg_jurisdictional_support": 0.0,
        "leg_precedent_match": 0.0,
        "leg_llm_available": 0.0,
    }


# ── Public dispatcher ──────────────────────────────────────────────────────────

async def _dispatch_async(
    event_family: str,
    question: str,
    headlines: list[str],
    client: OllamaClient,
) -> dict[str, float]:
    if event_family == "political":
        return await _extract_political_async(question, headlines, client)
    elif event_family == "economic":
        return await _extract_economic_async(question, headlines, client)
    elif event_family == "legal":
        return await _extract_legal_async(question, headlines, client)
    # conflict and others: no domain-specific features (handled by text_features.py)
    return {}


def extract_domain_features(
    event_family: str,
    question: str,
    headlines: list[str],
    client: Optional[OllamaClient] = None,
) -> dict[str, float]:
    """
    Synchronous dispatcher. Extracts domain-specific LLM features based on event_family.

    Returns a flat dict of domain-namespaced features (pol_*, eco_*, leg_*).
    Returns empty dict for conflict/unknown families (handled by text_features.py).
    Returns zero-filled features if Ollama unavailable (non-fatal).

    Args:
        event_family : 'political' | 'economic' | 'legal' | 'conflict' | ...
        question     : binary question text
        headlines    : event headlines (most recent first)
        client       : OllamaClient singleton; auto-creates if None
    """
    if event_family not in ("political", "economic", "legal"):
        return {}

    if not headlines:
        if event_family == "political":
            return _political_zeros()
        elif event_family == "economic":
            return _economic_zeros()
        elif event_family == "legal":
            return _legal_zeros()
        return {}

    if client is None:
        client = get_client()

    t0 = time.monotonic()
    try:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            logger.warning("domain_features: called from running loop — skipping")
            if event_family == "political":
                return _political_zeros()
            elif event_family == "economic":
                return _economic_zeros()
            elif event_family == "legal":
                return _legal_zeros()
            return {}

        result = asyncio.run(_dispatch_async(event_family, question, headlines, client))
        latency_ms = (time.monotonic() - t0) * 1000
        logger.info(
            "domain_features[%s]: extracted %d features in %.0fms",
            event_family, len(result), latency_ms,
        )
        return result

    except Exception as e:
        logger.warning("domain_features[%s]: extraction failed: %s", event_family, e)
        if event_family == "political":
            return _political_zeros()
        elif event_family == "economic":
            return _economic_zeros()
        elif event_family == "legal":
            return _legal_zeros()
        return {}
