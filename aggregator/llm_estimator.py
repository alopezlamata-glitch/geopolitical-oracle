from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Optional

import aiohttp

from collector.base import EvidenceBlock
from pipeline.temporal import TemporalSignal
from pipeline.scorer import RiskScore
from pipeline.clusterer import ClusterResult

logger = logging.getLogger(__name__)

_DEFAULT_OLLAMA_URL = "http://localhost:11434"
_DEFAULT_MODEL = "llama3.1:8b"
_TIMEOUT = aiohttp.ClientTimeout(total=180)

_SYSTEM_PROMPT = """\
You are a world-class superforecaster with deep expertise in geopolitics, \
economics, and calibrated probabilistic reasoning.

Your core principles:
- Always reason from base rates first, then update on current evidence
- Avoid round numbers (never exactly 50%, 30%, 20%, 10%, etc.)
- Generate THREE distinct scenarios: base case, upside, and downside
- Each scenario must have a weight (probability of that scenario) and a \
probability_of_yes (if this scenario plays out, how likely is YES?)
- Weights must sum to exactly 1.0
- Express genuine uncertainty — never fake precision when evidence is thin
- Never anchor to market prices (you are not shown any)

You MUST respond with ONLY a valid JSON object. No prose, no markdown fences.\
"""

_USER_TEMPLATE = """\
QUESTION: {question}

═══ BACKGROUND (Wikipedia) ═══
{wikipedia}

═══ RECENT NEWS (last 72h) ═══
{news}

═══ MEDIA SENTIMENT (GDELT snapshot) ═══
{gdelt_snapshot}

═══ TEMPORAL INTELLIGENCE (30-day trend) ═══
{temporal}

═══ RISK SCORE ═══
{risk_score}

═══ CONFLICT DATA (ACLED) ═══
{acled}

═══ EVENT CLUSTERS ═══
{clusters}

═══ YOUR TASK ═══
Generate THREE forecast scenarios for this question. For each scenario:
1. Name it descriptively (e.g., "Diplomatic breakthrough", "Status quo", "Escalation")
2. Assign it a weight (how likely is this scenario to be the one that unfolds?) — weights must sum to 1.0
3. Give the probability_of_yes WITHIN that scenario
4. Write a concise causal narrative (2-3 sentences)

The overall probability of YES = sum(weight_i * probability_of_yes_i).

Respond ONLY with this JSON:
{{
  "scenarios": [
    {{
      "name": "<descriptive scenario name>",
      "weight": <float, scenario likelihood>,
      "probability_of_yes": <float 0.05-0.95>,
      "narrative": "<2-3 sentence causal chain explaining this scenario>"
    }},
    {{
      "name": "<second scenario>",
      "weight": <float>,
      "probability_of_yes": <float>,
      "narrative": "<narrative>"
    }},
    {{
      "name": "<third scenario>",
      "weight": <float>,
      "probability_of_yes": <float>,
      "narrative": "<narrative>"
    }}
  ],
  "confidence": "<low|medium|high>",
  "key_factors": ["<factor 1>", "<factor 2>", "<factor 3>"],
  "abstain": false
}}\
"""


def _build_prompt(
    question: str,
    wikipedia_content: str,
    rss_headlines: str,
    gdelt_tone: Optional[float],
    gdelt_count: Optional[int],
    temporal: Optional[TemporalSignal],
    risk_score: Optional[RiskScore],
    acled_content: Optional[str],
    cluster_result: Optional[ClusterResult],
) -> str:
    wiki = wikipedia_content.strip() or "(no Wikipedia background available)"
    news = rss_headlines.strip() or "(no recent news found in last 72h)"

    if gdelt_tone is not None:
        sentiment = "positive" if gdelt_tone > 2 else "negative" if gdelt_tone < -2 else "neutral"
        gdelt_snapshot = f"Tone: {gdelt_tone:+.2f} ({sentiment}) | Articles: {gdelt_count}"
    else:
        gdelt_snapshot = "(GDELT snapshot unavailable)"

    temporal_text = temporal.to_prompt_text() if temporal else "(30-day time series unavailable)"
    risk_text = risk_score.to_prompt_text() if risk_score else "(risk score unavailable)"
    acled_text = acled_content.strip() if acled_content else "(ACLED data not configured or no events found)"
    cluster_text = cluster_result.to_prompt_text() if cluster_result else "(semantic clustering unavailable)"

    return _USER_TEMPLATE.format(
        question=question,
        wikipedia=wiki,
        news=news,
        gdelt_snapshot=gdelt_snapshot,
        temporal=temporal_text,
        risk_score=risk_text,
        acled=acled_text,
        clusters=cluster_text,
    )


def _extract_evidence_fields(
    evidence: list[EvidenceBlock],
) -> dict[str, Any]:
    """Extract prompt-relevant fields. Market probabilities are deliberately excluded."""
    fields: dict[str, Any] = {
        "wikipedia_content": "",
        "rss_headlines": "",
        "gdelt_tone": None,
        "gdelt_count": None,
        "acled_content": None,
    }
    for block in evidence:
        if block.source == "wikipedia" and block.content:
            fields["wikipedia_content"] = block.content
        elif block.source == "rss" and block.content and block.quality != "insufficient":
            fields["rss_headlines"] = block.content
        elif block.source == "gdelt" and block.quality != "insufficient":
            fields["gdelt_tone"] = block.metadata.get("avg_tone")
            fields["gdelt_count"] = block.metadata.get("article_count")
        elif block.source == "acled" and block.quality != "insufficient":
            fields["acled_content"] = block.content
    return fields


def _parse_scenarios(raw: str) -> dict[str, Any]:
    """Robust JSON extraction with three fallback layers."""
    # Layer 1: direct parse
    try:
        return _validate(json.loads(raw))
    except (json.JSONDecodeError, ValueError):
        pass

    # Layer 2: extract first {...} block
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return _validate(json.loads(match.group(0)))
        except (json.JSONDecodeError, ValueError):
            pass

    # Layer 3: construct minimal valid result
    logger.warning("llm_estimator: JSON parse failed completely — abstaining")
    return {"scenarios": [], "confidence": "low", "key_factors": [], "abstain": True}


def _validate(result: dict) -> dict:
    """Validate and normalize the parsed scenario result."""
    scenarios = result.get("scenarios", [])

    if not isinstance(scenarios, list) or len(scenarios) == 0:
        result["abstain"] = True
        return result

    # Normalize each scenario
    valid_scenarios = []
    for s in scenarios:
        try:
            w = float(s.get("weight", 0))
            p = float(s.get("probability_of_yes", 0.5))
            p = max(0.05, min(0.95, p))
            valid_scenarios.append({
                "name": str(s.get("name", "Scenario")),
                "weight": w,
                "probability_of_yes": p,
                "narrative": str(s.get("narrative", "")),
            })
        except (TypeError, ValueError):
            continue

    if not valid_scenarios:
        result["abstain"] = True
        return result

    # Normalize weights to sum to 1.0
    total_weight = sum(s["weight"] for s in valid_scenarios)
    if total_weight > 0:
        for s in valid_scenarios:
            s["weight"] = round(s["weight"] / total_weight, 4)

    result["scenarios"] = valid_scenarios

    # Compute aggregated probability (weighted sum)
    result["probability"] = sum(
        s["weight"] * s["probability_of_yes"] for s in valid_scenarios
    )

    result.setdefault("abstain", False)
    if result.get("confidence") not in ("low", "medium", "high"):
        result["confidence"] = "medium"
    if not isinstance(result.get("key_factors"), list):
        result["key_factors"] = []

    return result


async def check_ollama(ollama_url: str, model: str) -> None:
    """Verify Ollama is reachable and model is available."""
    check_timeout = aiohttp.ClientTimeout(total=5)
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(f"{ollama_url}/api/tags", timeout=check_timeout) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Ollama returned HTTP {resp.status}")
                data = await resp.json()
        available = [m["name"] for m in data.get("models", [])]
        model_base = model.split(":")[0]
        if not any(model_base in m for m in available):
            avail_str = ", ".join(available) or "(none pulled yet)"
            raise RuntimeError(
                f"Model '{model}' not found in Ollama.\n"
                f"  Available: {avail_str}\n"
                f"  Fix: ollama pull {model}"
            )
    except aiohttp.ClientConnectorError:
        raise RuntimeError(
            f"Cannot connect to Ollama at {ollama_url}.\n"
            "  Fix: ollama serve"
        )


async def estimate(
    question: str,
    evidence: list[EvidenceBlock],
    temporal: Optional[TemporalSignal] = None,
    risk_score: Optional[RiskScore] = None,
    cluster_result: Optional[ClusterResult] = None,
) -> dict[str, Any]:
    """
    Call local Ollama (Llama 3.1 by default) and return 3-scenario probability estimate.
    Market probabilities are NEVER included in the prompt.
    """
    ollama_url = os.environ.get("OLLAMA_URL", _DEFAULT_OLLAMA_URL).rstrip("/")
    model = os.environ.get("OLLAMA_MODEL", _DEFAULT_MODEL)
    endpoint = f"{ollama_url}/api/chat"

    fields = _extract_evidence_fields(evidence)
    user_content = _build_prompt(
        question=question,
        temporal=temporal,
        risk_score=risk_score,
        cluster_result=cluster_result,
        **fields,
    )

    payload = {
        "model": model,
        "stream": False,
        "format": "json",
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ],
        "options": {
            "temperature": 0.2,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
            "num_predict": 900,   # more tokens for 3 scenarios + narratives
        },
    }

    logger.debug("llm_estimator: POST %s  model=%s", endpoint, model)

    async with aiohttp.ClientSession() as session:
        async with session.post(endpoint, json=payload, timeout=_TIMEOUT) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"Ollama HTTP {resp.status}: {body[:300]}")
            data = await resp.json()

    raw = data.get("message", {}).get("content", "") or data.get("response", "")
    logger.debug("llm_estimator: raw response: %s", raw[:400])

    return _parse_scenarios(raw)
