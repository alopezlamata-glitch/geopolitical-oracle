from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Optional

import aiohttp

from collector.base import EvidenceBlock

logger = logging.getLogger(__name__)

_DEFAULT_OLLAMA_URL = "http://localhost:11434"
_DEFAULT_MODEL = "llama3.1:8b"
# Generous timeout — 70b models on CPU can be slow
_TIMEOUT = aiohttp.ClientTimeout(total=180)

_SYSTEM_PROMPT = """\
You are a world-class superforecaster with deep expertise in geopolitics, \
economics, and calibrated probabilistic reasoning.

Your core principles:
- Always reason from base rates first, then update on new evidence
- Express genuine uncertainty — avoid round numbers like 50%, 30%, 20%
- Consider both the reference class (how often does this type of event happen?) \
and the specific case (what makes this situation unusual?)
- Weigh recent news heavily but discount media sensationalism
- When evidence is thin, widen your uncertainty — don't fake precision
- Never anchor to market prices (you are not shown any)

You MUST respond with ONLY a valid JSON object — no prose, no markdown fences, \
no explanation outside the JSON. Failure to produce valid JSON is unacceptable.\
"""

_USER_TEMPLATE = """\
QUESTION: {question}

═══ BACKGROUND CONTEXT (Wikipedia) ═══
{wikipedia}

═══ RECENT NEWS (last 72 hours) ═══
{news}

═══ MEDIA SENTIMENT (GDELT) ═══
{gdelt}

═══ YOUR TASK ═══
Carefully reason through:
1. BASE RATE — How often does this type of event occur historically?
2. CURRENT EVIDENCE — What do the news and context tell you about direction?
3. KEY UNCERTAINTIES — What could dramatically shift the probability?
4. FINAL ESTIMATE — A precise probability (NOT a round number).

Respond with ONLY this JSON object:
{{
  "probability": <float 0.05-0.95, never exactly 0.5, 0.3, 0.2, 0.1, etc.>,
  "abstain": <true ONLY if the question is unanswerable or nonsensical>,
  "reasoning": "<concise 3-4 sentence chain of thought explaining your estimate>",
  "confidence": "<low|medium|high>",
  "key_factors": ["<most important factor 1>", "<most important factor 2>", "<factor 3>"]
}}\
"""


def _build_prompt(
    question: str,
    wikipedia_content: str,
    rss_headlines: str,
    gdelt_tone: Optional[float],
    gdelt_count: Optional[int],
    gdelt_diversity: Optional[int],
) -> str:
    wiki = wikipedia_content.strip() or "(no Wikipedia background available)"
    news = rss_headlines.strip() or "(no recent news found in last 72h)"

    if gdelt_tone is not None:
        sentiment = "positive" if gdelt_tone > 2 else "negative" if gdelt_tone < -2 else "neutral"
        gdelt = (
            f"Tone: {gdelt_tone:+.2f} ({sentiment}) | "
            f"Articles: {gdelt_count} | "
            f"Distinct sources: {gdelt_diversity}"
        )
    else:
        gdelt = "(GDELT data unavailable)"

    return _USER_TEMPLATE.format(
        question=question,
        wikipedia=wiki,
        news=news,
        gdelt=gdelt,
    )


def _extract_evidence_fields(evidence: list[EvidenceBlock]) -> dict[str, Any]:
    """Extract prompt-relevant fields. Market probabilities are deliberately excluded."""
    fields: dict[str, Any] = {
        "wikipedia_content": "",
        "rss_headlines": "",
        "gdelt_tone": None,
        "gdelt_count": None,
        "gdelt_diversity": None,
    }
    for block in evidence:
        if block.source == "wikipedia" and block.content:
            fields["wikipedia_content"] = block.content
        elif block.source == "rss" and block.content and block.quality != "insufficient":
            fields["rss_headlines"] = block.content
        elif block.source == "gdelt" and block.quality != "insufficient":
            fields["gdelt_tone"] = block.metadata.get("avg_tone")
            fields["gdelt_count"] = block.metadata.get("article_count")
            fields["gdelt_diversity"] = block.metadata.get("source_diversity")
    return fields


def _parse_response(raw: str) -> dict[str, Any]:
    """
    Robust JSON extraction with three fallback layers.
    With Ollama format='json' this should almost never be needed,
    but we keep it as a safety net.
    """
    # Layer 1: direct parse (works when format=json is respected)
    try:
        return _validate(json.loads(raw))
    except (json.JSONDecodeError, ValueError):
        pass

    # Layer 2: extract first {...} block (handles leading/trailing prose)
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return _validate(json.loads(match.group(0)))
        except (json.JSONDecodeError, ValueError):
            pass

    # Layer 3: field-by-field regex (last resort)
    logger.warning("llm_estimator: falling back to regex field extraction")

    def grab(field: str) -> Optional[str]:
        m = re.search(rf'"{field}"\s*:\s*(.+?)(?:,\s*\n|\n|\}})', raw, re.DOTALL)
        return m.group(1).strip().strip('"') if m else None

    prob_raw = grab("probability")
    try:
        probability = float(prob_raw) if prob_raw and prob_raw.lower() != "null" else None
    except (TypeError, ValueError):
        probability = None

    abstain_raw = grab("abstain")
    abstain = (abstain_raw or "").lower() in ("true", "1") if abstain_raw else probability is None

    return _validate({
        "probability": probability,
        "abstain": abstain,
        "reasoning": grab("reasoning") or "Could not extract reasoning.",
        "confidence": grab("confidence") or "low",
        "key_factors": [],
    })


def _validate(result: dict) -> dict:
    """Type-check and range-validate the parsed result."""
    p = result.get("probability")
    if p is not None:
        try:
            p = float(p)
            if not (0.0 <= p <= 1.0):
                raise ValueError(f"out of range: {p}")
            result["probability"] = p
        except (TypeError, ValueError) as e:
            logger.warning("llm_estimator: invalid probability (%s) — abstaining", e)
            result["probability"] = None
            result["abstain"] = True

    result.setdefault("abstain", result.get("probability") is None)
    result.setdefault("reasoning", "")
    if result.get("confidence") not in ("low", "medium", "high"):
        result["confidence"] = "medium"
    if not isinstance(result.get("key_factors"), list):
        result["key_factors"] = []

    return result


async def check_ollama(ollama_url: str, model: str) -> None:
    """
    Verify Ollama is reachable and the requested model is available.
    Raises RuntimeError with a helpful message on failure.
    """
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


async def estimate(question: str, evidence: list[EvidenceBlock]) -> dict[str, Any]:
    """
    Call local Ollama (Llama 3.1 by default) with non-market evidence only.

    Environment variables:
        OLLAMA_URL    Ollama server base URL  (default: http://localhost:11434)
        OLLAMA_MODEL  Model tag               (default: llama3.1:8b)
    """
    ollama_url = os.environ.get("OLLAMA_URL", _DEFAULT_OLLAMA_URL).rstrip("/")
    model = os.environ.get("OLLAMA_MODEL", _DEFAULT_MODEL)
    endpoint = f"{ollama_url}/api/chat"

    fields = _extract_evidence_fields(evidence)
    user_content = _build_prompt(question, **fields)

    payload = {
        "model": model,
        "stream": False,
        "format": "json",       # forces valid JSON output — eliminates parse failures
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ],
        "options": {
            "temperature": 0.15,    # low temp = consistent structured output
            "top_p": 0.9,
            "repeat_penalty": 1.1,
            "num_predict": 600,
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
    logger.debug("llm_estimator: raw response: %s", raw[:300])

    return _parse_response(raw)
