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
_DEFAULT_MODEL = "tinyllama"
_TIMEOUT = aiohttp.ClientTimeout(total=180)

_SYSTEM_PROMPT = "You are a geopolitical superforecaster. Output ONLY valid JSON. No prose."

_USER_TEMPLATE = """\
QUESTION: {question}

CONTEXT (summary):
{context}

SIGNALS: risk={risk} | trend={trend} | tone={tone}

Output JSON with 3 scenarios. Weights must sum to 1.0. Be precise (no round numbers).
{{"scenarios":[{{"name":"<name>","weight":<0-1>,"probability_of_yes":<0.05-0.95>,"narrative":"<1 sentence>"}},{{"name":"<name>","weight":<0-1>,"probability_of_yes":<0.05-0.95>,"narrative":"<1 sentence>"}},{{"name":"<name>","weight":<0-1>,"probability_of_yes":<0.05-0.95>,"narrative":"<1 sentence>"}}],"confidence":"<low|medium|high>","key_factors":["<f1>","<f2>"],"abstain":false}}\
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
    # Compact context: first 400 chars of wikipedia + top 3 headlines
    wiki_snippet = (wikipedia_content.strip()[:400] + "...") if wikipedia_content.strip() else ""
    headlines = []
    if rss_headlines.strip():
        headlines = [l.strip() for l in rss_headlines.split("\n") if l.strip()][:3]
    context_parts = []
    if wiki_snippet:
        context_parts.append(wiki_snippet)
    if headlines:
        context_parts.append("News: " + " | ".join(h[:80] for h in headlines))
    if acled_content:
        context_parts.append(acled_content.strip()[:150])
    context = "\n".join(context_parts) or "(no context available)"

    risk_str = f"{risk_score.score:.0f}/100 [{risk_score.label}]" if risk_score else "N/A"
    trend_str = temporal.risk_label if temporal else "N/A"
    tone_str = f"{gdelt_tone:+.1f}" if gdelt_tone is not None else "N/A"

    return _USER_TEMPLATE.format(
        question=question,
        context=context,
        risk=risk_str,
        trend=trend_str,
        tone=tone_str,
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
            "num_predict": 350,
        },
    }

    logger.debug("llm_estimator: POST %s  model=%s", endpoint, model)

    # Use urllib for the LLM call — no session-level timeout that can interfere
    import json as _json
    import urllib.request as _urllib
    req_bytes = _json.dumps(payload).encode()
    req = _urllib.Request(endpoint, data=req_bytes, headers={"Content-Type": "application/json"})
    loop = asyncio.get_event_loop()

    def _do_request():
        with _urllib.urlopen(req, timeout=300) as r:
            return _json.loads(r.read().decode())

    data = await loop.run_in_executor(None, _do_request)
    raw = data.get("message", {}).get("content", "") or data.get("response", "")
    logger.debug("llm_estimator: raw response: %s", raw[:400])

    return _parse_scenarios(raw)
