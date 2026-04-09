from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any, Optional

import anthropic

from collector.base import EvidenceBlock

logger = logging.getLogger(__name__)

_MODEL = "claude-sonnet-4-20250514"
_MAX_TOKENS = 800

_SYSTEM_PROMPT = (
    "You are a superforecaster trained in calibrated probabilistic reasoning. "
    "You reason carefully about base rates, current evidence, and historical analogies. "
    "You never anchor to market prices. You always express genuine uncertainty and avoid "
    "round numbers (never say exactly 50%, 30%, etc.). "
    "Output ONLY valid JSON — no prose, no markdown, no explanation outside the JSON object."
)


def _build_user_prompt(
    question: str,
    wikipedia_content: str,
    rss_headlines: str,
    gdelt_tone: Optional[float],
    gdelt_count: Optional[int],
    gdelt_diversity: Optional[int],
) -> str:
    gdelt_section = ""
    if gdelt_tone is not None:
        gdelt_section = f"\nMedia sentiment signal: tone={gdelt_tone:.2f}, articles={gdelt_count}, sources={gdelt_diversity}"

    wiki_section = wikipedia_content.strip() if wikipedia_content.strip() else "(no Wikipedia background found)"
    rss_section = rss_headlines.strip() if rss_headlines.strip() else "(no recent news found)"

    return (
        f"Question: {question}\n\n"
        f"Background context (Wikipedia):\n{wiki_section}\n\n"
        f"Recent news headlines (last 72h):\n{rss_section}"
        f"{gdelt_section}\n\n"
        "Analyze this evidence carefully. Consider:\n"
        "1. What is the base rate for this type of event historically?\n"
        "2. What does the current evidence suggest about direction and magnitude?\n"
        "3. What would significantly change your estimate?\n\n"
        "Respond ONLY with this JSON (no other text):\n"
        "{\n"
        '  "probability": <float between 0.05 and 0.95, or null if abstaining>,\n'
        '  "abstain": <true if evidence is too thin to forecast, otherwise false>,\n'
        '  "reasoning": "<3-4 sentence chain of thought, max 300 chars>",\n'
        '  "confidence": "<low|medium|high>",\n'
        '  "key_factors": ["<factor1>", "<factor2>"]\n'
        "}"
    )


def _extract_evidence_fields(evidence: list[EvidenceBlock]) -> dict[str, Any]:
    """Pull relevant fields from evidence blocks for prompt construction."""
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
        elif block.source == "rss" and block.content:
            fields["rss_headlines"] = block.content
        elif block.source == "gdelt" and block.quality != "insufficient":
            fields["gdelt_tone"] = block.metadata.get("avg_tone")
            fields["gdelt_count"] = block.metadata.get("article_count")
            fields["gdelt_diversity"] = block.metadata.get("source_diversity")
    return fields


def _parse_llm_response(raw: str) -> dict[str, Any]:
    """
    Three-layer JSON extraction defense:
    1. Regex extract {...} block
    2. json.loads
    3. Field-by-field regex fallback
    """
    # Layer 1: extract JSON object
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    candidate = match.group(0) if match else raw

    # Layer 2: full JSON parse
    try:
        result = json.loads(candidate)
        return _validate_result(result)
    except (json.JSONDecodeError, ValueError):
        logger.debug("llm_estimator: full JSON parse failed, trying field extraction")

    # Layer 3: field-by-field extraction
    def extract_field(text: str, field: str) -> Optional[str]:
        pattern = rf'"{field}"\s*:\s*(.+?)(?:,\s*\n|\n|\}})'
        m = re.search(pattern, text, re.DOTALL)
        return m.group(1).strip().strip('"') if m else None

    prob_raw = extract_field(raw, "probability")
    abstain_raw = extract_field(raw, "abstain")
    reasoning_raw = extract_field(raw, "reasoning")
    confidence_raw = extract_field(raw, "confidence")

    try:
        probability = float(prob_raw) if prob_raw and prob_raw.lower() != "null" else None
    except (TypeError, ValueError):
        probability = None

    abstain = abstain_raw.lower() in ("true", "1") if abstain_raw else probability is None

    return {
        "probability": probability,
        "abstain": abstain,
        "reasoning": reasoning_raw or "Could not extract reasoning.",
        "confidence": confidence_raw or "low",
        "key_factors": [],
    }


def _validate_result(result: dict) -> dict:
    """Validate types and ranges from parsed JSON."""
    p = result.get("probability")
    if p is not None:
        try:
            p = float(p)
            if not (0.0 <= p <= 1.0):
                raise ValueError(f"probability out of range: {p}")
            result["probability"] = p
        except (TypeError, ValueError) as e:
            logger.warning("llm_estimator: %s — setting abstain=True", e)
            result["probability"] = None
            result["abstain"] = True

    if "abstain" not in result:
        result["abstain"] = result.get("probability") is None

    if "reasoning" not in result:
        result["reasoning"] = ""

    if "confidence" not in result or result["confidence"] not in ("low", "medium", "high"):
        result["confidence"] = "medium"

    if not isinstance(result.get("key_factors"), list):
        result["key_factors"] = []

    return result


def _blocking_claude_call(client: anthropic.Anthropic, system: str, user: str) -> str:
    """Synchronous Claude API call — meant to be run in an executor."""
    message = client.messages.create(
        model=_MODEL,
        max_tokens=_MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return message.content[0].text


async def estimate(question: str, evidence: list[EvidenceBlock]) -> dict[str, Any]:
    """
    Call Claude with only non-market evidence and return a probability estimate.
    Note: Metaculus and Polymarket probabilities are NEVER included in the prompt.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic(api_key=api_key)
    fields = _extract_evidence_fields(evidence)
    user_prompt = _build_user_prompt(question, **fields)

    logger.debug("llm_estimator: calling Claude (%s)", _MODEL)
    loop = asyncio.get_event_loop()
    raw_response = await loop.run_in_executor(
        None, _blocking_claude_call, client, _SYSTEM_PROMPT, user_prompt
    )

    logger.debug("llm_estimator: raw response: %s", raw_response[:200])
    result = _parse_llm_response(raw_response)
    return result
