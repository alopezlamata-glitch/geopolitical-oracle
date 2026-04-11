"""
LLM-based geopolitical feature extractor.

Uses Ollama (local LLM) to extract 6 structured features from event headlines.
These features enrich the prediction pipeline beyond keyword counts:

  llm_threat_level      — how dangerous/threatening is the situation (0=safe, 1=active war)
  llm_escalation        — is the situation escalating (0=calming, 1=escalating)
  llm_deescalation      — are there peace/resolution signals (0=none, 1=strong)
  llm_event_certainty   — how confirmed/verified are the events (0=rumors, 1=confirmed)
  llm_actor_hostility   — inter-actor hostility level (0=cooperative, 1=hostile)
  llm_available         — 1.0 if Ollama was reachable, 0.0 otherwise (model flag)

Design:
  - Single prompt per prediction call (not per event) — keeps latency manageable
  - Up to MAX_HEADLINES most-recent headlines sent to the model
  - JSON format enforced at API level (Ollama format="json")
  - All values clipped to [0.0, 1.0] after extraction
  - Returns zero-filled LLMFeatures on any failure (non-fatal)

The v3 XGBoost model ignores these features (its feature_names list doesn't
include them). They are stored in feature_snapshots.explicit_features for
future v4 model training.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional

from llm.client import OllamaClient, get_client

logger = logging.getLogger(__name__)

MAX_HEADLINES = 25        # max events sent to LLM per call
MAX_HEADLINE_LEN = 120    # characters per headline (keep prompt short)

# Feature names that will be added to the feature dict
LLM_FEATURE_NAMES = [
    "llm_threat_level",
    "llm_escalation",
    "llm_deescalation",
    "llm_event_certainty",
    "llm_actor_hostility",
    "llm_available",
]

_SYSTEM_PROMPT = (
    "You are a geopolitical analyst. "
    "Extract numerical risk indicators from news headlines. "
    "Output only valid JSON with float values between 0.0 and 1.0. "
    "Do not explain. Do not add keys other than those requested."
)

_PROMPT_TEMPLATE = """\
Question to forecast: {question}

Recent geopolitical news ({n} headlines):
{headlines}

Rate each indicator from 0.0 to 1.0 based on the news:
- threat_level: How dangerous/threatening is the situation? (0=peaceful, 1=active armed conflict)
- escalation: Is the situation escalating? (0=calming down, 1=rapidly escalating)
- deescalation: Are there peace or resolution signals? (0=none visible, 1=strong signals)
- event_certainty: How confirmed/verified are these events? (0=rumors/unverified, 1=officially confirmed)
- actor_hostility: How hostile are the main actors toward each other? (0=cooperative, 1=openly hostile)

Respond with JSON only:
{{"threat_level": 0.0, "escalation": 0.0, "deescalation": 0.0, "event_certainty": 0.0, "actor_hostility": 0.0}}"""


@dataclass
class LLMFeatures:
    """
    Structured output from the LLM feature extractor.

    All floats are clipped to [0.0, 1.0].
    available=False means Ollama was not reachable or the model failed;
    all numeric fields will be 0.0 in that case.
    """
    threat_level: float = 0.0
    escalation: float = 0.0
    deescalation: float = 0.0
    event_certainty: float = 0.0
    actor_hostility: float = 0.0
    available: bool = False
    latency_ms: Optional[float] = None
    model_used: Optional[str] = None

    def to_feature_dict(self) -> dict[str, float]:
        """Convert to the flat feature dict expected by features/builder.py."""
        return {
            "llm_threat_level":    self.threat_level,
            "llm_escalation":      self.escalation,
            "llm_deescalation":    self.deescalation,
            "llm_event_certainty": self.event_certainty,
            "llm_actor_hostility": self.actor_hostility,
            "llm_available":       1.0 if self.available else 0.0,
        }


def _clip(v) -> float:
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return 0.0


def _build_prompt(question: str, headlines: list[str]) -> str:
    truncated = [h[:MAX_HEADLINE_LEN] for h in headlines[:MAX_HEADLINES]]
    joined = "\n".join(f"- {h}" for h in truncated)
    return _PROMPT_TEMPLATE.format(
        question=question[:200],
        n=len(truncated),
        headlines=joined,
    )


def _parse_response(data: Optional[dict]) -> Optional[LLMFeatures]:
    """Extract and validate LLMFeatures from the raw JSON dict."""
    if not isinstance(data, dict):
        return None

    # Accept both bare keys and nested under various wrapper keys
    src = data
    for wrapper in ("features", "indicators", "result", "analysis"):
        if wrapper in data and isinstance(data[wrapper], dict):
            src = data[wrapper]
            break

    try:
        return LLMFeatures(
            threat_level    = _clip(src.get("threat_level",    src.get("threat", 0.0))),
            escalation      = _clip(src.get("escalation",      src.get("escalation_level", 0.0))),
            deescalation    = _clip(src.get("deescalation",    src.get("de_escalation", 0.0))),
            event_certainty = _clip(src.get("event_certainty", src.get("certainty", 0.0))),
            actor_hostility = _clip(src.get("actor_hostility", src.get("hostility", 0.0))),
            available=True,
        )
    except Exception as e:
        logger.warning("llm text_features: parse error: %s | raw=%s", e, str(data)[:200])
        return None


async def _extract_async(
    question: str,
    headlines: list[str],
    client: OllamaClient,
) -> LLMFeatures:
    """Core async extraction — called from the sync wrapper."""
    import time
    t0 = time.monotonic()

    prompt = _build_prompt(question, headlines)
    raw = await client.generate_json(
        prompt=prompt,
        system=_SYSTEM_PROMPT,
        temperature=0.0,
    )
    latency_ms = (time.monotonic() - t0) * 1000

    parsed = _parse_response(raw)
    if parsed is None:
        logger.warning(
            "llm text_features: no valid response (latency=%.0fms, raw=%s)",
            latency_ms, str(raw)[:100],
        )
        return LLMFeatures(available=False)

    parsed.latency_ms = round(latency_ms, 1)
    parsed.model_used = client.text_model
    logger.info(
        "llm text_features: threat=%.2f esc=%.2f deesc=%.2f cert=%.2f host=%.2f "
        "(%.0fms, model=%s)",
        parsed.threat_level, parsed.escalation, parsed.deescalation,
        parsed.event_certainty, parsed.actor_hostility,
        latency_ms, client.text_model,
    )
    return parsed


def extract_llm_features(
    question: str,
    headlines: list[str],
    client: Optional[OllamaClient] = None,
) -> LLMFeatures:
    """
    Synchronous wrapper — can be called from non-async code (e.g. features/builder.py).

    Returns zero-filled LLMFeatures on any failure, including Ollama being
    unavailable. Never raises.

    Args:
        question  : the binary question being forecast
        headlines : list of event titles/descriptions (most recent first)
        client    : OllamaClient instance; uses module singleton if None
    """
    if not headlines:
        return LLMFeatures(available=False)

    if client is None:
        client = get_client()

    try:
        # If we're already in an event loop (e.g. called from an async context)
        # we need to be careful not to call asyncio.run() inside a running loop.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            # We're inside an async context — schedule as a task and return zero
            # (this shouldn't happen in practice since builder.py is sync)
            logger.warning(
                "llm text_features: called from running event loop — skipping "
                "(use await _extract_async() directly in async contexts)"
            )
            return LLMFeatures(available=False)

        return asyncio.run(_extract_async(question, headlines, client))
    except Exception as e:
        logger.warning("llm text_features: extraction failed: %s", e)
        return LLMFeatures(available=False)
