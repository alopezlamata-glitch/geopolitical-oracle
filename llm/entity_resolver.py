"""
LLM-based entity resolution using Ollama.

Converts raw entity mentions ("Putin", "el Kremlin", "Russian forces",
"Vladimir Putin", "IRGC") to a canonical form with type classification.

This populates the `entities` and `entity_aliases` tables in the lakehouse,
enabling the temporal knowledge graph and richer actor-level features.

Integration point: called from data_layer/pipeline_hooks.py after
canonical events are inserted, using the actor/target mentions extracted
by the normalizer.

Design:
  - One LLM call per unique mention (not per event)
  - Results cached in memory per process (same mention → same answer)
  - Returns EntityCanon(canonical="", type="unknown") on failure
  - Non-fatal: a resolver failure never blocks pipeline writes

Entity types match the schema:
  'person' | 'country' | 'organization' | 'location' | 'group' | 'other'
"""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Optional

from llm.client import OllamaClient, get_client

logger = logging.getLogger(__name__)

# In-process cache: mention → EntityCanon
_resolution_cache: dict[str, "EntityCanon"] = {}

VALID_TYPES = {"person", "country", "organization", "location", "group", "other"}

_SYSTEM_PROMPT = (
    "You are an entity resolution system. "
    "Given a text mention of an entity, output its canonical name and type. "
    "Output only valid JSON. No explanation."
)

_PROMPT_TEMPLATE = """\
Entity mention: "{mention}"
Country/region context: "{context}"

What is the canonical name and type for this entity?
Types: person, country, organization, location, group, other

Examples:
- "Putin" → {{"canonical": "Vladimir Putin", "type": "person"}}
- "el Kremlin" → {{"canonical": "Russia", "type": "country"}}
- "IRGC" → {{"canonical": "Islamic Revolutionary Guard Corps", "type": "organization"}}
- "Russian forces" → {{"canonical": "Russia Armed Forces", "type": "organization"}}
- "Washington" → {{"canonical": "United States", "type": "country"}}

JSON only: {{"canonical": "...", "type": "..."}}"""


@dataclass
class EntityCanon:
    """Result of entity resolution."""
    canonical: str           # canonical entity name
    entity_type: str         # one of VALID_TYPES
    mention: str             # original mention
    confidence: float = 1.0  # 0-1, lower when LLM unavailable
    from_cache: bool = False
    resolved_by: str = "llm" # "llm" | "fallback"


def _parse_resolution(data: Optional[dict], mention: str) -> Optional[EntityCanon]:
    if not isinstance(data, dict):
        return None
    try:
        canonical = str(data.get("canonical", "")).strip()
        entity_type = str(data.get("type", "other")).lower().strip()

        if not canonical:
            return None
        if entity_type not in VALID_TYPES:
            entity_type = "other"

        return EntityCanon(
            canonical=canonical,
            entity_type=entity_type,
            mention=mention,
            confidence=0.9,
            resolved_by="llm",
        )
    except Exception as e:
        logger.debug("entity_resolver: parse error: %s", e)
        return None


def _fallback_resolution(mention: str) -> EntityCanon:
    """
    Simple rule-based fallback when Ollama is unavailable.

    Uses title-case as canonical form, guesses type from length/patterns.
    """
    canonical = re.sub(r"\s+", " ", mention).strip().title()
    # Heuristic type guessing
    if re.match(r"^[A-Z][a-z]+ [A-Z][a-z]+$", mention):
        entity_type = "person"
    elif mention.isupper() and len(mention) <= 6:
        entity_type = "organization"
    else:
        entity_type = "other"

    return EntityCanon(
        canonical=canonical or mention,
        entity_type=entity_type,
        mention=mention,
        confidence=0.3,
        resolved_by="fallback",
    )


async def _resolve_async(
    mention: str,
    context: str,
    client: OllamaClient,
) -> EntityCanon:
    prompt = _PROMPT_TEMPLATE.format(
        mention=mention[:100],
        context=context[:60],
    )
    raw = await client.generate_json(
        prompt=prompt,
        system=_SYSTEM_PROMPT,
        temperature=0.0,
    )
    parsed = _parse_resolution(raw, mention)
    if parsed is None:
        logger.debug("entity_resolver: LLM failed for '%s', using fallback", mention)
        return _fallback_resolution(mention)
    return parsed


def resolve_entity(
    mention: str,
    context: str = "",
    client: Optional[OllamaClient] = None,
    use_cache: bool = True,
) -> EntityCanon:
    """
    Resolve a raw entity mention to its canonical form.

    Synchronous wrapper, safe to call from non-async code.
    Results are cached in-process to avoid repeated LLM calls for the same mention.

    Args:
        mention    : raw entity mention from news text
        context    : country/region context for disambiguation
        client     : OllamaClient; uses module singleton if None
        use_cache  : whether to use/update the in-process cache

    Returns:
        EntityCanon with canonical name and type.
        Falls back to rule-based resolution if LLM fails.
    """
    mention = mention.strip()
    if not mention:
        return EntityCanon(canonical="", entity_type="other", mention="",
                           confidence=0.0, resolved_by="fallback")

    cache_key = f"{mention.lower()}|{context.lower()}"
    if use_cache and cache_key in _resolution_cache:
        cached = _resolution_cache[cache_key]
        return EntityCanon(
            canonical=cached.canonical,
            entity_type=cached.entity_type,
            mention=mention,
            confidence=cached.confidence,
            from_cache=True,
            resolved_by=cached.resolved_by,
        )

    if client is None:
        client = get_client()

    try:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            # Inside async context — use fallback to avoid nested loop issues
            result = _fallback_resolution(mention)
        else:
            result = asyncio.run(_resolve_async(mention, context, client))
    except Exception as e:
        logger.warning("entity_resolver: error for '%s': %s", mention, e)
        result = _fallback_resolution(mention)

    if use_cache:
        _resolution_cache[cache_key] = result

    return result


def resolve_entities_batch(
    mentions: list[str],
    context: str = "",
    client: Optional[OllamaClient] = None,
) -> list[EntityCanon]:
    """
    Resolve a list of entity mentions. Uses cache to avoid redundant LLM calls.

    Args:
        mentions : list of raw entity mentions
        context  : shared country/region context
        client   : OllamaClient; uses module singleton if None

    Returns list of EntityCanon in the same order as `mentions`.
    """
    return [resolve_entity(m, context=context, client=client) for m in mentions]


def clear_cache() -> None:
    """Clear the in-process entity resolution cache. Useful in tests."""
    _resolution_cache.clear()
