"""
Entity graph — loads causal neighbors and injects their world states
as additional context features for the prediction of a target entity.

When predicting about Ukraine:
  - Russia is a causal neighbor (military_intensity → eco_market_volatility)
  - We inject Russia's current military_intensity as a feature
  - This gives the predictor cross-entity signal without the user asking

Feature injection convention:
  neighbor_{neighbor_name_slug}_{source_feature}
  e.g. "neighbor_russia_military_intensity_7d"

These features are NOT in _FEATURE_NAMES (v3 model never sees them).
They are passed to base_rate_predictor as additional log-odds weights
via the domain feature weight accumulation step.
"""
from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_CAUSAL_PATH = Path(__file__).parent.parent / "data" / "causal_links.json"


@lru_cache(maxsize=1)
def _load_links() -> list[dict]:
    if not _CAUSAL_PATH.exists():
        return []
    try:
        return json.loads(_CAUSAL_PATH.read_text()).get("links", [])
    except Exception:
        return []


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def get_causal_neighbors(entity_name: str) -> list[dict]:
    """
    Return all causal links where entity_name is the target.
    Each dict: {source_entity, source_feature, target_feature, weight, decay_days}
    """
    links = _load_links()
    return [l for l in links if l.get("target_entity", "").lower() == entity_name.lower()]


def get_neighbor_context(entity_name: str) -> dict[str, float]:
    """
    For each causal neighbor of entity_name, load their current world state
    and return the relevant source features as prefixed feature names.

    Returns empty dict if no world state exists for neighbors.
    """
    neighbors = get_causal_neighbors(entity_name)
    if not neighbors:
        return {}

    from world_state.reader import get_world_state

    context: dict[str, float] = {}
    seen_entities: set[str] = set()

    for link in neighbors:
        source = link["source_entity"]
        source_feat = link["source_feature"]
        weight = float(link.get("weight", 0.0))

        if source in seen_entities:
            # State already loaded — just grab the feature
            key = f"neighbor_{_slug(source)}_{source_feat}"
            if key not in context:
                state = get_world_state(source)
                if state:
                    val = state.get(source_feat)
                    if val is not None:
                        context[key] = round(float(val), 6)
            continue

        state = get_world_state(source)
        seen_entities.add(source)

        if state is None:
            logger.debug("entity_graph: no world state for neighbor %s", source)
            continue

        val = state.get(source_feat)
        if val is None:
            continue

        key = f"neighbor_{_slug(source)}_{source_feat}"
        context[key] = round(float(val), 6)

        # Also store weighted contribution as a direct log-odds delta hint
        # (used by coherence layer, not by XGBoost)
        context[f"_causal_{key}_weight"] = weight

        staleness = state.get("_world_state_staleness_days", 99)
        if staleness > 2:
            logger.debug(
                "entity_graph: %s state is %d days stale (source for %s)",
                source, staleness, entity_name,
            )

    if context:
        n = sum(1 for k in context if not k.startswith("_"))
        logger.info(
            "entity_graph: %d neighbor features for %s (from %d causal links)",
            n, entity_name, len(neighbors),
        )

    return context


def get_all_neighbors_states(entity_name: str) -> dict[str, Optional[dict]]:
    """
    Return {neighbor_name: world_state_dict} for all causal neighbors.
    Used by coherence checker to verify cross-entity consistency.
    """
    neighbors = get_causal_neighbors(entity_name)
    sources = list({l["source_entity"] for l in neighbors})

    from world_state.reader import get_world_state
    return {src: get_world_state(src) for src in sources}


def get_causal_implications(
    entity_name: str,
    predicate: str,
    probability: float,
) -> list[dict]:
    """
    Given P(predicate | entity) = probability, return implied probability
    bounds for downstream entities based on causal links going OUT from entity.

    Used to generate consistency warnings.
    Returns list of {target_entity, target_feature, implied_delta, link_id}
    """
    links = _load_links()
    outbound = [l for l in links if l.get("source_entity", "").lower() == entity_name.lower()]

    implications = []
    for link in outbound:
        weight = float(link.get("weight", 0.0))
        # High probability of the source feature → strong causal pressure
        # This is a directional hint, not a hard constraint
        implied_delta = weight * probability
        implications.append({
            "target_entity":  link["target_entity"],
            "target_feature": link["target_feature"],
            "implied_delta":  round(implied_delta, 4),
            "link_id":        link.get("link_id", "?"),
            "direction":      link.get("direction", "positive"),
        })

    return implications
