"""
World model canonical query layer.

Single entry point for all world-state data. Features, predictors, and
coherence checks call these functions — never the underlying modules directly.

Functions
---------
get_entity_state(entity_name)       world_state row + staleness metadata
get_entity_relations(entity_name)   active relations (tenure, sanctions, …)
get_entity_trajectory(...)          VAR trajectory → marginalised probability
build_entity_context(entity_name)   merged dict used by features/builder.py

All functions are:
  - sync (no async)
  - non-fatal (return {} / None on any error)
  - cached at the reader's existing 5-min TTL
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Optional

logger = logging.getLogger(__name__)


# ── 1. Raw world state ────────────────────────────────────────────────────────

def get_entity_state(
    entity_name: str,
    as_of_date: Optional[date] = None,
) -> dict:
    """
    Return the most recent world_state row for entity_name, plus staleness.
    Returns {} if entity has no world state.
    """
    try:
        from world_state.reader import get_world_state
        ws = get_world_state(entity_name, as_of_date=as_of_date)
        return ws if ws is not None else {}
    except Exception as e:
        logger.debug("get_entity_state(%s) failed: %s", entity_name, e)
        return {}


# ── 2. Relations ──────────────────────────────────────────────────────────────

def get_entity_relations(entity_name: str) -> dict:
    """
    Return active relation features (leader_tenure_years, n_sanctions,
    has_active_conflict, has_formal_alliance, leader_under_investigation).
    Returns {} if relation data is unavailable.
    """
    try:
        from world_state.relation_reader import get_country_relation_features
        return get_country_relation_features(entity_name) or {}
    except Exception as e:
        logger.debug("get_entity_relations(%s) failed: %s", entity_name, e)
        return {}


# ── 3. Trajectory ─────────────────────────────────────────────────────────────

def get_entity_trajectory(
    entity_name: str,
    predicate: str,
    event_family: str,
    horizon_days: int,
    n_samples: int = 100,
) -> dict:
    """
    Return marginalised trajectory probability with uncertainty bounds.
    Returns {} if no VAR model exists for the entity.
    """
    try:
        from world_state.trajectory import marginalize_with_uncertainty
        result = marginalize_with_uncertainty(
            entity_name=entity_name,
            predicate=predicate,
            event_family=event_family,
            horizon_days=horizon_days,
            n_samples=n_samples,
        )
        return result or {}
    except Exception as e:
        logger.debug("get_entity_trajectory(%s) failed: %s", entity_name, e)
        return {}


# ── 4. Full context (used by features/builder.py) ────────────────────────────

def build_entity_context(entity_name: str) -> dict:
    """
    Merge world_state + relations + causal-neighbour context into one dict.

    Key namespaces in the returned dict:
      <feature>             world_state columns (military_count_7d, eco_*, …)
      <feature>             relation features (leader_tenure_years, n_sanctions, …)
      neighbor_<src>_<feat> causal-neighbour injected values
      _world_state_*        metadata (staleness, availability)

    Callers must NOT mutate the returned dict — make a copy if needed.
    Returns {} when the entity has no world state and no relations.
    """
    ctx: dict = {}

    # 1. World state (largest, most important block)
    ws = get_entity_state(entity_name)
    ctx.update(ws)

    # 2. Relation features (tenure, sanctions, etc.)
    # Only fill keys not already set by world state
    rel = get_entity_relations(entity_name)
    for k, v in rel.items():
        if k not in ctx:
            ctx[k] = v

    # 3. Causal-neighbour context
    try:
        from world_state.entity_graph import get_neighbor_context
        nb = get_neighbor_context(entity_name) or {}
        for k, v in nb.items():
            if not k.startswith("_"):
                ctx[k] = v
    except Exception as e:
        logger.debug("build_entity_context: neighbor_context failed: %s", e)

    staleness = ctx.get("_world_state_staleness_days", 99)
    if ctx:
        logger.debug(
            "build_entity_context(%s): %d keys, staleness=%s",
            entity_name, len(ctx), staleness,
        )

    return ctx


# ── Staleness-aware blend weight ──────────────────────────────────────────────

def world_state_weight(ctx: dict, default_if_stale: float = 0.3) -> float:
    """
    Return the appropriate world-state blend weight given staleness.

    staleness ≤ 1d  → 0.60  (fresh — trust the world model heavily)
    staleness ≤ 3d  → 0.40
    staleness ≤ 7d  → 0.25
    otherwise       → default_if_stale (0.30 legacy behaviour)

    Pass this to builder.py so the blend is centralised here.
    """
    staleness = ctx.get("_world_state_staleness_days")
    if staleness is None:
        return default_if_stale
    if staleness <= 1:
        return 0.60
    if staleness <= 3:
        return 0.40
    if staleness <= 7:
        return 0.25
    return default_if_stale
