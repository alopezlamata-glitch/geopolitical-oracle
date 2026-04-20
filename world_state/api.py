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

class EntityContext(dict):
    """
    Flat dict of entity features with typed namespace metadata.

    Fully backward-compatible with all dict operations (get, items, update,
    iteration, ``isinstance(ctx, dict)``). New code that needs to distinguish
    world-state features from relation features from causal-neighbour context
    should use ``ctx.namespaces`` instead of prefix-checking on keys.

    Namespace keys in ``ctx.namespaces``:
      "world_state"  — event-derived EMA features from the world_state DB table
      "relations"    — static/temporal relation features (leader_tenure_years, …)
      "neighbors"    — causal-neighbour injected values (neighbor_* prefixed)
      "metadata"     — internal metadata (_world_state_* prefixed keys)
    """
    __slots__ = ("namespaces",)

    def __init__(self, flat: dict, namespaces: "dict[str, dict]"):
        super().__init__(flat)
        self.namespaces = namespaces


def build_entity_context(entity_name: str) -> EntityContext:
    """
    Merge world_state + relations + causal-neighbour context into one EntityContext.

    Returns an EntityContext (dict subclass) with the same flat key layout as
    before — all existing callers are unaffected.  New code can access typed
    namespaces via ``ctx.namespaces["world_state"]`` etc.

    Returns an empty EntityContext when the entity has no world state and no relations.
    """
    ns_world_state: dict = {}
    ns_relations:   dict = {}
    ns_neighbors:   dict = {}
    ns_metadata:    dict = {}

    # 1. World state — split into feature keys and _metadata_ keys
    ws = get_entity_state(entity_name)
    for k, v in ws.items():
        if k.startswith("_"):
            ns_metadata[k] = v
        else:
            ns_world_state[k] = v

    # 2. Relation features — only keys not already covered by world state
    rel = get_entity_relations(entity_name)
    for k, v in rel.items():
        if k not in ns_world_state:
            ns_relations[k] = v

    # 3. Causal-neighbour context
    try:
        from world_state.entity_graph import get_neighbor_context
        nb = get_neighbor_context(entity_name) or {}
        for k, v in nb.items():
            if not k.startswith("_"):
                ns_neighbors[k] = v
    except Exception as e:
        logger.debug("build_entity_context: neighbor_context failed: %s", e)

    # Build flat dict (merge priority: world_state > metadata > relations > neighbors)
    flat: dict = {}
    flat.update(ns_world_state)
    flat.update(ns_metadata)
    for k, v in ns_relations.items():
        if k not in flat:
            flat[k] = v
    flat.update(ns_neighbors)

    namespaces = {
        "world_state": ns_world_state,
        "relations":   ns_relations,
        "neighbors":   ns_neighbors,
        "metadata":    ns_metadata,
    }

    staleness = flat.get("_world_state_staleness_days", 99)
    if flat:
        logger.debug(
            "build_entity_context(%s): %d keys (ws=%d rel=%d nb=%d meta=%d) staleness=%s",
            entity_name, len(flat),
            len(ns_world_state), len(ns_relations), len(ns_neighbors), len(ns_metadata),
            staleness,
        )

    return EntityContext(flat, namespaces)


# ── Staleness-aware blend weight ──────────────────────────────────────────────

def world_state_weight(ctx: dict, default_if_stale: float = 0.3) -> float:
    """
    Return the appropriate world-state blend weight given staleness AND coverage density.

    Staleness tiers (base weight):
      staleness ≤ 1d  → 0.60
      staleness ≤ 3d  → 0.40
      staleness ≤ 7d  → 0.25
      otherwise       → default_if_stale

    Density multiplier (applied multiplicatively to the staleness base):
      density_factor = clamp(sqrt(n_events_used / 30.0) * data_completeness, 0.30, 1.0)

    A 1d-fresh snapshot built from 3 events at 50% completeness receives
    0.60 × clamp(sqrt(0.1)×0.50) ≈ 0.60 × 0.30 = 0.18 instead of 0.60.
    A 1d-fresh snapshot with 300 events at 100% completeness keeps its 0.60.

    When n_events_used or data_completeness are absent from ctx, the density
    block is skipped and staleness-only behaviour is preserved exactly.

    Final weight is clamped to [0.10, 0.60].
    """
    staleness = ctx.get("_world_state_staleness_days")
    if staleness is None:
        return default_if_stale

    # Staleness tier
    if staleness <= 1:
        base = 0.60
    elif staleness <= 3:
        base = 0.40
    elif staleness <= 7:
        base = 0.25
    else:
        base = default_if_stale

    # Density multiplier
    n_events     = ctx.get("n_events_used")
    completeness = ctx.get("data_completeness")
    if n_events is not None and completeness is not None:
        import math
        raw_density    = math.sqrt(max(0.0, float(n_events)) / 30.0) * float(completeness)
        density_factor = max(0.30, min(1.0, raw_density))
        base           = max(0.10, min(0.60, base * density_factor))

    return base
