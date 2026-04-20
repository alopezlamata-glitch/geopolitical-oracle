"""
Global geopolitical risk index.

Aggregates all available world states into:
  - global_instability: weighted average across all entities (0–1)
  - top_hotspots: top-5 highest-risk entities with driver features
  - risk_breakdown: {conflict, political, economic} component shares

Usage
-----
  from world_state.global_risk import compute_global_risk, format_global_risk
  risk = compute_global_risk()

CLI:
  python main.py global-risk
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_REGISTRY_PATH = Path(__file__).parent.parent / "data" / "entity_registry.json"

# Per-feature weight in conflict/political/economic risk index
_CONFLICT_FEATURES = {
    "escalation_index":      0.30,
    "military_intensity_7d": 0.25,
    "military_accel":        0.20,
    "military_share_7d":     0.15,
    "ceasefire_ratio_7d":   -0.10,   # de-escalation signal (negative contribution)
}
_POLITICAL_FEATURES = {
    "pol_approval_pressure":   0.35,
    "pol_resignation_signals": 0.30,
    "pol_coalition_stability": -0.20,  # higher stability = lower risk
    "pol_judicial_pressure":   0.15,
}
_ECONOMIC_FEATURES = {
    "eco_debt_stress":        0.35,
    "eco_market_volatility":  0.30,
    "eco_policy_uncertainty": 0.25,
    "eco_gdp_momentum":      -0.10,   # positive momentum = lower risk
}

# Entity priority weights (high priority entities count more)
_PRIORITY_WEIGHTS = {"high": 1.0, "medium": 0.6, "low": 0.3}

# Normalisation caps for raw feature values
_CAPS = {
    "escalation_index":      10.0,
    "military_intensity_7d": 5.0,
    "military_accel":        5.0,
    "military_share_7d":     1.0,
    "pol_approval_pressure": 1.0,
    "pol_resignation_signals": 1.0,
    "pol_coalition_stability": 1.0,
    "pol_judicial_pressure": 1.0,
    "eco_debt_stress":       1.0,
    "eco_market_volatility": 1.0,
    "eco_policy_uncertainty":1.0,
    "eco_gdp_momentum":      1.0,
    "ceasefire_ratio_7d":    1.0,
}


@dataclass
class EntityRisk:
    entity:        str
    priority:      str
    risk_score:    float          # 0–1 composite
    conflict_risk: float
    political_risk: float
    economic_risk: float
    top_driver:    str            # feature with highest contribution
    staleness_days: Optional[int]


@dataclass
class GlobalRisk:
    global_instability: float
    top_hotspots:       list[EntityRisk] = field(default_factory=list)
    risk_breakdown:     dict[str, float] = field(default_factory=dict)
    n_entities:         int = 0
    n_with_data:        int = 0


def _norm(val: float, cap: float) -> float:
    """Clip and normalise to [0, 1]."""
    return min(1.0, max(0.0, val) / cap) if cap > 0 else 0.0


def _score_domain(state: dict, feature_weights: dict) -> tuple[float, str]:
    """
    Compute domain risk score and top-driver feature.
    Returns (score, top_driver_feature_name).
    """
    total = 0.0
    contribs: dict[str, float] = {}
    for feat, w in feature_weights.items():
        raw = float(state.get(feat, 0.0))
        cap = _CAPS.get(feat, 1.0)
        normed = _norm(raw, cap)
        contrib = w * normed
        total += contrib
        contribs[feat] = abs(contrib)

    # Clamp to [0, 1]
    score = max(0.0, min(1.0, total))
    top_driver = max(contribs, key=contribs.get) if contribs else "unknown"
    return round(score, 4), top_driver


def _load_registry() -> list[dict]:
    try:
        data = json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))
        return data.get("entities", [])
    except Exception as e:
        logger.warning("global_risk: registry load failed: %s", e)
        return []


def compute_global_risk() -> GlobalRisk:
    """
    Load all entity world states, compute risk scores, return GlobalRisk.
    Non-fatal: entities with no world state are skipped.
    """
    from world_state.api import get_entity_state

    entities = _load_registry()
    entity_risks: list[EntityRisk] = []
    total_weight = 0.0
    weighted_sum = 0.0

    for ent in entities:
        name     = ent.get("name") or ent.get("canonical_name", "")
        priority = ent.get("priority", "medium")
        pw       = _PRIORITY_WEIGHTS.get(priority, 0.5)

        state = get_entity_state(name)
        if not state or not state.get("_world_state_available", False):
            continue

        staleness = state.get("_world_state_staleness_days")

        conflict_score, cf_driver   = _score_domain(state, _CONFLICT_FEATURES)
        political_score, pol_driver = _score_domain(state, _POLITICAL_FEATURES)
        economic_score, eco_driver  = _score_domain(state, _ECONOMIC_FEATURES)

        # Composite risk: conflict 50%, political 30%, economic 20%
        composite = (
            0.50 * conflict_score +
            0.30 * political_score +
            0.20 * economic_score
        )

        # Top driver = whichever domain is highest
        scores = {
            cf_driver:  conflict_score,
            pol_driver: political_score,
            eco_driver: economic_score,
        }
        top_driver = max(scores, key=scores.get)

        entity_risks.append(EntityRisk(
            entity         = name,
            priority       = priority,
            risk_score     = round(composite, 4),
            conflict_risk  = conflict_score,
            political_risk = political_score,
            economic_risk  = economic_score,
            top_driver     = top_driver,
            staleness_days = staleness,
        ))

        weighted_sum  += composite * pw
        total_weight  += pw

    entity_risks.sort(key=lambda e: e.risk_score, reverse=True)

    global_instability = (weighted_sum / total_weight) if total_weight > 0 else 0.0

    # Component breakdown across all entities (unweighted average)
    n = len(entity_risks)
    breakdown = {
        "conflict":  round(sum(e.conflict_risk  for e in entity_risks) / max(1, n), 4),
        "political": round(sum(e.political_risk for e in entity_risks) / max(1, n), 4),
        "economic":  round(sum(e.economic_risk  for e in entity_risks) / max(1, n), 4),
    }

    logger.info(
        "global_risk: instability=%.3f  entities=%d/%d  top=%s",
        global_instability, n, len(entities),
        entity_risks[0].entity if entity_risks else "none",
    )

    return GlobalRisk(
        global_instability = round(global_instability, 4),
        top_hotspots       = entity_risks[:10],
        risk_breakdown     = breakdown,
        n_entities         = len(entities),
        n_with_data        = n,
    )


def format_global_risk(risk: GlobalRisk) -> str:
    """CLI-printable risk report."""
    W = 66

    def row(s: str) -> str:
        return f"│ {s:<{W}} │"

    def sep() -> str:
        return "├─" + "─" * W + "─┤"

    inst = risk.global_instability
    bar  = "█" * int(inst * 30) + "░" * (30 - int(inst * 30))
    bd   = risk.risk_breakdown

    lines = [
        "┌─" + "─" * W + "─┐",
        row("  GLOBAL GEOPOLITICAL RISK INDEX"),
        row(f"  Instability : {inst:.3f}  {bar}"),
        row(f"  Breakdown   : conflict={bd.get('conflict',0):.3f}  "
            f"political={bd.get('political',0):.3f}  "
            f"economic={bd.get('economic',0):.3f}"),
        row(f"  Coverage    : {risk.n_with_data}/{risk.n_entities} entities with world state"),
        sep(),
        row(f"  {'Entity':<20} {'Risk':>6}  {'Conflict':>8}  {'Political':>9}  {'Economic':>8}  {'Driver'}"),
        sep(),
    ]

    for e in risk.top_hotspots[:10]:
        stale = f"[{e.staleness_days}d]" if e.staleness_days is not None else ""
        lines.append(row(
            f"  {e.entity:<20} {e.risk_score:>6.3f}  "
            f"{e.conflict_risk:>8.3f}  {e.political_risk:>9.3f}  "
            f"{e.economic_risk:>8.3f}  {e.top_driver[:14]}{stale}"
        ))

    lines.append("└─" + "─" * W + "─┘")
    return "\n".join(lines)


def run() -> GlobalRisk:
    risk = compute_global_risk()
    print(format_global_risk(risk))
    return risk
