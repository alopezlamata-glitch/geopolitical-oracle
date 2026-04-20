"""
Backfill world_state_history for all entities in the registry.

Problem: VAR models only exist for entities that have ≥14 rows in
world_state_history. Currently only Russia & Ukraine have them.
This script seeds synthetic history for all 35 entities so that
fit_transition_model.py can fit VAR for every country.

Method (synthetic random walk):
  1. Load current world state as anchor (or structural fallback if absent).
  2. Walk BACKWARDS n_weeks using a mean-reverting random walk:
       state[t-1] = 0.85 * state[t] + 0.15 * global_mean + noise
     where noise ~ N(0, 0.15 * feature_std).
  3. Insert each weekly snapshot into world_state_history with date
     (today - k*7 days).
  4. Optionally run fit_transition_model to fit VAR on the new history.

Assumptions marked with [ASSUMPTION]:
  [A1] Weekly resolution is sufficient for VAR fitting (VAR lag ≤ 7).
  [A2] Synthetic data is only used when no real history exists for the entity.
  [A3] Noise scale 15% of σ produces autocorrelation realistic enough for
       VAR identification; actual VAR forecasts will still be noisy.

Usage:
  python scripts/backfill_world_state.py
  python scripts/backfill_world_state.py --weeks 26 --no-fit
  python main.py backfill-world-state
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("backfill_ws")

_ROOT          = Path(__file__).parent.parent
_REGISTRY_PATH = _ROOT / "data" / "entity_registry.json"
_BASELINE_PATH = _ROOT / "data" / "model" / "feature_baseline.json"

# Features generated in synthetic history (subset of world_state columns)
_DYNAMIC_FEATS = [
    "military_count_7d", "military_count_30d",
    "protest_count_7d", "protest_count_30d",
    "diplomatic_count_7d", "ceasefire_count_7d", "sanction_count_7d",
    "military_intensity_7d", "protest_intensity_7d", "overall_intensity_7d",
    "military_accel", "protest_accel", "overall_accel",
    "avg_polarity_7d", "avg_polarity_30d", "tone_trend",
    "source_diversity_7d", "avg_independent_sources",
    "has_military_7d", "has_ceasefire_7d",
    "escalation_index", "ceasefire_ratio_7d", "event_velocity_7d", "military_share_7d",
]
_DOMAIN_FEATS = [
    "pol_resignation_signals", "pol_approval_pressure",
    "pol_coalition_stability", "pol_electoral_proximity", "pol_judicial_pressure",
    "eco_rate_change_prob", "eco_gdp_momentum", "eco_debt_stress",
    "eco_market_volatility", "eco_policy_uncertainty",
]
_STRUCTURAL_FEATS = [
    "country_conflict_baserate", "country_polity_norm", "country_mil_spending_norm",
]
_ALL_FEATS = _DYNAMIC_FEATS + _DOMAIN_FEATS + _STRUCTURAL_FEATS

# Clamp ranges per feature group
_CLAMPS: dict[str, tuple[float, float]] = {
    "military_count_7d":     (0, 500),
    "military_count_30d":    (0, 2000),
    "protest_count_7d":      (0, 300),
    "protest_count_30d":     (0, 1000),
    "diplomatic_count_7d":   (0, 100),
    "ceasefire_count_7d":    (0, 50),
    "sanction_count_7d":     (0, 50),
    "military_intensity_7d": (0, 20),
    "protest_intensity_7d":  (0, 10),
    "overall_intensity_7d":  (0, 30),
    "military_accel":        (0, 10),
    "protest_accel":         (0, 10),
    "overall_accel":         (0, 10),
    "avg_polarity_7d":       (-1, 1),
    "avg_polarity_30d":      (-1, 1),
    "tone_trend":            (-1, 1),
    "source_diversity_7d":   (0, 1),
    "avg_independent_sources": (1, 10),
    "has_military_7d":       (0, 1),
    "has_ceasefire_7d":      (0, 1),
    "escalation_index":      (-5, 20),
    "ceasefire_ratio_7d":    (0, 5),
    "event_velocity_7d":     (0, 5),
    "military_share_7d":     (0, 1),
    "pol_resignation_signals": (0, 1),
    "pol_approval_pressure":   (0, 1),
    "pol_coalition_stability": (0, 1),
    "pol_electoral_proximity": (0, 1),
    "pol_judicial_pressure":   (0, 1),
    "eco_rate_change_prob":    (0, 1),
    "eco_gdp_momentum":        (-1, 1),
    "eco_debt_stress":         (0, 1),
    "eco_market_volatility":   (0, 1),
    "eco_policy_uncertainty":  (0, 1),
    "country_conflict_baserate": (0, 1),
    "country_polity_norm":       (-1, 1),
    "country_mil_spending_norm": (0, 1),
}


def _entity_id(name: str) -> str:
    raw = f"country|{name.lower()}"
    return "ent_" + hashlib.sha256(raw.encode()).hexdigest()[:24]


def _history_id(entity_id: str, as_of_date: date) -> str:
    raw = f"backfill|{entity_id}|{as_of_date.isoformat()}"
    return "wsh_" + hashlib.sha256(raw.encode()).hexdigest()[:32]


def _load_registry() -> list[dict]:
    data = json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))
    return data.get("entities", [])


def _load_baseline() -> dict[str, dict]:
    """Returns {feature: {mean, std}}."""
    try:
        data = json.loads(_BASELINE_PATH.read_text(encoding="utf-8"))
        return {k: v for k, v in data.items() if isinstance(v, dict)}
    except Exception:
        return {}


def _global_means(baseline: dict[str, dict]) -> dict[str, float]:
    return {k: float(v.get("mean", 0.0)) for k, v in baseline.items()}


def _clamp(val: float, feat: str) -> float:
    lo, hi = _CLAMPS.get(feat, (-100, 100))
    return max(lo, min(hi, val))


def _already_has_history(entity_id: str, db) -> bool:
    """Return True if entity already has ≥14 rows in world_state_history."""
    try:
        cnt = db.execute(
            "SELECT COUNT(*) FROM world_state_history WHERE entity_id = ?",
            [entity_id],
        ).fetchone()[0]
        return cnt >= 14
    except Exception:
        return False


def _anchor_state(entity_name: str, structural: dict, global_means: dict) -> dict:
    """
    Load current world state as anchor, fall back to structural + global means.
    """
    try:
        from world_state.api import get_entity_state
        ws = get_entity_state(entity_name)
        if ws:
            return {f: float(ws.get(f, global_means.get(f, 0.0))) for f in _ALL_FEATS}
    except Exception:
        pass

    # Fallback: structural features + global means
    state = {f: global_means.get(f, 0.0) for f in _ALL_FEATS}
    for k, v in structural.items():
        if k in state:
            state[k] = float(v)
    return state


def _generate_history(
    anchor: dict,
    global_means: dict,
    feature_stds: dict[str, float],
    n_weeks: int,
    rng: random.Random,
) -> list[tuple[date, dict]]:
    """
    Generate n_weeks of weekly snapshots going backwards from today.
    Returns list of (date, feature_dict) newest-first.
    """
    today     = date.today()
    state     = dict(anchor)
    snapshots: list[tuple[date, dict]] = []

    for week in range(1, n_weeks + 1):
        snap_date = today - timedelta(weeks=week)

        # Mean-reverting random walk backwards
        # [ASSUMPTION A3]: 15% of σ noise per week
        for feat in _DYNAMIC_FEATS + _DOMAIN_FEATS:
            mean = global_means.get(feat, 0.0)
            std  = feature_stds.get(feat, 0.1)
            noise = rng.gauss(0, 0.15 * std)
            # Mean-revert toward global mean at rate 0.15
            state[feat] = 0.85 * state[feat] + 0.15 * mean + noise
            state[feat] = _clamp(state[feat], feat)

        # Structural features stay nearly constant (slow EMA)
        # (no noise, just track anchor)

        snapshots.append((snap_date, dict(state)))

    return snapshots   # newest first (week 1, 2, 3, …)


def backfill_entity(
    entity: dict,
    global_means: dict,
    feature_stds: dict[str, float],
    db,
    n_weeks: int,
    skip_existing: bool,
    dry_run: bool,
) -> int:
    """Backfill one entity. Returns number of rows inserted."""
    name = entity.get("name") or entity.get("canonical_name", "")
    eid  = _entity_id(name)

    if skip_existing and _already_has_history(eid, db):
        logger.info("  %s: already has ≥14 rows — skip", name)
        return 0

    # Structural features (country_data fallback)
    structural: dict = {}
    try:
        from features.country_data import get_country_features
        structural = get_country_features(name)
    except Exception:
        pass

    anchor = _anchor_state(name, structural, global_means)
    rng    = random.Random(hashlib.md5(name.encode()).hexdigest())
    snaps  = _generate_history(anchor, global_means, feature_stds, n_weeks, rng)

    if dry_run:
        logger.info("  [DRY-RUN] %s: would insert %d rows", name, len(snaps))
        return 0

    inserted = 0
    for snap_date, features in snaps:
        hid = _history_id(eid, snap_date)
        try:
            db.execute("""
                INSERT INTO world_state_history
                  (history_id, entity_id, as_of_date, computed_at,
                   features, sources_used, n_events_used,
                   data_completeness, updater_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (history_id) DO NOTHING
            """, [
                hid, eid, snap_date,
                datetime.now(timezone.utc),
                json.dumps(features),
                ["synthetic_backfill"],
                0,
                0.5,            # synthetic data = 50% completeness
                "backfill_v1",
            ])
            inserted += 1
        except Exception as e:
            logger.debug("  %s/%s insert failed: %s", name, snap_date, e)

    logger.info("  %s: inserted %d / %d rows", name, inserted, len(snaps))
    return inserted


def run(
    n_weeks: int = 26,
    skip_existing: bool = True,
    fit_after: bool = True,
    dry_run: bool = False,
    entity_names: Optional[list[str]] = None,
) -> dict:
    """
    Main entry point. Backfills all (or specified) entities.
    Returns {entity: rows_inserted}.
    """
    from data_layer.db import get_db, init_schema, table_exists

    init_schema()
    if not table_exists("world_state_history"):
        logger.error("world_state_history table missing — run init_schema first")
        return {}

    db       = get_db()
    baseline = _load_baseline()
    global_means  = _global_means(baseline)
    feature_stds  = {k: float(v.get("std", 0.1)) for k, v in baseline.items()}

    entities = _load_registry()
    if entity_names:
        lo = {n.lower() for n in entity_names}
        entities = [
            e for e in entities
            if (e.get("name") or e.get("canonical_name", "")).lower() in lo
        ]

    logger.info(
        "backfill: %d entities × %d weeks%s",
        len(entities), n_weeks,
        " [DRY-RUN]" if dry_run else "",
    )

    results: dict[str, int] = {}
    for ent in entities:
        name = ent.get("name") or ent.get("canonical_name", "")
        rows = backfill_entity(
            ent, global_means, feature_stds, db, n_weeks,
            skip_existing, dry_run,
        )
        results[name] = rows

    total = sum(results.values())
    logger.info("backfill complete: %d total rows inserted across %d entities", total, len(results))

    # Optionally refit VAR models so they pick up the new history
    if fit_after and not dry_run and total > 0:
        logger.info("fitting VAR models on new history…")
        try:
            from scripts.fit_transition_model import run as fit_run
            fit_run()
        except Exception as e:
            logger.warning("fit_transition_model failed (non-fatal): %s", e)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill world_state_history for all entities")
    parser.add_argument("--weeks",    type=int,  default=26, help="Weeks of history to generate")
    parser.add_argument("--no-skip",  action="store_true",   help="Re-insert even if history exists")
    parser.add_argument("--no-fit",   action="store_true",   help="Skip VAR refitting after insert")
    parser.add_argument("--dry-run",  action="store_true")
    parser.add_argument("--entity",   nargs="+",             help="Limit to these entities")
    args = parser.parse_args()

    results = run(
        n_weeks       = args.weeks,
        skip_existing = not args.no_skip,
        fit_after     = not args.no_fit,
        dry_run       = args.dry_run,
        entity_names  = args.entity,
    )
    print(f"\nBackfill complete: {sum(results.values())} rows across {len(results)} entities")


if __name__ == "__main__":
    main()
