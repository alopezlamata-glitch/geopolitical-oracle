"""
Shock detector — identifies structural breaks in an entity's world state.

Compares the current world state snapshot against its own 30-day rolling
history.  Features that deviate by ≥ 2.5σ are flagged as shocks.

Severity tiers:
  |z| >= 3.5  → severe   (CI × 2.5)
  |z| >= 2.5  → moderate (CI × 1.8)
  |z| >= 1.8  → mild     (CI × 1.3)

Usage:
  from world_state.shock_detector import detect_shocks, get_shock_multiplier
  shocks = detect_shocks("Ukraine")
  multiplier = get_shock_multiplier(shocks)  # used by trajectory.py
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

_Z_MILD     = 1.8
_Z_MODERATE = 2.5
_Z_SEVERE   = 3.5

_CI_MULTIPLIERS = {
    "severe":   2.5,
    "moderate": 1.8,
    "mild":     1.3,
}

# Features to monitor (subset of world_state columns most predictive of shocks)
_MONITORED = [
    "military_count_7d",
    "escalation_index",
    "military_intensity_7d",
    "event_velocity_7d",
    "military_accel",
    "avg_polarity_7d",
    "pol_resignation_signals",
    "pol_approval_pressure",
    "pol_coalition_stability",
    "eco_market_volatility",
    "eco_debt_stress",
    "eco_policy_uncertainty",
    "fred_vix",
]


@dataclass
class ShockSignal:
    feature:       str
    current_value: float
    rolling_mean:  float
    rolling_std:   float
    z_score:       float
    severity:      str   # 'mild' | 'moderate' | 'severe'
    direction:     str   # '+' | '-'


def detect_shocks(
    entity_name: str,
    window_days: int = 30,
) -> dict[str, ShockSignal]:
    """
    Compare current world state against the last `window_days` of history.
    Returns {feature: ShockSignal} for every feature with |z| >= _Z_MILD.
    Returns empty dict on any error (non-fatal, graceful degradation).
    """
    history = _load_history(entity_name, window_days)
    if not history:
        return {}

    from world_state.reader import get_world_state
    current = get_world_state(entity_name)
    if current is None:
        return {}

    try:
        import numpy as np
    except ImportError:
        return {}

    shocks: dict[str, ShockSignal] = {}

    for feat in _MONITORED:
        current_val = current.get(feat)
        if current_val is None:
            continue

        vals = [float(row[feat]) for row in history if row.get(feat) is not None]
        if len(vals) < 5:
            continue

        arr  = np.array(vals, dtype=float)
        mean = float(arr.mean())
        std  = float(arr.std())

        if std < 1e-6:
            continue

        z = (float(current_val) - mean) / std
        abs_z = abs(z)

        if   abs_z >= _Z_SEVERE:   severity = "severe"
        elif abs_z >= _Z_MODERATE: severity = "moderate"
        elif abs_z >= _Z_MILD:     severity = "mild"
        else:                      continue

        shocks[feat] = ShockSignal(
            feature       = feat,
            current_value = float(current_val),
            rolling_mean  = round(mean, 4),
            rolling_std   = round(std,  4),
            z_score       = round(z,    3),
            severity      = severity,
            direction     = "+" if z > 0 else "-",
        )

    if shocks:
        logger.info(
            "shock_detector[%s]: %d shock(s) — %s",
            entity_name,
            len(shocks),
            ", ".join(
                f"{f}{s.direction}(z={s.z_score:+.1f},{s.severity})"
                for f, s in sorted(shocks.items(), key=lambda x: abs(x[1].z_score), reverse=True)[:4]
            ),
        )

    return shocks


def get_shock_multiplier(shocks: dict[str, ShockSignal]) -> float:
    """
    Return a CI widening factor driven by the most severe shock present.
    1.0 when no shocks; up to 2.5× for severe structural breaks.
    """
    if not shocks:
        return 1.0
    _rank = {"mild": 0, "moderate": 1, "severe": 2}
    worst = max(shocks.values(), key=lambda s: _rank[s.severity])
    return _CI_MULTIPLIERS[worst.severity]


def get_shock_summary(shocks: dict[str, ShockSignal]) -> str:
    """One-line human-readable summary, empty when no shocks."""
    if not shocks:
        return ""
    parts = [
        f"{s.feature}{s.direction}(z={s.z_score:+.1f},{s.severity})"
        for s in sorted(shocks.values(), key=lambda x: abs(x.z_score), reverse=True)[:3]
    ]
    return "; ".join(parts)


def _load_history(entity_name: str, window_days: int) -> list[dict]:
    """Load world_state_history rows for the entity over the rolling window."""
    try:
        from data_layer.db import get_db, init_schema, table_exists
        init_schema()
        if not table_exists("world_state_history"):
            return []

        db = get_db()
        col_list = ", ".join(_MONITORED)
        rows = db.execute(
            f"""
            SELECT {col_list}
            FROM world_state_history
            WHERE entity_name ILIKE ?
              AND recorded_at >= CURRENT_TIMESTAMP - INTERVAL '{window_days} days'
            ORDER BY recorded_at ASC
            """,
            [entity_name],
        ).fetchall()

        return [dict(zip(_MONITORED, row)) for row in rows]
    except Exception as e:
        logger.debug("shock_detector._load_history failed: %s", e)
        return []
