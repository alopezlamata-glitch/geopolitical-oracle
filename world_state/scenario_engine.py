"""
Scenario engine — branching futures for world model predictions.

Generates 3 conditional trajectory branches by σ-perturbing the entity's
baseline trajectory at every step (effect decays over the horizon).

Scenarios
---------
  escalation    conflict features spike +2σ above VAR baseline
  baseline      unperturbed VAR expected path
  deescalation  ceasefire/diplomatic features spike +2σ above baseline

Usage
-----
  from world_state.scenario_engine import generate_scenarios
  result = generate_scenarios("Ukraine", "military_escalation", "conflict", 90)
  result["escalation"].p_event   # worst-case P(event by deadline)
  result["baseline"].p_event
  result["deescalation"].p_event
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_BASELINE_PATH = Path(__file__).parent.parent / "data" / "model" / "feature_baseline.json"

# σ-multipliers per feature per scenario.
# Positive = push feature up; negative = push it down.
_SCENARIO_DEFS: dict[str, dict[str, float]] = {
    "escalation": {
        "military_intensity_7d":  +2.0,
        "escalation_index":       +2.0,
        "military_accel":         +1.5,
        "military_count_7d":      +2.0,
        "military_share_7d":      +1.5,
        "avg_polarity_7d":        -1.5,   # more hostile
        "tone_trend":             -1.0,
        "ceasefire_count_7d":     -1.0,
        "ceasefire_ratio_7d":     -1.0,
    },
    "baseline": {},   # no perturbation
    "deescalation": {
        "ceasefire_count_7d":     +2.0,
        "ceasefire_ratio_7d":     +2.0,
        "avg_polarity_7d":        +1.5,   # more positive
        "tone_trend":             +1.0,
        "military_accel":         -1.5,
        "escalation_index":       -1.5,
        "military_intensity_7d":  -1.0,
        "military_count_7d":      -1.0,
    },
}

_SCENARIO_LABELS = {
    "escalation":    "Escalation   ",
    "baseline":      "Baseline     ",
    "deescalation":  "De-escalation",
}


@dataclass
class ScenarioResult:
    name:        str
    label:       str
    p_event:     float
    p_lo:        float
    p_hi:        float
    risk_profile: str
    peak_day:    int
    n_days:      int


def _load_feature_stds() -> dict[str, float]:
    """Load per-feature std from feature_baseline.json."""
    try:
        data = json.loads(_BASELINE_PATH.read_text(encoding="utf-8"))
        return {k: float(v.get("std", 1.0)) for k, v in data.items() if isinstance(v, dict)}
    except Exception as e:
        logger.debug("scenario_engine: failed to load feature_baseline: %s", e)
        return {}


def _perturb_trajectory(
    trajectory: list[dict],
    perturbations: dict[str, float],
    feature_stds: dict[str, float],
    horizon_days: int,
) -> list[dict]:
    """
    Copy trajectory and apply σ-weighted perturbations at each step.
    Effect decays exponentially: delta *= exp(-day / (horizon/3)).
    """
    if not perturbations:
        return [dict(s) for s in trajectory]

    decay_constant = max(1.0, horizon_days / 3.0)
    perturbed = []

    for day_idx, state in enumerate(trajectory):
        s = dict(state)
        decay = math.exp(-day_idx / decay_constant)

        for feat, sigma_mult in perturbations.items():
            std = feature_stds.get(feat, 1.0)
            delta = sigma_mult * std * decay
            current = float(s.get(feat, 0.0))
            s[feat] = round(max(-5.0, min(100.0, current + delta)), 6)

        perturbed.append(s)

    return perturbed


def generate_scenarios(
    entity_name: str,
    predicate: str,
    event_family: str,
    horizon_days: int,
    n_samples: int = 80,
) -> dict[str, ScenarioResult]:
    """
    Generate escalation / baseline / deescalation scenario results.

    Returns {scenario_name: ScenarioResult}.
    Returns {} if no trajectory can be built for the entity.
    """
    from world_state.trajectory import predict_trajectory, marginalize_probability

    # Baseline trajectory (mean path from VAR or static fallback)
    baseline_traj = predict_trajectory(entity_name, horizon_days, n_samples=n_samples)
    if baseline_traj is None:
        logger.debug("scenario_engine: no trajectory for %s", entity_name)
        return {}

    feature_stds = _load_feature_stds()
    results: dict[str, ScenarioResult] = {}

    for scenario_name, perturbations in _SCENARIO_DEFS.items():
        try:
            traj = _perturb_trajectory(
                baseline_traj, perturbations, feature_stds, horizon_days
            )
            marg = marginalize_probability(traj, predicate, event_family)

            p = marg["p_trajectory"]

            # Simple symmetric CI: ±10% of p for escalation/deescalation,
            # use actual distribution spread for baseline
            half = min(0.15, p * 0.20)
            p_lo = round(max(0.01, p - half), 4)
            p_hi = round(min(0.99, p + half), 4)

            results[scenario_name] = ScenarioResult(
                name         = scenario_name,
                label        = _SCENARIO_LABELS[scenario_name],
                p_event      = round(p, 4),
                p_lo         = p_lo,
                p_hi         = p_hi,
                risk_profile = marg.get("risk_profile", "unknown"),
                peak_day     = marg.get("peak_day", 0),
                n_days       = len(traj),
            )
        except Exception as e:
            logger.debug("scenario_engine[%s/%s]: failed: %s", entity_name, scenario_name, e)

    if results:
        esc  = results.get("escalation")
        base = results.get("baseline")
        desc = results.get("deescalation")
        logger.info(
            "scenarios[%s/%s %dd]: esc=%.3f base=%.3f desc=%.3f",
            entity_name, predicate, horizon_days,
            esc.p_event  if esc  else 0,
            base.p_event if base else 0,
            desc.p_event if desc else 0,
        )

    return results


def format_scenarios(scenarios: dict[str, ScenarioResult], horizon_days: int) -> str:
    """One-line-per-scenario formatted string for embedding in output box."""
    if not scenarios:
        return ""
    lines = [f"  SCENARIOS ({horizon_days}d horizon)"]
    for name in ("escalation", "baseline", "deescalation"):
        s = scenarios.get(name)
        if s is None:
            continue
        bar_len = int(s.p_event * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        lines.append(
            f"  {s.label}: {s.p_event:.3f}  {bar}  ({s.risk_profile})"
        )
    return "\n".join(lines)
