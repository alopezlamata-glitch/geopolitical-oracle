"""
Trajectory predictor — assembles per-domain VAR forecasts into full
world state trajectories, then marginalizes probability over them.

The key computation:

  P(event by deadline | current_state) =
    1 - prod_{day=1}^{horizon} (1 - P(event | state_t+day))

Where P(event | state) uses the base-rate + feature-weight accumulation
from base_rate_predictor, but driven by the future state vector instead
of the current one.

This converts the system from:
  "What is P(event) given today's evidence?"
to:
  "What is P(event) given the full expected trajectory?"

The second is strictly more informative when the VAR model has been
fitted on enough history to have meaningful forecasts.
"""
from __future__ import annotations

import hashlib
import logging
import math
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import numpy as np

from model.transition_var import (
    DOMAIN_FEATURES, ALL_DOMAINS,
    load_model, predict, predict_distribution,
)

logger = logging.getLogger(__name__)

# Minimum days of VAR history before trusting trajectory over current state
MIN_HISTORY_FOR_TRAJECTORY = 14

_BASE_RATES_PATH = None   # lazy-loaded

# Blend factor for causal neighbor influence per trajectory step.
# Small so neighbor pressure nudges — does not override — the VAR forecast.
_CAUSAL_BLEND = 0.05


def _entity_id(name: str, entity_type: str = "country") -> str:
    raw = f"{entity_type}|{name.lower()}"
    return "ent_" + hashlib.sha256(raw.encode()).hexdigest()[:24]


# ── Assemble full trajectory ─────────────────────────────────────────────────

def _load_causal_links_for_entity(entity_name: str) -> list[dict]:
    """Return causal links where entity_name is the target."""
    try:
        from world_state.entity_graph import get_causal_neighbors
        return get_causal_neighbors(entity_name)
    except Exception:
        return []


def _load_neighbor_current_states(links: list[dict]) -> dict[str, dict]:
    """Load current world state for each unique source entity in links."""
    from world_state.reader import get_world_state
    states: dict[str, dict] = {}
    for link in links:
        src = link.get("source_entity", "")
        if src and src not in states:
            st = get_world_state(src)
            if st:
                states[src] = st
    return states


def _apply_causal_nudges(
    state: dict,
    day_idx: int,
    links: list[dict],
    neighbor_states: dict[str, dict],
) -> None:
    """
    Nudge target features in `state` based on causal neighbor values.

    Each link contributes:
      Δ = weight × source_value × exp(-day_idx / decay_days) × CAUSAL_BLEND

    Effect decays with time; CAUSAL_BLEND keeps it from overwhelming VAR.
    Mutates state in-place.
    """
    for link in links:
        src       = link.get("source_entity", "")
        src_feat  = link.get("source_feature", "")
        tgt_feat  = link.get("target_feature") or link.get("source_feature")
        weight    = float(link.get("weight", 0.0))
        decay     = float(link.get("decay_days", 30.0))

        nb_state = neighbor_states.get(src)
        if nb_state is None:
            continue

        src_val = nb_state.get(src_feat)
        if src_val is None or src_val == 0.0:
            continue

        exp_decay = math.exp(-day_idx / max(1.0, decay))
        delta     = weight * float(src_val) * exp_decay * _CAUSAL_BLEND

        if tgt_feat in state:
            state[tgt_feat] = round(
                max(-5.0, min(100.0, float(state[tgt_feat]) + delta)), 6
            )


def predict_trajectory(
    entity_name: str,
    horizon_days: int,
    n_samples: int = 100,
) -> Optional[list[dict]]:
    """
    Predict the full world state trajectory for `entity_name` over
    `horizon_days` days.

    Returns list of length `horizon_days`, each element being a flat
    feature dict compatible with world_state columns.

    Returns None if no VAR models are fitted for this entity.

    Cross-entity propagation: at each step, causal neighbor states nudge
    target features (decaying with day index) so that e.g. Russia's current
    military buildup actively bends Ukraine's conflict trajectory.
    """
    eid = _entity_id(entity_name)

    # Load current state as anchor
    from world_state.reader import get_world_state
    current_state = get_world_state(entity_name)
    if current_state is None:
        return None

    # Load causal links + neighbor states for cross-entity propagation
    causal_links   = _load_causal_links_for_entity(entity_name)
    neighbor_states = _load_neighbor_current_states(causal_links)
    if neighbor_states:
        logger.debug(
            "trajectory[%s]: %d causal links from %d neighbor(s)",
            entity_name, len(causal_links), len(neighbor_states),
        )

    # Forecast per domain
    domain_forecasts: dict[str, Optional[np.ndarray]] = {}
    for domain in ALL_DOMAINS:
        fc = predict(eid, domain, horizon_days)
        domain_forecasts[domain] = fc

    if all(v is None for v in domain_forecasts.values()):
        logger.debug("trajectory: no VAR models for %s — using static state", entity_name)
        # Static fallback: repeat current state for all days (still apply causal nudges)
        trajectory: list[dict] = []
        for day_idx in range(horizon_days):
            state = dict(current_state)
            _apply_causal_nudges(state, day_idx, causal_links, neighbor_states)
            trajectory.append(state)
        return trajectory

    # Assemble per-day state dicts
    trajectory = []
    for day_idx in range(horizon_days):
        state = dict(current_state)  # start from current

        for domain, fc in domain_forecasts.items():
            if fc is None:
                continue
            features = DOMAIN_FEATURES[domain]
            if day_idx >= len(fc):
                continue
            day_vector = fc[day_idx]
            for i, fname in enumerate(features):
                if i < len(day_vector):
                    val = float(day_vector[i])
                    val = max(-5.0, min(100.0, val))
                    state[fname] = round(val, 6)

        # Cross-entity causal nudge (applied after VAR step)
        _apply_causal_nudges(state, day_idx, causal_links, neighbor_states)

        trajectory.append(state)

    return trajectory


def predict_trajectory_distribution(
    entity_name: str,
    horizon_days: int,
    n_samples: int = 200,
) -> Optional[list[list[dict]]]:
    """
    Sample `n_samples` full-state trajectories (for uncertainty quantification).
    Returns list[n_samples] of list[horizon_days] of state dicts.
    Cross-entity causal nudges are applied to every sampled trajectory.
    """
    eid = _entity_id(entity_name)

    from world_state.reader import get_world_state
    current_state = get_world_state(entity_name)
    if current_state is None:
        return None

    # Load causal links + neighbor states once (shared across all samples)
    causal_links    = _load_causal_links_for_entity(entity_name)
    neighbor_states = _load_neighbor_current_states(causal_links)

    # Sample per domain
    domain_samples: dict[str, Optional[np.ndarray]] = {}
    for domain in ALL_DOMAINS:
        samples = predict_distribution(eid, domain, horizon_days, n_samples=n_samples)
        domain_samples[domain] = samples   # (n_samples, horizon, n_feat) or None

    # Assemble n_samples trajectories
    all_trajectories: list[list[dict]] = []
    for s in range(n_samples):
        traj: list[dict] = []
        for day_idx in range(horizon_days):
            state = dict(current_state)
            for domain, samples_arr in domain_samples.items():
                if samples_arr is None:
                    continue
                features = DOMAIN_FEATURES[domain]
                day_vector = samples_arr[s, day_idx, :]
                for i, fname in enumerate(features):
                    if i < len(day_vector):
                        state[fname] = round(float(np.clip(day_vector[i], -5.0, 100.0)), 6)
            # Cross-entity causal nudge (deterministic — same neighbor state for all samples)
            _apply_causal_nudges(state, day_idx, causal_links, neighbor_states)
            traj.append(state)
        all_trajectories.append(traj)

    return all_trajectories


# ── Probability marginalization ───────────────────────────────────────────────

def _state_to_logodds(
    state: dict,
    predicate: str,
    event_family: str,
    horizon_days: int,
) -> float:
    """
    Compute log-odds of event occurring given a single state vector.
    Uses base_rate_predictor logic (base rate + feature weight accumulation).
    """
    from predictor.base_rate_predictor import (
        _lookup_base_rate, _logit, _load_feature_weights,
    )

    base_rate = _lookup_base_rate(predicate, horizon_days)
    log_odds = _logit(base_rate)

    weights_data = _load_feature_weights()
    domain_weights: dict = weights_data.get(event_family, {})

    for fname, w in domain_weights.items():
        if fname.startswith("_"):
            continue
        value = state.get(fname, 0.0)
        if isinstance(value, (int, float)) and value != 0.0:
            log_odds += float(w) * float(value)

    return log_odds


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, x))))


def marginalize_probability(
    trajectory: list[dict],
    predicate: str,
    event_family: str,
) -> dict:
    """
    Compute P(event by deadline) by marginalizing over trajectory.

    P(event by day H) = 1 - prod_{d=1}^{H} (1 - P(event on day d))

    Where P(event on day d) is the single-day probability derived from
    state[d] using base_rate + feature weights.

    Returns dict with:
      p_trajectory  : P(event by deadline)
      daily_probs   : list of single-day probabilities
      peak_day      : day with highest single-day risk
      peak_prob     : peak single-day probability
      risk_profile  : 'early' | 'late' | 'sustained' | 'declining'
    """
    n_days = len(trajectory)
    if n_days == 0:
        return {"p_trajectory": 0.15, "daily_probs": [], "peak_day": 0,
                "peak_prob": 0.15, "risk_profile": "unknown"}

    # Single-day horizon for marginal probability at each step
    daily_probs = []
    for state in trajectory:
        lo = _state_to_logodds(state, predicate, event_family, horizon_days=7)
        p_day = max(0.001, min(0.999, _sigmoid(lo)))
        daily_probs.append(p_day)

    # Survival product: P(no event up to day H)
    p_no_event = 1.0
    for p in daily_probs:
        p_no_event *= (1.0 - p)
        p_no_event = max(0.0, p_no_event)

    p_trajectory = 1.0 - p_no_event

    # Risk profile
    peak_day = int(np.argmax(daily_probs))
    peak_prob = float(daily_probs[peak_day])

    first_half_mean = float(np.mean(daily_probs[: n_days // 2])) if n_days > 1 else peak_prob
    second_half_mean = float(np.mean(daily_probs[n_days // 2:])) if n_days > 1 else peak_prob

    if abs(first_half_mean - second_half_mean) < 0.02:
        risk_profile = "sustained"
    elif first_half_mean > second_half_mean * 1.2:
        risk_profile = "declining"
    elif second_half_mean > first_half_mean * 1.2:
        risk_profile = "late"
    else:
        risk_profile = "early" if peak_day < n_days // 2 else "late"

    return {
        "p_trajectory": round(min(0.99, p_trajectory), 4),
        "daily_probs": [round(p, 4) for p in daily_probs],
        "peak_day": peak_day,
        "peak_prob": round(peak_prob, 4),
        "risk_profile": risk_profile,
    }


def marginalize_with_uncertainty(
    entity_name: str,
    predicate: str,
    event_family: str,
    horizon_days: int,
    n_samples: int = 200,
) -> dict:
    """
    Compute trajectory probability with uncertainty bounds via Monte Carlo.

    Returns:
      p_mean        : mean P(event by deadline) across all trajectory samples
      p_lo, p_hi    : 10th–90th percentile credible interval
      p_trajectory  : same as p_mean (compatibility alias)
      n_samples     : actual number of samples used
      risk_profile  : modal risk profile across samples
    """
    traj_dist = predict_trajectory_distribution(entity_name, horizon_days, n_samples)

    if traj_dist is None:
        # No world model available — return None so caller uses current-state prediction
        return {}

    all_probs = []
    profiles = []
    for traj in traj_dist:
        result = marginalize_probability(traj, predicate, event_family)
        all_probs.append(result["p_trajectory"])
        profiles.append(result["risk_profile"])

    arr = np.array(all_probs)
    p_mean = float(arr.mean())
    p_lo   = float(np.percentile(arr, 10))
    p_hi   = float(np.percentile(arr, 90))

    # Modal risk profile
    from collections import Counter
    risk_profile = Counter(profiles).most_common(1)[0][0]

    # Shock detection: widen CI when current state has structural breaks
    shock_mult = 1.0
    shock_summary = ""
    try:
        from world_state.shock_detector import detect_shocks, get_shock_multiplier, get_shock_summary
        shocks = detect_shocks(entity_name)
        shock_mult = get_shock_multiplier(shocks)
        shock_summary = get_shock_summary(shocks)
        if shock_mult > 1.0:
            half_range = (p_hi - p_lo) / 2.0
            p_lo = max(0.01, p_mean - half_range * shock_mult)
            p_hi = min(0.99, p_mean + half_range * shock_mult)
            logger.info(
                "trajectory[%s]: shock multiplier=%.1f — CI widened to [%.3f, %.3f]",
                entity_name, shock_mult, p_lo, p_hi,
            )
    except Exception as e:
        logger.debug("trajectory: shock detection skipped: %s", e)

    ci_method = "var_monte_carlo" if any(
        load_model(_entity_id(entity_name), d) is not None
        and load_model(_entity_id(entity_name), d).get("type") == "var"
        for d in ALL_DOMAINS
    ) else "random_walk_monte_carlo"
    if shock_mult > 1.0:
        ci_method += f"+shock_x{shock_mult:.1f}"

    result = {
        "p_trajectory":  round(min(0.99, p_mean), 4),
        "p_mean":        round(min(0.99, p_mean), 4),
        "p_lo":          round(max(0.01, p_lo),   4),
        "p_hi":          round(min(0.99, p_hi),   4),
        "n_samples":     len(traj_dist),
        "risk_profile":  risk_profile,
        "trajectory_ci_method": ci_method,
    }
    if shock_summary:
        result["shock_signals"] = shock_summary
    return result
