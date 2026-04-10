from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional

import numpy as np

from model.trainer import load_model
from model.calibrator import load_calibrator
from features.builder import build_features, get_feature_names
from normalizer.canonical import CanonicalEvent

logger = logging.getLogger(__name__)


def compute_shap_attribution(
    features: dict[str, float],
    provenance: dict[str, list[dict]],
    events_by_id: dict[str, CanonicalEvent],
) -> dict:
    """
    Returns {
      top_positive: [...],  # list of {event_id, title, event_type, date, contribution}
      top_negative: [...],
      counterfactuals: [...],  # {event_id, p_without, delta}
      shap_values: {feature: phi}
    }
    """
    try:
        import shap
    except ImportError:
        logger.warning("attribution: shap not installed. Run: pip install shap")
        return {"top_positive": [], "top_negative": [], "counterfactuals": [], "shap_values": {}}

    model, feature_names = load_model()
    calibrator = load_calibrator()
    if model is None:
        return {"top_positive": [], "top_negative": [], "counterfactuals": [], "shap_values": {}}

    feat_names = feature_names or get_feature_names()
    x = np.array([[float(features.get(f, 0.0)) for f in feat_names]], dtype=np.float32)

    # TreeSHAP — pass booster directly to avoid XGBoost 3.x base_score format issue
    try:
        booster = model.get_booster()
        explainer = shap.TreeExplainer(booster)
    except Exception:
        explainer = shap.TreeExplainer(model)
    shap_raw = explainer.shap_values(x)
    if isinstance(shap_raw, list):
        phi_arr = shap_raw[1][0]  # class 1 (binary classifier)
    elif shap_raw.ndim == 3:
        phi_arr = shap_raw[0, :, 1]   # (samples, features, classes) → class 1
    else:
        phi_arr = shap_raw[0]

    shap_by_feature = {fname: float(phi_arr[i]) for i, fname in enumerate(feat_names)}

    # Map SHAP → events via provenance
    event_contributions: dict[str, float] = defaultdict(float)
    for fname, phi in shap_by_feature.items():
        contributors = provenance.get(fname, [])
        total_w = sum(c["weight"] for c in contributors) or 1.0
        for c in contributors:
            event_contributions[c["event_id"]] += phi * (c["weight"] / total_w)

    # Sort
    all_contribs = sorted(event_contributions.items(), key=lambda x: x[1], reverse=True)
    positive = [(eid, c) for eid, c in all_contribs if c > 0][:5]
    negative = sorted([(eid, c) for eid, c in event_contributions.items() if c < 0], key=lambda x: x[1])[:3]

    def _format_event(eid: str, contrib: float) -> dict:
        ev = events_by_id.get(eid)
        return {
            "event_id": eid,
            "title": ev.raw_title if ev else "",
            "event_type": ev.event_type if ev else "",
            "date": ev.occurred_at.strftime("%Y-%m-%d") if ev else "",
            "country": ev.country if ev else "",
            "contribution": round(contrib, 4),
        }

    top_positive = [_format_event(eid, c) for eid, c in positive]
    top_negative = [_format_event(eid, c) for eid, c in negative]

    # Counterfactuals for top 3 positive events.
    # We use the SHAP-based approximation rather than re-running the full pipeline,
    # because isotonic calibration is a step function: nearby raw probabilities
    # frequently map to the same calibrated value, making model-rerun deltas = 0.
    #
    # SHAP approximation: removing event e shifts raw log-odds by -phi_e, so
    #   logit(p_without_raw) ≈ logit(p_raw) - event_contribution_in_logodds
    # TreeSHAP values from XGBoost are in log-odds (margin) space by default,
    # but after output_margin=False they are in probability space. We therefore
    # work in probability space and apply a logit→sigmoid round-trip.
    base_prob_raw = float(model.predict_proba(x)[0, 1])
    base_prob = float(calibrator.predict([base_prob_raw])[0]) if calibrator else base_prob_raw

    eps = 1e-6
    logit_base = float(np.log(np.clip(base_prob_raw, eps, 1 - eps) /
                               (1 - np.clip(base_prob_raw, eps, 1 - eps))))

    # Re-run SHAP in margin (log-odds) space for accurate counterfactuals
    try:
        booster = model.get_booster()
        shap_margin = booster.predict(
            booster.DMatrix(x), output_margin=True
        )  # raw log-odds score
        # Per-feature SHAP in log-odds space via separate explainer call
        ex_margin = shap.TreeExplainer(booster, output_type="margin") \
            if hasattr(shap.TreeExplainer, "__init__") else explainer
        phi_margin = ex_margin.shap_values(x)
        if hasattr(phi_margin, "ndim") and phi_margin.ndim == 2:
            phi_margin_arr = phi_margin[0]
        else:
            phi_margin_arr = phi_arr  # fallback to probability-space SHAP
    except Exception:
        phi_margin_arr = phi_arr  # fallback

    shap_margin_by_feature = {
        fname: float(phi_margin_arr[i]) for i, fname in enumerate(feat_names)
    }

    # Re-map SHAP (margin space) to event contributions
    event_contribs_margin: dict[str, float] = defaultdict(float)
    for fname, phi in shap_margin_by_feature.items():
        contributors = provenance.get(fname, [])
        total_w = sum(c["weight"] for c in contributors) or 1.0
        for c in contributors:
            event_contribs_margin[c["event_id"]] += phi * (c["weight"] / total_w)

    counterfactuals = []
    for eid, _ in positive[:3]:
        ev = events_by_id.get(eid)
        if ev is None:
            continue
        delta_logodds = event_contribs_margin.get(eid, event_contributions.get(eid, 0.0))
        logit_minus = logit_base - delta_logodds
        p_minus_raw = float(1.0 / (1.0 + np.exp(-logit_minus)))
        p_minus = float(calibrator.predict([p_minus_raw])[0]) if calibrator else p_minus_raw
        p_minus = max(0.01, min(0.99, p_minus))
        # If calibrator step function collapsed the delta, fall back to raw delta
        delta_cal = round(base_prob - p_minus, 4)
        if delta_cal == 0.0 and abs(delta_logodds) > 0.001:
            # Use logit-space delta scaled to probability via local derivative p*(1-p)
            p_raw_clamped = max(eps, min(1 - eps, base_prob_raw))
            delta_cal = round(delta_logodds * p_raw_clamped * (1 - p_raw_clamped), 4)
            p_minus = round(max(0.01, min(0.99, base_prob - delta_cal)), 4)
        counterfactuals.append({
            "event_id": eid,
            "title": ev.raw_title,
            "event_type": ev.event_type,
            "date": ev.occurred_at.strftime("%Y-%m-%d"),
            "p_without": p_minus,
            "delta": delta_cal,
        })

    return {
        "top_positive": top_positive,
        "top_negative": top_negative,
        "counterfactuals": counterfactuals,
        "shap_values": {k: round(v, 5) for k, v in shap_by_feature.items()},
    }
