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

    # TreeSHAP
    explainer = shap.TreeExplainer(model)
    shap_raw = explainer.shap_values(x)
    if isinstance(shap_raw, list):
        phi_arr = shap_raw[1][0]  # class 1
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
    negative = [(eid, c) for eid, c in all_contribs if c < 0][-3:]

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

    # Counterfactuals for top 3
    base_prob_raw = float(model.predict_proba(x)[0, 1])
    base_prob = float(calibrator.predict([base_prob_raw])[0]) if calibrator else base_prob_raw

    counterfactuals = []
    for eid, _ in positive[:3]:
        ev = events_by_id.get(eid)
        if ev is None:
            continue
        # Remove this event from the list and recompute features
        remaining = [e for e in events_by_id.values() if e.event_id != eid]
        meta_p = features.get("metaculus_p")
        poly_p = features.get("polymarket_p")
        feat_minus, _ = build_features(
            remaining,
            metaculus_p=meta_p if meta_p and meta_p > 0 else None,
            polymarket_p=poly_p if poly_p and poly_p > 0 else None,
        )
        x_minus = np.array([[float(feat_minus.get(f, 0.0)) for f in feat_names]], dtype=np.float32)
        p_minus_raw = float(model.predict_proba(x_minus)[0, 1])
        p_minus = float(calibrator.predict([p_minus_raw])[0]) if calibrator else p_minus_raw
        counterfactuals.append({
            "event_id": eid,
            "title": ev.raw_title,
            "event_type": ev.event_type,
            "date": ev.occurred_at.strftime("%Y-%m-%d"),
            "p_without": round(p_minus, 4),
            "delta": round(base_prob - p_minus, 4),
        })

    return {
        "top_positive": top_positive,
        "top_negative": top_negative,
        "counterfactuals": counterfactuals,
        "shap_values": {k: round(v, 5) for k, v in shap_by_feature.items()},
    }
