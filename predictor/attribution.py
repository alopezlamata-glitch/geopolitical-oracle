from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np

from model.trainer import load_model
from model.calibrator import load_calibrator
from features.builder import get_feature_names
from normalizer.canonical import CanonicalEvent

logger = logging.getLogger(__name__)


def compute_shap_attribution(
    features: dict[str, float],
    provenance: dict[str, list[dict]],
    events_by_id: dict[str, CanonicalEvent],
) -> dict:
    """
    Compute TreeSHAP attribution and counterfactuals.

    SHAP space: we use the raw XGBoost booster so that TreeExplainer returns
    values in margin (log-odds) space, where additivity holds exactly:
        sum(phi_i) + expected_value = logit(p_raw)

    Display space: we convert to probability via local linearization
        Δp ≈ Δlogit * p_raw * (1 - p_raw)
    so contribution numbers in the output correspond to approximate probability shifts.

    Counterfactuals: we subtract the event's log-odds contribution from logit(p_raw),
    apply sigmoid, then calibrate — bypassing the isotonic step-function collapse
    that causes Δ=0 when re-running full pipeline.
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

    # ── SHAP in margin (log-odds) space ─────────────────────────────────────
    # Passing the raw booster (not the sklearn wrapper) ensures TreeExplainer
    # returns margin-space SHAP values regardless of shap version.
    booster = model.get_booster()
    explainer = shap.TreeExplainer(booster)
    phi_raw = explainer.shap_values(x)

    # Normalise output shape across shap versions
    if isinstance(phi_raw, list):
        phi_arr = np.array(phi_raw[1][0])   # binary: class-1 SHAP
    elif phi_raw.ndim == 3:
        phi_arr = phi_raw[0, :, 1]          # (samples, features, classes)
    else:
        phi_arr = phi_raw[0]                # (samples, features)

    # phi_arr[i] is the marginal log-odds contribution of feature i
    shap_margin = {fname: float(phi_arr[i]) for i, fname in enumerate(feat_names)}

    # ── Map log-odds SHAP to events via provenance ───────────────────────────
    event_margin: dict[str, float] = defaultdict(float)
    for fname, phi in shap_margin.items():
        contributors = provenance.get(fname, [])
        total_w = sum(c["weight"] for c in contributors) or 1.0
        for c in contributors:
            event_margin[c["event_id"]] += phi * (c["weight"] / total_w)

    # ── Convert to probability space for display ─────────────────────────────
    # Local linearisation: Δp ≈ Δlogit * p*(1-p)
    # Valid when |Δlogit| is small; sufficient for ranking and display.
    p_raw = float(model.predict_proba(x)[0, 1])
    local_slope = max(p_raw * (1.0 - p_raw), 1e-4)
    event_prob = {eid: v * local_slope for eid, v in event_margin.items()}

    # ── Rank events ──────────────────────────────────────────────────────────
    all_ranked = sorted(event_prob.items(), key=lambda kv: kv[1], reverse=True)
    positive = [(eid, c) for eid, c in all_ranked if c > 0][:5]
    negative = sorted(
        [(eid, c) for eid, c in event_prob.items() if c < 0], key=lambda kv: kv[1]
    )[:3]

    def _fmt(eid: str, contrib: float) -> dict:
        ev = events_by_id.get(eid)
        return {
            "event_id": eid,
            "title": ev.raw_title if ev else "",
            "event_type": ev.event_type if ev else "",
            "date": ev.occurred_at.strftime("%Y-%m-%d") if ev else "",
            "country": ev.country if ev else "",
            "contribution": round(contrib, 4),
        }

    top_positive = [_fmt(eid, c) for eid, c in positive]
    top_negative = [_fmt(eid, c) for eid, c in negative]

    # ── Counterfactuals (margin-space, no pipeline re-run) ───────────────────
    # Remove event's log-odds contribution, apply sigmoid, calibrate.
    # This avoids isotonic step-function collapse (delta=0) that occurs when
    # re-running the full pipeline with a single event removed.
    #
    # If calibrator still collapses the delta (step function too coarse),
    # fall back to the local linearisation: Δp_cal ≈ Δlogit * local_slope.
    # This is an approximation, not a causal estimate, and is labelled as such.
    eps = 1e-6
    p_raw_c = float(np.clip(p_raw, eps, 1 - eps))
    logit_base = float(np.log(p_raw_c / (1.0 - p_raw_c)))
    base_prob = float(calibrator.predict([p_raw])[0]) if calibrator else p_raw
    base_prob = max(0.01, min(0.99, base_prob))

    counterfactuals = []
    for eid, _ in positive[:3]:
        ev = events_by_id.get(eid)
        if ev is None:
            continue
        delta_logodds = event_margin.get(eid, 0.0)
        p_minus_raw = float(1.0 / (1.0 + np.exp(-(logit_base - delta_logodds))))
        p_minus_raw = float(np.clip(p_minus_raw, eps, 1 - eps))

        if calibrator is not None:
            p_minus = float(np.clip(calibrator.predict([p_minus_raw])[0], 0.01, 0.99))
        else:
            p_minus = p_minus_raw

        delta_cal = base_prob - p_minus

        # Fallback: calibrator step too coarse → use linearisation
        if abs(delta_cal) < 1e-4 and abs(delta_logodds) > 0.001:
            delta_cal = delta_logodds * local_slope
            p_minus = float(np.clip(base_prob - delta_cal, 0.01, 0.99))

        counterfactuals.append({
            "event_id": eid,
            "title": ev.raw_title,
            "event_type": ev.event_type,
            "date": ev.occurred_at.strftime("%Y-%m-%d"),
            "p_without": round(float(p_minus), 4),
            "delta": round(float(delta_cal), 4),
        })

    # ── SHAP values for output (probability space) ───────────────────────────
    shap_prob = {k: round(v * local_slope, 5) for k, v in shap_margin.items()}

    return {
        "top_positive": top_positive,
        "top_negative": top_negative,
        "counterfactuals": counterfactuals,
        "shap_values": shap_prob,
    }
