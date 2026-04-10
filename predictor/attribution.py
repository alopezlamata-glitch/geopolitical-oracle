from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np

from model.trainer import load_model
from model.calibrator import load_calibrator
from features.builder import get_feature_names
from normalizer.canonical import CanonicalEvent

logger = logging.getLogger(__name__)

# ── Process-level singletons — avoid rebuilding TreeExplainer per request ─────
_model_cache: dict = {}
_calibrator_cache: dict = {}
_explainer_cache: dict = {}


def _get_model_and_explainer():
    """Return cached (model, feature_names, explainer, booster). Build once per process."""
    if "model" not in _model_cache:
        model, feature_names = load_model()
        _model_cache["model"] = model
        _model_cache["feature_names"] = feature_names
        if model is not None:
            try:
                import shap
                booster = model.get_booster()
                _explainer_cache["explainer"] = shap.TreeExplainer(booster)
                _explainer_cache["booster"] = booster
            except Exception as e:
                logger.warning("attribution: could not build TreeExplainer: %s", e)
    return (
        _model_cache.get("model"),
        _model_cache.get("feature_names"),
        _explainer_cache.get("explainer"),
    )


def _get_calibrator():
    if "cal" not in _calibrator_cache:
        _calibrator_cache["cal"] = load_calibrator()
    return _calibrator_cache["cal"]


def compute_shap_attribution(
    features: dict[str, float],
    provenance: dict[str, list[dict]],
    events_by_id: dict[str, CanonicalEvent],
) -> dict:
    """
    Compute TreeSHAP attribution and counterfactuals.

    ## SHAP space
    We use the raw XGBoost booster so that TreeExplainer returns values in
    margin (log-odds) space, where additivity holds exactly:
        sum(phi_i) + expected_value = logit(p_raw)

    ## Display space
    Convert to probability via local linearisation:
        delta_prob_approx ≈ delta_logodds * p_raw * (1 - p_raw)
    Contribution numbers in the output correspond to approximate probability
    shifts. They are labelled `delta_prob_approx` to distinguish from the
    exact `delta_logodds` values (which should always be preferred for
    ranking and arithmetic).

    ## Counterfactuals
    Subtract the event's log-odds contribution from logit(p_raw), apply
    sigmoid, then calibrate. This bypasses the isotonic step-function
    collapse (Δ=0) that occurs when re-running the full pipeline.

    The result is labelled `approximate_removal_effect`, not a causal
    estimate: we remove the event's *modelled contribution* to the
    prediction, not the event from reality.

    ## Flip-set (minimal removal)
    Greedy search: remove events (largest |delta_logodds| first) that
    support the current answer, until the predicted side of the threshold
    changes. This is the minimum set whose removal (per the model) would
    flip YES↔NO.
    """
    try:
        import shap
    except ImportError:
        logger.warning("attribution: shap not installed. Run: pip install shap")
        return _empty()

    model, feature_names, explainer = _get_model_and_explainer()
    calibrator = _get_calibrator()

    if model is None or explainer is None:
        return _empty()

    feat_names = feature_names or get_feature_names()
    x = np.array([[float(features.get(f, 0.0)) for f in feat_names]], dtype=np.float32)

    # ── SHAP in margin (log-odds) space ──────────────────────────────────────
    phi_raw = explainer.shap_values(x)

    # Normalise output shape across shap versions
    if isinstance(phi_raw, list):
        phi_arr = np.array(phi_raw[1][0])    # binary: class-1 SHAP
    elif phi_raw.ndim == 3:
        phi_arr = phi_raw[0, :, 1]           # (samples, features, classes)
    else:
        phi_arr = phi_raw[0]                 # (samples, features)

    shap_margin = {fname: float(phi_arr[i]) for i, fname in enumerate(feat_names)}

    # ── Map log-odds SHAP → events via provenance ─────────────────────────────
    event_margin: dict[str, float] = defaultdict(float)
    for fname, phi in shap_margin.items():
        contributors = provenance.get(fname, [])
        total_w = sum(c["weight"] for c in contributors) or 1.0
        for c in contributors:
            event_margin[c["event_id"]] += phi * (c["weight"] / total_w)

    # ── Convert to probability space for display ──────────────────────────────
    p_raw = float(model.predict_proba(x)[0, 1])
    local_slope = max(p_raw * (1.0 - p_raw), 1e-4)
    event_prob = {eid: v * local_slope for eid, v in event_margin.items()}

    # ── Rank events ───────────────────────────────────────────────────────────
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
            "sub_event_type": ev.sub_event_type if ev else "",
            "date": ev.occurred_at.strftime("%Y-%m-%d") if ev else "",
            "country": ev.country if ev else "",
            # Two spaces: exact additive (log-odds) and approximate display (prob)
            "delta_logodds": round(event_margin.get(eid, 0.0), 5),
            "delta_prob_approx": round(contrib, 4),
            # Backward-compat alias
            "contribution": round(contrib, 4),
        }

    top_positive = [_fmt(eid, c) for eid, c in positive]
    top_negative = [_fmt(eid, c) for eid, c in negative]

    # ── Calibrated base probability ───────────────────────────────────────────
    eps = 1e-6
    p_raw_c = float(np.clip(p_raw, eps, 1 - eps))
    logit_base = float(np.log(p_raw_c / (1.0 - p_raw_c)))
    base_prob = float(calibrator.predict([p_raw])[0]) if calibrator else p_raw
    base_prob = max(0.01, min(0.99, base_prob))

    # ── Per-event counterfactuals (top positive + top negative) ───────────────
    # Labelled `approximate_removal_effect`: removing the event's modelled
    # log-odds contribution, not a causal statement about the event itself.
    counterfactuals = []
    for eid, _ in list(positive[:3]) + list(negative[:2]):
        ev = events_by_id.get(eid)
        if ev is None:
            continue
        delta_lo = event_margin.get(eid, 0.0)
        p_minus_raw = float(1.0 / (1.0 + np.exp(-(logit_base - delta_lo))))
        p_minus_raw = float(np.clip(p_minus_raw, eps, 1 - eps))

        if calibrator is not None:
            p_minus = float(np.clip(calibrator.predict([p_minus_raw])[0], 0.01, 0.99))
        else:
            p_minus = p_minus_raw

        delta_cal = base_prob - p_minus

        # Fallback: isotonic step too coarse → local linearisation
        if abs(delta_cal) < 1e-4 and abs(delta_lo) > 0.001:
            delta_cal = delta_lo * local_slope
            p_minus = float(np.clip(base_prob - delta_cal, 0.01, 0.99))

        counterfactuals.append({
            "event_id": eid,
            "title": ev.raw_title,
            "event_type": ev.event_type,
            "sub_event_type": ev.sub_event_type,
            "date": ev.occurred_at.strftime("%Y-%m-%d"),
            "approximate_removal_effect": round(float(delta_cal), 4),
            "p_without": round(float(p_minus), 4),
            # Backward-compat alias
            "delta": round(float(delta_cal), 4),
        })

    # ── Flip-set: minimal greedy removal to cross decision threshold ──────────
    # Removes events that support the current answer (in log-odds) until the
    # calibrated probability crosses 0.5. This is an approximation: we work
    # in log-odds assuming independence of contributions.
    flip_set = _compute_flip_set(
        logit_base, event_margin, base_prob, calibrator, events_by_id, local_slope, eps
    )

    # ── SHAP values for output (probability space) ────────────────────────────
    shap_prob = {k: round(v * local_slope, 5) for k, v in shap_margin.items()}

    return {
        "top_positive": top_positive,
        "top_negative": top_negative,
        "counterfactuals": counterfactuals,
        "flip_set": flip_set,
        "shap_values": shap_prob,
    }


def _compute_flip_set(
    logit_base: float,
    event_margin: dict[str, float],
    base_prob: float,
    calibrator,
    events_by_id: dict[str, CanonicalEvent],
    local_slope: float,
    eps: float,
) -> dict:
    """
    Greedy minimal-removal flip-set.

    Strategy:
    - Determine current answer (YES if base_prob >= 0.5, else NO).
    - Sort events by |delta_logodds| descending.
    - Remove events whose contribution supports the current answer (i.e.
      positive events for YES, negative events for NO).
    - Stop when the calibrated probability crosses 0.5 or we run out of events.

    Returns the set of event IDs whose removal (per the model) flips the
    answer, along with the resulting probability.
    """
    answer_is_yes = base_prob >= 0.5
    ranked = sorted(event_margin.items(), key=lambda kv: abs(kv[1]), reverse=True)

    selected_ids = []
    logit_running = logit_base

    for eid, dlogit in ranked:
        # Only remove events that support the current answer
        if answer_is_yes and dlogit <= 0:
            continue
        if not answer_is_yes and dlogit >= 0:
            continue

        logit_running -= dlogit
        selected_ids.append(eid)

        p_minus_raw = float(np.clip(1.0 / (1.0 + np.exp(-logit_running)), eps, 1 - eps))
        if calibrator is not None:
            p_minus = float(np.clip(calibrator.predict([p_minus_raw])[0], 0.01, 0.99))
            if abs(base_prob - p_minus) < 1e-4 and abs(dlogit) > 0.001:
                # calibrator collapsed — use linearisation
                p_minus = float(np.clip(
                    base_prob - (logit_base - logit_running) * local_slope,
                    0.01, 0.99
                ))
        else:
            p_minus = p_minus_raw

        flipped = (answer_is_yes and p_minus < 0.5) or (not answer_is_yes and p_minus >= 0.5)
        if flipped:
            events = []
            for flip_eid in selected_ids:
                ev = events_by_id.get(flip_eid)
                events.append({
                    "event_id": flip_eid,
                    "title": ev.raw_title if ev else "",
                    "event_type": ev.event_type if ev else "",
                    "delta_logodds": round(event_margin.get(flip_eid, 0.0), 5),
                })
            return {
                "events": events,
                "p_after_removal": round(float(p_minus), 4),
                "delta": round(float(p_minus - base_prob), 4),
                "note": "approximate_removal_effect — not a causal estimate",
            }

    return {"events": [], "p_after_removal": None, "delta": None,
            "note": "no flip found within available events"}


def _empty() -> dict:
    return {
        "top_positive": [],
        "top_negative": [],
        "counterfactuals": [],
        "flip_set": {"events": [], "p_after_removal": None, "delta": None, "note": "model unavailable"},
        "shap_values": {},
    }
