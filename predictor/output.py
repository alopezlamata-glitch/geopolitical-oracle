from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_PREDICTIONS_DIR = Path(__file__).parent.parent / "data" / "predictions"
_MODEL_PATH = Path(__file__).parent.parent / "data" / "model" / "xgb_model.json"
_SCHEMA_VERSION = "1.1"


def _model_sha256() -> str:
    """SHA-256 of the model artifact for reproducibility / audit trail."""
    if not _MODEL_PATH.exists():
        return "untrained"
    h = hashlib.sha256()
    h.update(_MODEL_PATH.read_bytes())
    return h.hexdigest()[:16]   # first 16 hex chars — enough for drift detection


def _verbal(p: float) -> str:
    if p < 0.10: return "Very unlikely"
    if p < 0.25: return "Unlikely"
    if p < 0.40: return "Somewhat unlikely"
    if p < 0.60: return "Uncertain"
    if p < 0.75: return "Somewhat likely"
    if p < 0.90: return "Likely"
    return "Very likely"


def _evidence_quality(features: dict[str, float], n_events: int) -> str:
    diversity = features.get("source_diversity_7d", 0.0)
    market = features.get("market_available", 0.0)
    if diversity > 0.5 and n_events > 10:
        return "HIGH"
    elif n_events >= 5 or market:
        return "MEDIUM"
    return "LOW"


def _slug(text: str) -> str:
    return re.sub(r"[^\w]+", "_", text.lower())[:50].strip("_")


def format_output(
    question: str,
    prediction: dict,
    attribution: dict,
    features: dict[str, float],
    n_events: int,
    drift_flags: list[str],
) -> str:
    p = prediction["calibrated_prob"]
    raw = prediction["raw_prob"]
    lo = prediction["ci_lo"]
    hi = prediction["ci_hi"]
    ans = prediction["answer"]
    untrained = prediction.get("untrained", False)
    ci_method = prediction.get("ci_method", "heuristic")
    verbal = _verbal(p)
    quality = _evidence_quality(features, n_events)

    W = 62  # inner width (between │ and │)

    def row(content: str) -> str:
        """Pad content to exactly W chars and wrap with │."""
        # Strip ANSI just in case; truncate if over
        c = content[:W]
        return f"│{c:<{W}}│"

    lines = [
        "┌" + "─" * W + "┐",
        row(f"  ORACLE PREDICTION"),
        row(f"  {'─' * (W - 4)}"),
        row(f"  Question  : {question[:W-16]}"),
        row(f"  Answer    : {ans}"),
        row(f"  Probability : {p:.3f}  ({verbal})"),
        row(f"  Calibrated  : {p:.3f}  [{lo:.2f}, {hi:.2f}]  80% CI ({ci_method})"),
        row(f"  Raw model   : {raw:.3f}"),
        row(f"  Evidence quality: {quality}"),
    ]

    if untrained:
        lines.append(row(f"  ⚠  UNTRAINED MODEL — prior estimate only"))

    if drift_flags:
        drift_summary = f"{len(drift_flags)} features" if len(drift_flags) > 3 else ', '.join(drift_flags)
        lines.append(row(f"  ⚠  DRIFT detected: {drift_summary}"))

    lines.append(row(""))

    top_pos = attribution.get("top_positive", [])
    top_neg = attribution.get("top_negative", [])

    if top_pos or top_neg:
        lines.append(row("  TOP CONTRIBUTING EVENTS"))
        for ev in top_pos[:4]:
            c = ev["contribution"]
            tag = ev.get("event_type", "?")[:18]
            dt = ev.get("date", "")
            title = ev.get("title", "")[:22]
            lines.append(row(f"  +{c:.3f}  {tag:<18}  {dt}  {title}"))
        for ev in top_neg[:2]:
            c = ev["contribution"]
            tag = ev.get("event_type", "?")[:18]
            dt = ev.get("date", "")
            title = ev.get("title", "")[:22]
            lines.append(row(f"  {c:.3f}  {tag:<18}  {dt}  {title}"))

    cfs = attribution.get("counterfactuals", [])
    if cfs:
        lines.append(row(""))
        lines.append(row("  COUNTERFACTUALS"))
        for cf in cfs:
            lines.append(row(f"  Without '{cf['title'][:20]}': p → {cf['p_without']:.3f} (Δ={cf['delta']:+.3f})"))

    lines.append("└" + "─" * W + "┘")
    return "\n".join(lines)


def save_prediction(
    question: str,
    prediction: dict,
    attribution: dict,
    features: dict[str, float],
    provenance: dict,
    n_events: int,
) -> Path:
    """
    Persist a versioned, auditable prediction record.

    Schema v1.1 adds:
      - schema_version      : for forward-compatible parsing
      - prediction_id       : {timestamp}_{slug}, stable identifier
      - model_version       : SHA-256 prefix of xgb_model.json
      - ci_method           : conformal | conformal-sym | heuristic
      - provenance          : full feature→event mapping (compressed to top-5 per feature)
      - flip_set            : minimal removal set from attribution
    """
    _PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    ts = now.strftime("%Y%m%dT%H%M%S")
    slug = _slug(question)
    prediction_id = f"{ts}_{slug}"
    path = _PREDICTIONS_DIR / f"{prediction_id}.json"

    # Compress provenance: keep top-5 contributors per feature to bound file size
    provenance_compact: dict = {}
    for fname, contribs in (provenance or {}).items():
        top = sorted(contribs, key=lambda c: c.get("weight", 0), reverse=True)[:5]
        if top:
            provenance_compact[fname] = top

    payload = {
        "schema_version": _SCHEMA_VERSION,
        "prediction_id": prediction_id,
        "model_version": _model_sha256(),
        "question": question,
        "timestamp": now.isoformat(),
        "answer": prediction["answer"],
        "calibrated_prob": prediction["calibrated_prob"],
        "raw_prob": prediction["raw_prob"],
        "ci_lo": prediction["ci_lo"],
        "ci_hi": prediction["ci_hi"],
        "ci_method": prediction.get("ci_method", "heuristic"),
        "untrained": prediction.get("untrained", False),
        "n_events": n_events,
        "features": features,
        "attribution": {
            "top_positive": attribution.get("top_positive", []),
            "top_negative": attribution.get("top_negative", []),
            "counterfactuals": attribution.get("counterfactuals", []),
            "flip_set": attribution.get("flip_set", {}),
        },
        "provenance": provenance_compact,
        "resolved": False,
        "outcome": None,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return path
