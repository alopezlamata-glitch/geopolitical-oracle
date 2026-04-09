from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_PREDICTIONS_DIR = Path(__file__).parent.parent / "data" / "predictions"


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
        row(f"  Calibrated  : {p:.3f}  [{lo:.2f}, {hi:.2f}]  80% CI"),
        row(f"  Raw model   : {raw:.3f}"),
        row(f"  Evidence quality: {quality}"),
    ]

    if untrained:
        lines.append(row(f"  ⚠  UNTRAINED MODEL — prior estimate only"))

    if drift_flags:
        lines.append(row(f"  ⚠  DRIFT: {', '.join(drift_flags[:3])}"))

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
    _PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    slug = _slug(question)
    path = _PREDICTIONS_DIR / f"{ts}_{slug}.json"
    payload = {
        "question": question,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "answer": prediction["answer"],
        "calibrated_prob": prediction["calibrated_prob"],
        "raw_prob": prediction["raw_prob"],
        "ci_lo": prediction["ci_lo"],
        "ci_hi": prediction["ci_hi"],
        "untrained": prediction.get("untrained", False),
        "n_events": n_events,
        "features": features,
        "attribution": {
            "top_positive": attribution.get("top_positive", []),
            "top_negative": attribution.get("top_negative", []),
            "counterfactuals": attribution.get("counterfactuals", []),
        },
        "resolved": False,
        "outcome": None,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return path
