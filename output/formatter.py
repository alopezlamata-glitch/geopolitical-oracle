from __future__ import annotations

import json
import logging
import os
import re
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from scipy.stats import beta as beta_dist

from collector.base import EvidenceBlock

logger = logging.getLogger(__name__)

_BOX_WIDTH = 60
_PREDICTIONS_DIR = Path(__file__).parent.parent / "data" / "predictions"

_VERBAL_LABELS = [
    (0.10, "Very unlikely"),
    (0.25, "Unlikely"),
    (0.40, "Somewhat unlikely"),
    (0.60, "Uncertain"),
    (0.75, "Somewhat likely"),
    (0.90, "Likely"),
    (1.01, "Very likely"),
]


def verbal_label(p: float) -> str:
    for threshold, label in _VERBAL_LABELS:
        if p < threshold:
            return label
    return "Very likely"


def beta_confidence_interval(p: float, n: int = 20, ci: float = 0.80) -> tuple[float, float]:
    """Beta distribution approximation for confidence interval."""
    p = max(0.001, min(0.999, p))
    alpha_p = p * n
    beta_p = (1 - p) * n
    lo = beta_dist.ppf((1 - ci) / 2, alpha_p, beta_p)
    hi = beta_dist.ppf(1 - (1 - ci) / 2, alpha_p, beta_p)
    return round(float(lo), 3), round(float(hi), 3)


def evidence_quality(evidence: list[EvidenceBlock]) -> str:
    """
    HIGH: Metaculus >100 forecasters AND Polymarket >$100k volume
    MEDIUM: at least one market with sufficient coverage
    LOW: LLM only, no market coverage
    """
    metaculus_high = False
    polymarket_high = False
    any_market = False

    for block in evidence:
        if block.source == "metaculus" and block.quality != "insufficient":
            any_market = True
            if block.metadata.get("forecasters", 0) > 100:
                metaculus_high = True
        elif block.source == "polymarket" and block.quality != "insufficient":
            any_market = True
            if block.metadata.get("volume", 0) > 100_000:
                polymarket_high = True

    if metaculus_high and polymarket_high:
        return "HIGH"
    if any_market:
        return "MEDIUM"
    return "LOW"


def _question_slug(question: str) -> str:
    slug = re.sub(r"[^\w\s]", "", question.lower())
    slug = re.sub(r"\s+", "_", slug).strip("_")
    return slug[:50]


def _box_line(content: str = "", width: int = _BOX_WIDTH) -> str:
    inner = width - 2  # account for │ on each side
    return f"│ {content:<{inner}} │"


def _box_top(width: int = _BOX_WIDTH) -> str:
    return "┌" + "─" * (width - 2) + "┐"


def _box_bottom(width: int = _BOX_WIDTH) -> str:
    return "└" + "─" * (width - 2) + "┘"


def _box_separator(width: int = _BOX_WIDTH) -> str:
    return "├" + "─" * (width - 2) + "┤"


def format_output(
    question: str,
    p_final: float,
    evidence: list[EvidenceBlock],
    llm_result: dict[str, Any],
    p_llm: float,
    p_market: Optional[float],
) -> str:
    ci_lo, ci_hi = beta_confidence_interval(p_final)
    label = verbal_label(p_final)
    answer = "YES" if p_final >= 0.5 else "NO"
    eq = evidence_quality(evidence)
    p_llm_pct = f"{p_llm:.0%}"

    lines = [_box_top()]
    lines.append(_box_line("  ORACLE ANSWER"))
    lines.append(_box_separator())

    # Wrap long question
    q_wrapped = textwrap.wrap(question, width=_BOX_WIDTH - 14)
    lines.append(_box_line(f"  Question: {q_wrapped[0]}"))
    for extra in q_wrapped[1:]:
        lines.append(_box_line(f"            {extra}"))

    lines.append(_box_line(f"  Answer:   {answer}"))
    lines.append(_box_line(f"  Prob:     {p_final:.0%}  ({label})"))
    lines.append(_box_line(f"  CI (80%): [{ci_lo:.0%}, {ci_hi:.0%}]"))
    lines.append(_box_line(f"  Evidence: {eq}"))
    lines.append(_box_separator())
    lines.append(_box_line("  Sources used:"))

    # Metaculus
    meta = next((b for b in evidence if b.source == "metaculus"), None)
    if meta and meta.quality != "insufficient" and meta.metadata.get("probability") is not None:
        fc = meta.metadata.get("forecasters", "?")
        mp = meta.metadata["probability"]
        lines.append(_box_line(f"  ✓ Metaculus:  {mp:.0%} ({fc} forecasters)"))
    else:
        lines.append(_box_line("  ✗ Metaculus:  insufficient"))

    # Polymarket
    poly = next((b for b in evidence if b.source == "polymarket"), None)
    if poly and poly.quality != "insufficient" and poly.metadata.get("probability") is not None:
        vol = poly.metadata.get("volume", 0)
        pp = poly.metadata["probability"]
        lines.append(_box_line(f"  ✓ Polymarket: {pp:.0%} (${vol:,.0f} vol)"))
    else:
        lines.append(_box_line("  ✗ Polymarket: insufficient"))

    # LLM
    lines.append(_box_line(f"  ✓ LLM:        {p_llm_pct} estimate"))

    # GDELT
    gdelt = next((b for b in evidence if b.source == "gdelt"), None)
    if gdelt and gdelt.quality != "insufficient":
        tone = gdelt.metadata.get("avg_tone", 0)
        count = gdelt.metadata.get("article_count", 0)
        lines.append(_box_line(f"  ✓ GDELT:      tone={tone:.2f}, {count} articles"))
    else:
        lines.append(_box_line("  ✗ GDELT:      no data"))

    # RSS
    rss = next((b for b in evidence if b.source == "rss"), None)
    if rss and rss.quality != "insufficient":
        n_art = rss.metadata.get("articles_matched", 0)
        lines.append(_box_line(f"  ✓ RSS:        {n_art} matching articles"))
    else:
        lines.append(_box_line("  ✗ RSS:        no matching articles"))

    lines.append(_box_separator())
    lines.append(_box_line("  Reasoning:"))

    reasoning = llm_result.get("reasoning", "")
    for chunk in textwrap.wrap(reasoning, width=_BOX_WIDTH - 5):
        lines.append(_box_line(f"  {chunk}"))

    key_factors = llm_result.get("key_factors", [])
    if key_factors:
        lines.append(_box_line(""))
        lines.append(_box_line("  Key factors:"))
        for kf in key_factors[:4]:
            for chunk in textwrap.wrap(f"  • {kf}", width=_BOX_WIDTH - 5):
                lines.append(_box_line(chunk))

    lines.append(_box_bottom())
    return "\n".join(lines)


def save_prediction(
    question: str,
    p_final: float,
    p_llm: float,
    p_market: Optional[float],
    llm_result: dict[str, Any],
    evidence: list[EvidenceBlock],
) -> Path:
    """Save prediction to JSON file for calibration tracking."""
    _PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%dT%H%M%SZ")
    slug = _question_slug(question)
    filename = f"{timestamp}_{slug}.json"

    eq = evidence_quality(evidence)
    sources_used = [b.source for b in evidence if b.quality != "insufficient"]

    record = {
        "question": question,
        "timestamp": now.isoformat(),
        "final_probability": round(p_final, 4),
        "p_llm": round(p_llm, 4),
        "p_market": round(p_market, 4) if p_market is not None else None,
        "llm_reasoning": llm_result.get("reasoning", ""),
        "llm_confidence": llm_result.get("confidence", ""),
        "key_factors": llm_result.get("key_factors", []),
        "evidence_quality": eq,
        "evidence_sources": sources_used,
        "resolved": False,
        "outcome": None,
    }

    path = _PREDICTIONS_DIR / filename
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.debug("Saved prediction to %s", path)
    return path
