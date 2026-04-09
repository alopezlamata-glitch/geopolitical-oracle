from __future__ import annotations

import json
import logging
import re
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from scipy.stats import beta as beta_dist

from collector.base import EvidenceBlock
from pipeline.temporal import TemporalSignal
from pipeline.scorer import RiskScore

logger = logging.getLogger(__name__)

_BOX_WIDTH = 72
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
    p = max(0.001, min(0.999, p))
    alpha_p = p * n
    beta_p = (1 - p) * n
    lo = beta_dist.ppf((1 - ci) / 2, alpha_p, beta_p)
    hi = beta_dist.ppf(1 - (1 - ci) / 2, alpha_p, beta_p)
    return round(float(lo), 3), round(float(hi), 3)


def evidence_quality(evidence: list[EvidenceBlock]) -> str:
    metaculus_high = polymarket_high = any_market = False
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


# ── Box drawing helpers ───────────────────────────────────────────────────────

def _box_line(content: str = "", width: int = _BOX_WIDTH) -> str:
    inner = width - 4
    return f"│  {content:<{inner}}  │"


def _box_top(width: int = _BOX_WIDTH) -> str:
    return "┌" + "─" * (width - 2) + "┐"


def _box_bottom(width: int = _BOX_WIDTH) -> str:
    return "└" + "─" * (width - 2) + "┘"


def _box_sep(width: int = _BOX_WIDTH) -> str:
    return "├" + "─" * (width - 2) + "┤"


def _wrap_lines(text: str, prefix: str = "", width: int = _BOX_WIDTH) -> list[str]:
    max_inner = width - 4 - len(prefix)
    chunks = textwrap.wrap(text, width=max_inner) if text else [""]
    return [_box_line(f"{prefix}{c}") for c in chunks]


# ── Main formatter ────────────────────────────────────────────────────────────

def format_output(
    question: str,
    p_final: float,
    evidence: list[EvidenceBlock],
    llm_result: dict[str, Any],
    p_llm: float,
    p_market: Optional[float],
    temporal: Optional[TemporalSignal] = None,
    risk_score: Optional[RiskScore] = None,
) -> str:
    ci_lo, ci_hi = beta_confidence_interval(p_final)
    label = verbal_label(p_final)
    answer = "YES" if p_final >= 0.5 else "NO"
    eq = evidence_quality(evidence)
    scenarios = llm_result.get("scenarios", [])

    W = _BOX_WIDTH
    lines: list[str] = [_box_top(W)]

    # ── Header ────────────────────────────────────────────────────────────────
    lines.append(_box_line("ORACLE ANSWER", W))
    lines.append(_box_sep(W))

    for chunk in textwrap.wrap(question, width=W - 14):
        lines.append(_box_line(f"Question: {chunk}", W))

    lines.append(_box_line(W))
    lines.append(_box_line(f"Answer:      {answer}", W))
    lines.append(_box_line(f"Probability: {p_final:.1%}  ({label})", W))
    lines.append(_box_line(f"CI (80%):    [{ci_lo:.0%}, {ci_hi:.0%}]", W))
    lines.append(_box_line(f"Evidence:    {eq}", W))

    # ── Risk score ────────────────────────────────────────────────────────────
    if risk_score:
        lines.append(_box_line(f"Risk score:  {risk_score.score:.0f}/100 [{risk_score.label}]", W))

    # ── Temporal signal ───────────────────────────────────────────────────────
    if temporal:
        lines.append(_box_line(f"Trend:       {temporal.summary}", W))

    # ── Scenarios ─────────────────────────────────────────────────────────────
    if scenarios:
        lines.append(_box_sep(W))
        lines.append(_box_line("SCENARIOS", W))
        lines.append(_box_line(W))
        for i, s in enumerate(scenarios, 1):
            name = s.get("name", f"Scenario {i}")
            weight = s.get("weight", 0)
            p_yes = s.get("probability_of_yes", 0)
            narrative = s.get("narrative", "")
            lines.append(_box_line(f"[{i}] {name}  (weight={weight:.0%}, p_yes={p_yes:.0%})", W))
            for chunk in textwrap.wrap(narrative, width=W - 10):
                lines.append(_box_line(f"    {chunk}", W))
            if i < len(scenarios):
                lines.append(_box_line(W))

    # ── Sources ───────────────────────────────────────────────────────────────
    lines.append(_box_sep(W))
    lines.append(_box_line("SOURCES", W))
    lines.append(_box_line(W))

    meta = next((b for b in evidence if b.source == "metaculus"), None)
    if meta and meta.quality != "insufficient" and meta.metadata.get("probability") is not None:
        fc = meta.metadata.get("forecasters", "?")
        mp = meta.metadata["probability"]
        lines.append(_box_line(f"✓ Metaculus:        {mp:.0%} ({fc} forecasters)", W))
    else:
        lines.append(_box_line("✗ Metaculus:        insufficient / blocked", W))

    poly = next((b for b in evidence if b.source == "polymarket"), None)
    if poly and poly.quality != "insufficient" and poly.metadata.get("probability") is not None:
        vol = poly.metadata.get("volume", 0)
        pp = poly.metadata["probability"]
        lines.append(_box_line(f"✓ Polymarket:       {pp:.0%} (${vol:,.0f} vol)", W))
    else:
        lines.append(_box_line("✗ Polymarket:       no matching market", W))

    lines.append(_box_line(f"✓ LLM estimate:     {p_llm:.1%} (weighted scenario avg)", W))

    gdelt = next((b for b in evidence if b.source == "gdelt"), None)
    if gdelt and gdelt.quality != "insufficient":
        tone = gdelt.metadata.get("avg_tone", 0)
        count = gdelt.metadata.get("article_count", 0)
        lines.append(_box_line(f"✓ GDELT snapshot:   tone={tone:+.2f}, {count} articles", W))
    else:
        lines.append(_box_line("✗ GDELT snapshot:   no data", W))

    ts = next((b for b in evidence if b.source == "gdelt_timeseries"), None)
    if ts and ts.quality != "insufficient" and temporal:
        lines.append(_box_line(f"✓ GDELT 30-day:     {temporal.risk_label} — {temporal._volume_description()}", W))
    else:
        lines.append(_box_line("✗ GDELT 30-day:     unavailable", W))

    acled = next((b for b in evidence if b.source == "acled"), None)
    if acled and acled.quality != "insufficient":
        ev = acled.metadata.get("event_count", 0)
        fat = acled.metadata.get("fatalities", 0)
        lines.append(_box_line(f"✓ ACLED:            {ev} events, {fat} fatalities", W))
    else:
        lines.append(_box_line("✗ ACLED:            not configured or no events", W))

    rss = next((b for b in evidence if b.source == "rss"), None)
    if rss and rss.quality != "insufficient":
        n_art = rss.metadata.get("articles_matched", 0)
        lines.append(_box_line(f"✓ RSS feeds:        {n_art} matching articles", W))
    else:
        lines.append(_box_line("✗ RSS feeds:        no matching articles", W))

    # ── Key factors ───────────────────────────────────────────────────────────
    key_factors = llm_result.get("key_factors", [])
    if key_factors:
        lines.append(_box_sep(W))
        lines.append(_box_line("KEY FACTORS", W))
        lines.append(_box_line(W))
        for kf in key_factors[:4]:
            for chunk in textwrap.wrap(f"• {kf}", width=W - 8):
                lines.append(_box_line(f"  {chunk}", W))

    lines.append(_box_bottom(W))
    return "\n".join(lines)


def save_prediction(
    question: str,
    p_final: float,
    p_llm: float,
    p_market: Optional[float],
    llm_result: dict[str, Any],
    evidence: list[EvidenceBlock],
    temporal: Optional[TemporalSignal] = None,
    risk_score: Optional[RiskScore] = None,
) -> Path:
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
        "scenarios": llm_result.get("scenarios", []),
        "llm_confidence": llm_result.get("confidence", ""),
        "key_factors": llm_result.get("key_factors", []),
        "risk_score": risk_score.score if risk_score else None,
        "risk_label": risk_score.label if risk_score else None,
        "temporal_summary": temporal.summary if temporal else None,
        "evidence_quality": eq,
        "evidence_sources": sources_used,
        "resolved": False,
        "outcome": None,
    }

    path = _PREDICTIONS_DIR / filename
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.debug("Saved prediction to %s", path)
    return path
