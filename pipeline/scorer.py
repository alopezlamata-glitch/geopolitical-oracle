from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .temporal import TemporalSignal

logger = logging.getLogger(__name__)


@dataclass
class RiskScore:
    score: float                    # 0–100
    label: str                      # LOW / MODERATE / HIGH / CRITICAL
    breakdown: dict[str, float]     # contribution per signal
    explanation: str                # one-line summary for prompt

    def to_prompt_text(self) -> str:
        parts = [f"{k}: {v:+.1f}" for k, v in self.breakdown.items() if abs(v) > 0.1]
        return (
            f"Risk score: {self.score:.0f}/100 ({self.label})\n"
            f"Drivers: {' | '.join(parts) if parts else 'baseline only'}"
        )


def compute_risk_score(
    temporal: Optional[TemporalSignal],
    gdelt_tone: Optional[float],
    gdelt_article_count: Optional[int],
    rss_article_count: Optional[int],
    acled_fatalities: Optional[int],
    acled_event_count: Optional[int],
) -> RiskScore:
    """
    Aggregate multi-source signals into a 0–100 risk score.

    Score interpretation:
      0–25   → LOW      (little evidence of unusual activity)
      25–50  → MODERATE (some signals, mixed picture)
      50–75  → HIGH     (clear signals of elevated activity)
      75–100 → CRITICAL (extreme / anomalous activity)

    NOTE: A higher risk score means more geopolitical activity/tension,
    which increases the probability of action-oriented events (conflict, elections,
    military exercises). For stability-oriented questions, the caller may invert.
    """
    score = 50.0   # neutral baseline — no strong signal either way
    breakdown: dict[str, float] = {}

    # ── GDELT tone: negative tone = more conflict/tension ────────────────────
    if gdelt_tone is not None:
        # tone range roughly -10 to +10 in practice; map to ±20 score points
        contribution = float(np.clip(-gdelt_tone * 2.0, -20, 20))
        score += contribution
        breakdown["media_tone"] = contribution

    # ── Article volume: proxy for salience ───────────────────────────────────
    if gdelt_article_count is not None:
        # 0 articles → -5, 20 articles → 0, 50+ articles → +10
        contribution = float(np.clip((gdelt_article_count - 20) * 0.5, -5, 10))
        score += contribution
        breakdown["article_volume"] = contribution

    # ── RSS hit count: indicates active news coverage ────────────────────────
    if rss_article_count is not None:
        contribution = float(np.clip(rss_article_count * 1.5, 0, 10))
        score += contribution
        breakdown["rss_coverage"] = contribution

    # ── Temporal dynamics ────────────────────────────────────────────────────
    if temporal is not None:
        # Volume trend: surging coverage = elevated risk
        trend_contribution = float(np.clip(temporal.volume_trend_slope * 0.3, -10, 15))
        score += trend_contribution
        breakdown["volume_trend"] = trend_contribution

        # Anomaly detection: statistically unusual activity
        anomaly_contribution = float(np.clip(temporal.volume_anomaly_z * 3.0, -5, 12))
        score += anomaly_contribution
        breakdown["anomaly_signal"] = anomaly_contribution

        # Tone deterioration: worsening sentiment = higher risk
        tone_trend_contribution = float(np.clip(-temporal.tone_trend_slope * 10.0, -8, 8))
        score += tone_trend_contribution
        breakdown["tone_trajectory"] = tone_trend_contribution

        # Acceleration: rapidly accelerating coverage = compounding risk
        if temporal.volume_acceleration > 0:
            accel_contribution = float(np.clip(temporal.volume_acceleration * 0.2, 0, 8))
            score += accel_contribution
            breakdown["acceleration"] = accel_contribution

    # ── ACLED conflict data ───────────────────────────────────────────────────
    if acled_fatalities is not None and acled_fatalities > 0:
        contribution = float(np.clip(np.log1p(acled_fatalities) * 3, 0, 20))
        score += contribution
        breakdown["conflict_fatalities"] = contribution

    if acled_event_count is not None and acled_event_count > 0:
        contribution = float(np.clip(acled_event_count * 0.5, 0, 10))
        score += contribution
        breakdown["conflict_events"] = contribution

    score = float(np.clip(score, 0, 100))

    # ── Label ────────────────────────────────────────────────────────────────
    if score < 25:
        label = "LOW"
    elif score < 50:
        label = "MODERATE"
    elif score < 75:
        label = "HIGH"
    else:
        label = "CRITICAL"

    top_drivers = sorted(breakdown.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
    driver_str = ", ".join(f"{k} ({v:+.0f})" for k, v in top_drivers)
    explanation = f"{score:.0f}/100 [{label}] — top drivers: {driver_str or 'none'}"

    return RiskScore(score=score, label=label, breakdown=breakdown, explanation=explanation)
