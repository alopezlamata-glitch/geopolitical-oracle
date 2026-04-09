from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

_RECENT_WINDOW = 7   # days to consider "recent" for anomaly detection
_MIN_POINTS = 5      # minimum data points for meaningful analysis


@dataclass
class TemporalSignal:
    # Volume dynamics
    volume_trend_slope: float        # articles/day change (raw)
    volume_trend_pct: float          # % change from start to end of period
    volume_trend_pvalue: float       # statistical significance
    volume_acceleration: float       # recent slope vs older slope (2nd derivative)
    volume_anomaly_z: float          # z-score: recent vs historical baseline

    # Tone dynamics
    tone_trend_slope: float          # tone units/day (negative = deteriorating)
    tone_recent_mean: float          # avg tone in last 7 days
    tone_historical_mean: float      # avg tone over full period

    # Summary
    is_significant: bool             # volume trend p < 0.05
    risk_label: str                  # "STABLE" | "ELEVATED" | "SURGING" | "COLLAPSING"
    summary: str                     # human-readable one-liner

    def to_prompt_text(self) -> str:
        lines = [
            f"30-day volume trend: {self._volume_description()}",
            f"Trend significance: {'significant (p={:.3f})'.format(self.volume_trend_pvalue) if self.is_significant else 'not significant'}",
            f"Volume anomaly: {self._anomaly_description()}",
            f"Media tone (recent 7d): {self.tone_recent_mean:.2f} | Full period: {self.tone_historical_mean:.2f}",
            f"Tone trajectory: {self._tone_description()}",
            f"Overall signal: {self.risk_label}",
        ]
        return "\n".join(lines)

    def _volume_description(self) -> str:
        pct = self.volume_trend_pct
        acc = self.volume_acceleration
        direction = "UP" if pct > 0 else "DOWN"
        accel = " (accelerating)" if acc > 0 and pct > 0 else " (decelerating)" if acc < 0 and pct > 0 else ""
        return f"{direction} {abs(pct):.0f}%{accel}"

    def _anomaly_description(self) -> str:
        z = self.volume_anomaly_z
        if abs(z) < 1.0:
            return f"normal (z={z:.1f})"
        elif abs(z) < 2.0:
            return f"slightly {'elevated' if z > 0 else 'suppressed'} (z={z:.1f})"
        elif abs(z) < 3.0:
            return f"NOTABLE {'spike' if z > 0 else 'drop'} (z={z:.1f})"
        else:
            return f"EXTREME {'surge' if z > 0 else 'collapse'} (z={z:.1f})"

    def _tone_description(self) -> str:
        slope = self.tone_trend_slope
        if abs(slope) < 0.05:
            return "stable"
        elif slope < 0:
            return f"deteriorating ({slope:.3f}/day)"
        else:
            return f"improving ({slope:+.3f}/day)"


def analyze_temporal(
    volume_series: list[float],
    tone_series: list[float],
) -> Optional[TemporalSignal]:
    """
    Compute temporal dynamics from 30-day time series of article volumes and tones.

    Args:
        volume_series: daily article counts (oldest first)
        tone_series:   daily average tone scores (oldest first, same length)

    Returns:
        TemporalSignal or None if insufficient data
    """
    if len(volume_series) < _MIN_POINTS:
        logger.debug("temporal: insufficient data points (%d)", len(volume_series))
        return None

    vol = np.array(volume_series, dtype=float)
    tone = np.array(tone_series, dtype=float)
    x = np.arange(len(vol))

    # ── Volume trend (linear regression over full period) ────────────────────
    slope_v, intercept_v, r_v, p_v, _ = stats.linregress(x, vol)

    # % change: predicted last vs predicted first
    vol_pct = ((slope_v * x[-1] + intercept_v) - (slope_v * x[0] + intercept_v)) / max(
        abs(slope_v * x[0] + intercept_v), 1
    ) * 100

    # ── Acceleration (recent half slope vs older half slope) ─────────────────
    mid = len(vol) // 2
    if mid >= 3 and (len(vol) - mid) >= 3:
        slope_old, *_ = stats.linregress(x[:mid], vol[:mid])
        slope_new, *_ = stats.linregress(x[mid:], vol[mid:])
        acceleration = float(slope_new - slope_old)
    else:
        acceleration = 0.0

    # ── Anomaly: z-score of recent window vs historical baseline ─────────────
    w = min(_RECENT_WINDOW, len(vol) - 2)
    recent_vol = vol[-w:]
    historical_vol = vol[:-w] if len(vol) > w else vol
    h_mean = float(historical_vol.mean())
    h_std = float(historical_vol.std()) or 1.0
    anomaly_z = float((recent_vol.mean() - h_mean) / h_std)

    # ── Tone trend ───────────────────────────────────────────────────────────
    slope_t, *_ = stats.linregress(x, tone)
    tone_recent = float(tone[-w:].mean())
    tone_historical = float(tone.mean())

    # ── Risk label ───────────────────────────────────────────────────────────
    risk_label = _classify_risk(slope_v, anomaly_z, slope_t)

    summary = (
        f"Volume {'+' if vol_pct > 0 else ''}{vol_pct:.0f}% over period | "
        f"Anomaly z={anomaly_z:.1f} | "
        f"Tone {'↓' if slope_t < -0.05 else '↑' if slope_t > 0.05 else '→'} | "
        f"{risk_label}"
    )

    return TemporalSignal(
        volume_trend_slope=float(slope_v),
        volume_trend_pct=float(vol_pct),
        volume_trend_pvalue=float(p_v),
        volume_acceleration=acceleration,
        volume_anomaly_z=anomaly_z,
        tone_trend_slope=float(slope_t),
        tone_recent_mean=tone_recent,
        tone_historical_mean=tone_historical,
        is_significant=p_v < 0.05,
        risk_label=risk_label,
        summary=summary,
    )


def _classify_risk(slope: float, anomaly_z: float, tone_slope: float) -> str:
    if anomaly_z > 2.5 and slope > 0:
        return "SURGING"
    elif anomaly_z < -2.0:
        return "COLLAPSING"
    elif (slope > 0 and anomaly_z > 1.0) or (tone_slope < -0.1 and anomaly_z > 0.5):
        return "ELEVATED"
    else:
        return "STABLE"
