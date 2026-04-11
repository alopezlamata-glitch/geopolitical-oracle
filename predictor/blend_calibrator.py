"""
Phase B: Learn blend coefficients (alpha, beta, bias) from resolved history.

Problem: the fixed-weight formula in market_prior.py is a prior belief about
how much to trust market signals vs the model. Phase B replaces it with data.

Formula learned:
    logit(p_final) = alpha * logit(p_model) + beta * logit(p_market) + bias

Fitted on resolved predictions where both p_model_raw and p_market_raw are known.

Outputs:
    data/model/blend_weights.json   — loaded at inference time by market_prior.py

Gate: requires >= MIN_RESOLVED_WITH_MARKET examples. Below that, fixed weights
are safer (less variance) and the calibrator prints a diagnostic instead.

Diagnostics always computed regardless of gate (on all resolved examples):
    - Brier score: model_only vs fixed_weight_blend vs learned_blend
    - Performance by blend_strategy (model_only / model_plus_market / market_dominant)
    - n_resolved, n_with_market, base_rate

Usage:
    python -c "from predictor.blend_calibrator import run_blend_calibration; run_blend_calibration()"
    python main.py blend-calibrate
"""
from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_MODEL_DIR = Path(__file__).parent.parent / "data" / "model"
_BLEND_WEIGHTS_PATH = _MODEL_DIR / "blend_weights.json"

# Minimum resolved examples with a market signal to fit the model
MIN_RESOLVED_WITH_MARKET: int = 20
# Current fixed-weight fallback for comparison
_FIXED_W_MARKET: float = 0.35   # representative of logodds_v1 mid-range weight

BLEND_CALIBRATOR_VERSION: str = "logodds_learned_v1"


# ── Data fetching ──────────────────────────────────────────────────────────────

def _fetch_resolved_predictions() -> list[dict]:
    """
    Query the lakehouse for all resolved predictions.

    A prediction is "resolved" when:
      - predictions.was_correct IS NOT NULL (updated by cmd_label)
      - OR predictions.brier_component IS NOT NULL
      - OR feature_snapshots.outcome IS NOT NULL (joined via snapshot_id)

    Returns list of dicts with: p_model_raw, p_market_raw, blend_strategy,
    market_weight, outcome, calibrated_prob, predicted_at.
    """
    try:
        from data_layer.db import get_db, table_exists
    except ImportError:
        logger.warning("blend_calibrator: data_layer not available")
        return []

    if not table_exists("predictions"):
        return []

    db = get_db()
    try:
        rows = db.execute("""
            SELECT
                p.p_model_raw,
                p.p_market_raw,
                p.market_weight,
                p.blend_strategy,
                p.calibrated_prob,
                p.predicted_at,
                COALESCE(p.was_correct, fs.outcome) AS outcome
            FROM predictions p
            LEFT JOIN feature_snapshots fs
                ON p.snapshot_id = fs.snapshot_id
            WHERE COALESCE(p.was_correct, fs.outcome) IS NOT NULL
              AND p.p_model_raw IS NOT NULL
            ORDER BY p.predicted_at ASC
        """).fetchall()
    except Exception as e:
        logger.warning("blend_calibrator: query failed: %s", e)
        return []

    columns = ["p_model_raw", "p_market_raw", "market_weight",
               "blend_strategy", "calibrated_prob", "predicted_at", "outcome"]
    return [dict(zip(columns, row)) for row in rows]


# ── Math helpers ───────────────────────────────────────────────────────────────

def _clip(p: float) -> float:
    return max(0.001, min(0.999, float(p)))


def _logit(p: float) -> float:
    p = _clip(p)
    return math.log(p / (1.0 - p))


def _sigmoid(lo: float) -> float:
    return 1.0 / (1.0 + math.exp(-lo))


def _brier(probs: list[float], outcomes: list[int]) -> float:
    if not probs:
        return float("nan")
    return float(np.mean([(p - y) ** 2 for p, y in zip(probs, outcomes)]))


def _log_loss(probs: list[float], outcomes: list[int]) -> float:
    eps = 1e-7
    return float(-np.mean([
        y * math.log(max(p, eps)) + (1 - y) * math.log(max(1 - p, eps))
        for p, y in zip(probs, outcomes)
    ]))


def _apply_fixed_blend(p_model: float, p_market: float, w: float = _FIXED_W_MARKET) -> float:
    """Fixed log-odds blend at weight w — used as baseline."""
    lo = (1.0 - w) * _logit(p_model) + w * _logit(p_market)
    return _clip(_sigmoid(lo))


def _apply_learned_blend(
    p_model: float, p_market: float, alpha: float, beta: float, bias: float
) -> float:
    """Learned log-odds blend."""
    lo = alpha * _logit(p_model) + beta * _logit(p_market) + bias
    return _clip(_sigmoid(lo))


# ── Calibration fitting ────────────────────────────────────────────────────────

def fit_blend_weights(
    records: list[dict],
) -> Optional[dict]:
    """
    Fit logistic regression: (logit(p_model), logit(p_market)) → outcome.

    Returns coefficient dict or None if fitting fails.

    The logistic regression models:
        P(outcome=1) = σ(alpha * logit(p_model) + beta * logit(p_market) + bias)

    Regularization: C=1.0 (medium). Lower C = more shrinkage toward 0 (prior of
    no signal); raise C if you have large N and high-quality market data.
    """
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        logger.error("blend_calibrator: scikit-learn not installed")
        return None

    with_market = [r for r in records if r["p_market_raw"] is not None]
    if len(with_market) < MIN_RESOLVED_WITH_MARKET:
        logger.info(
            "blend_calibrator: only %d resolved examples with market signal "
            "(need %d) — skipping fit",
            len(with_market), MIN_RESOLVED_WITH_MARKET,
        )
        return None

    X = np.array([
        [_logit(r["p_model_raw"]), _logit(r["p_market_raw"])]
        for r in with_market
    ], dtype=np.float32)
    y = np.array([int(r["outcome"]) for r in with_market])

    try:
        lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500, fit_intercept=True)
        lr.fit(X, y)
    except Exception as e:
        logger.error("blend_calibrator: fit failed: %s", e)
        return None

    alpha = float(lr.coef_[0][0])
    beta  = float(lr.coef_[0][1])
    bias  = float(lr.intercept_[0])

    logger.info(
        "blend_calibrator: alpha=%.4f  beta=%.4f  bias=%.4f  n=%d",
        alpha, beta, bias, len(with_market),
    )
    return {"alpha": alpha, "beta": beta, "bias": bias, "n_training": len(with_market)}


# ── Diagnostics ────────────────────────────────────────────────────────────────

def _compute_diagnostics(records: list[dict], fit: Optional[dict]) -> dict:
    """
    Compare three strategies on all resolved examples with p_model_raw:
      1. model_only         — use p_model_raw as-is
      2. fixed_blend        — fixed log-odds blend at _FIXED_W_MARKET
      3. learned_blend      — use fitted alpha/beta/bias (if available)

    Also breaks down Brier by blend_strategy label.
    """
    diag: dict = {}
    if not records:
        return diag

    outcomes = [int(r["outcome"]) for r in records]
    base_rate = float(np.mean(outcomes))
    diag["n_resolved"] = len(records)
    diag["base_rate"] = round(base_rate, 4)

    # Brier climatology (always predict base_rate)
    brier_clim = _brier([base_rate] * len(records), outcomes)
    diag["brier_climatology"] = round(brier_clim, 4)

    # Strategy 1: model_only
    p_model_list = [_clip(r["p_model_raw"]) for r in records]
    b_model = _brier(p_model_list, outcomes)
    diag["brier_model_only"] = round(b_model, 4)
    diag["logloss_model_only"] = round(_log_loss(p_model_list, outcomes), 4)
    diag["brier_skill_model_only"] = round(1.0 - b_model / brier_clim, 4) if brier_clim > 0 else 0.0

    # Strategy 2: fixed blend (only for rows with market signal)
    with_market = [r for r in records if r["p_market_raw"] is not None]
    diag["n_with_market"] = len(with_market)
    if with_market:
        p_fixed = [_apply_fixed_blend(r["p_model_raw"], r["p_market_raw"]) for r in with_market]
        y_mkt = [int(r["outcome"]) for r in with_market]
        b_fixed = _brier(p_fixed, y_mkt)
        b_model_mkt = _brier([_clip(r["p_model_raw"]) for r in with_market], y_mkt)
        diag["brier_fixed_blend_on_market_subset"] = round(b_fixed, 4)
        diag["brier_model_only_on_market_subset"] = round(b_model_mkt, 4)
        diag["blend_improvement_vs_model"] = round(b_model_mkt - b_fixed, 4)
        diag["logloss_fixed_blend"] = round(_log_loss(p_fixed, y_mkt), 4)

    # Strategy 3: learned blend
    if fit and with_market:
        alpha, beta, bias = fit["alpha"], fit["beta"], fit["bias"]
        p_learned = [
            _apply_learned_blend(r["p_model_raw"], r["p_market_raw"], alpha, beta, bias)
            for r in with_market
        ]
        b_learned = _brier(p_learned, y_mkt)
        diag["brier_learned_blend"] = round(b_learned, 4)
        diag["logloss_learned_blend"] = round(_log_loss(p_learned, y_mkt), 4)
        diag["learned_vs_fixed_improvement"] = round(b_fixed - b_learned, 4)

    # Breakdown by blend_strategy
    strategies = {}
    for r in records:
        strat = r.get("blend_strategy") or "model_only"
        strategies.setdefault(strat, {"probs": [], "outcomes": []})
        strategies[strat]["probs"].append(_clip(r["calibrated_prob"] or r["p_model_raw"]))
        strategies[strat]["outcomes"].append(int(r["outcome"]))
    diag["by_strategy"] = {
        strat: {
            "n": len(v["outcomes"]),
            "brier": round(_brier(v["probs"], v["outcomes"]), 4),
            "base_rate": round(float(np.mean(v["outcomes"])), 4),
        }
        for strat, v in strategies.items()
    }

    return diag


# ── Public API ─────────────────────────────────────────────────────────────────

def run_blend_calibration(verbose: bool = True) -> dict:
    """
    Main entry point. Fetches resolved history, fits coefficients, saves weights.

    Returns a result dict with:
      - fit: fitted coefficients (None if not enough data)
      - diagnostics: Brier/log-loss comparison across strategies
      - saved: True if blend_weights.json was written
      - message: human-readable summary
    """
    records = _fetch_resolved_predictions()

    if not records:
        msg = (
            f"No resolved predictions found in lakehouse.\n"
            f"Label predictions with: python main.py label --prediction-id <id> --outcome yes|no\n"
            f"Need >= {MIN_RESOLVED_WITH_MARKET} resolved examples with market signal to fit."
        )
        if verbose:
            print(msg)
        return {"fit": None, "diagnostics": {}, "saved": False, "message": msg}

    fit = fit_blend_weights(records)
    diag = _compute_diagnostics(records, fit)
    saved = False

    if fit is not None:
        weights = {
            **fit,
            "blend_calibrator_version": BLEND_CALIBRATOR_VERSION,
            "fitted_at": datetime.now(timezone.utc).isoformat(),
            "diagnostics": diag,
        }
        _MODEL_DIR.mkdir(parents=True, exist_ok=True)
        _BLEND_WEIGHTS_PATH.write_text(json.dumps(weights, indent=2))
        saved = True
        logger.info("blend_calibrator: saved weights to %s", _BLEND_WEIGHTS_PATH)
        msg = (
            f"Blend weights fitted and saved.\n"
            f"  alpha={fit['alpha']:.4f}  beta={fit['beta']:.4f}  bias={fit['bias']:.4f}\n"
            f"  n_training={fit['n_training']}"
        )
    else:
        n_mkt = diag.get("n_with_market", 0)
        msg = (
            f"Not enough data to fit blend weights yet.\n"
            f"  Resolved total      : {diag.get('n_resolved', 0)}\n"
            f"  With market signal  : {n_mkt} (need {MIN_RESOLVED_WITH_MARKET})\n"
            f"  Fixed weights remain active (logodds_v1)."
        )

    if verbose:
        _print_diagnostics(diag, fit, msg)

    return {"fit": fit, "diagnostics": diag, "saved": saved, "message": msg}


def load_blend_weights() -> Optional[dict]:
    """
    Load saved blend weights. Returns None if not available (use fixed weights).

    The returned dict has: alpha, beta, bias, blend_calibrator_version, fitted_at.
    """
    if not _BLEND_WEIGHTS_PATH.exists():
        return None
    try:
        data = json.loads(_BLEND_WEIGHTS_PATH.read_text())
        # Basic validation
        if all(k in data for k in ("alpha", "beta", "bias")):
            return data
        logger.warning("blend_calibrator: invalid blend_weights.json — missing keys")
        return None
    except Exception as e:
        logger.warning("blend_calibrator: failed to load blend_weights.json: %s", e)
        return None


def _print_diagnostics(diag: dict, fit: Optional[dict], msg: str) -> None:
    W = 58
    def row(s): return f"│ {s:<{W}} │"

    lines = [
        "┌─" + "─" * W + "─┐",
        row("  BLEND CALIBRATION REPORT"),
        row("  " + "─" * (W - 2)),
    ]

    n_res = diag.get("n_resolved", 0)
    n_mkt = diag.get("n_with_market", 0)
    br = diag.get("base_rate", "?")
    lines += [
        row(f"  Resolved predictions     : {n_res}"),
        row(f"  With market signal       : {n_mkt}  (need {MIN_RESOLVED_WITH_MARKET} to fit)"),
        row(f"  Base rate                : {br}"),
        row(""),
    ]

    if "brier_model_only" in diag:
        lines += [
            row(f"  Brier (climatology)      : {diag['brier_climatology']:.4f}"),
            row(f"  Brier (model only)       : {diag['brier_model_only']:.4f}"
                f"  skill={diag.get('brier_skill_model_only', '?')}"),
        ]

    if "brier_fixed_blend_on_market_subset" in diag:
        imp = diag.get("blend_improvement_vs_model", 0)
        sign = "+" if imp >= 0 else ""
        lines += [
            row(""),
            row(f"  On {n_mkt} examples with market signal:"),
            row(f"    model only             : {diag['brier_model_only_on_market_subset']:.4f}"),
            row(f"    fixed blend (w={_FIXED_W_MARKET:.2f})  : {diag['brier_fixed_blend_on_market_subset']:.4f}"
                f"  ({sign}{imp:.4f})"),
        ]
        if "brier_learned_blend" in diag:
            lv_f = diag.get("learned_vs_fixed_improvement", 0)
            sign2 = "+" if lv_f >= 0 else ""
            lines.append(
                row(f"    learned blend          : {diag['brier_learned_blend']:.4f}"
                    f"  ({sign2}{lv_f:.4f} vs fixed)")
            )

    if fit:
        lines += [
            row(""),
            row(f"  Learned:  alpha={fit['alpha']:.4f}  beta={fit['beta']:.4f}"
                f"  bias={fit['bias']:.4f}"),
            row(f"  Version:  {BLEND_CALIBRATOR_VERSION}"),
            row("  Saved  :  data/model/blend_weights.json"),
        ]
    else:
        lines.append(row(f"  Status : fixed weights active (logodds_v1)"))

    if diag.get("by_strategy"):
        lines.append(row(""))
        lines.append(row("  By blend strategy:"))
        for strat, stats in diag["by_strategy"].items():
            lines.append(row(f"    {strat:<26}  n={stats['n']:4d}  brier={stats['brier']:.4f}"))

    lines.append("└─" + "─" * W + "─┘")
    print("\n".join(lines))
