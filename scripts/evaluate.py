"""
Evaluation pipeline — Brier score, log-loss, AUC, and calibration ECE.

Reads resolved predictions from the lakehouse and produces a report broken
down by domain, blend strategy, and probability bin.

Usage:
  python scripts/evaluate.py
  python scripts/evaluate.py --since 2025-01-01 --by domain
  python main.py evaluate

Output:
  Console table + data/model/eval_report.json
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate")

_REPORT_PATH = Path(__file__).parent.parent / "data" / "model" / "eval_report.json"

# ── Math helpers ──────────────────────────────────────────────────────────────

def _clip(p: float) -> float:
    return max(0.001, min(0.999, float(p)))


def _brier(probs: list[float], outcomes: list[int]) -> float:
    if not probs:
        return float("nan")
    return float(sum((p - y) ** 2 for p, y in zip(probs, outcomes)) / len(probs))


def _log_loss(probs: list[float], outcomes: list[int]) -> float:
    eps = 1e-9
    return float(-sum(
        y * math.log(max(p, eps)) + (1 - y) * math.log(max(1 - p, eps))
        for p, y in zip(probs, outcomes)
    ) / len(probs))


def _brier_skill(brier: float, base_rate: float) -> float:
    clim = base_rate * (1 - base_rate)
    if clim <= 0:
        return 0.0
    return round(1.0 - brier / clim, 4)


def _ece(probs: list[float], outcomes: list[int], n_bins: int = 10) -> float:
    if not probs:
        return float("nan")
    n = len(probs)
    bins = [i / n_bins for i in range(n_bins + 1)]
    err = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        idx = [j for j, p in enumerate(probs) if lo <= p < hi]
        if not idx:
            continue
        mean_p = sum(probs[j] for j in idx) / len(idx)
        mean_y = sum(outcomes[j] for j in idx) / len(idx)
        err += (len(idx) / n) * abs(mean_p - mean_y)
    return round(err, 5)


def _auc(probs: list[float], outcomes: list[int]) -> float:
    """Wilcoxon-Mann-Whitney AUC (works without sklearn)."""
    if not probs or sum(outcomes) == 0 or sum(1 - y for y in outcomes) == 0:
        return float("nan")
    pos = [probs[i] for i, y in enumerate(outcomes) if y == 1]
    neg = [probs[i] for i, y in enumerate(outcomes) if y == 0]
    concordant = sum(1 for p in pos for n in neg if p > n)
    tied       = sum(0.5 for p in pos for n in neg if p == n)
    total = len(pos) * len(neg)
    return round((concordant + tied) / total, 4) if total > 0 else float("nan")


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_resolved(since: Optional[datetime] = None) -> list[dict]:
    """
    Load all resolved predictions from the lakehouse.

    Joins predictions → questions → question_resolutions to get:
      calibrated_prob, p_model_raw, blend_strategy, event_family, outcome
    """
    try:
        from data_layer.db import get_db, init_schema, table_exists
        init_schema()
        if not table_exists("predictions"):
            return []

        db = get_db()
        since_clause = "AND p.predicted_at >= ?" if since else ""
        params = [since.isoformat()] if since else []

        rows = db.execute(
            f"""
            SELECT
                p.calibrated_prob,
                p.p_model_raw,
                p.p_market_raw,
                p.blend_strategy,
                p.model_id,
                p.predicted_at,
                COALESCE(q.event_family, 'unknown')    AS domain,
                COALESCE(q.jurisdiction, '')            AS country,
                COALESCE(p.was_correct, fs.outcome)    AS outcome
            FROM predictions p
            LEFT JOIN questions q        ON p.question_id = q.question_id
            LEFT JOIN feature_snapshots fs ON p.snapshot_id = fs.snapshot_id
            WHERE COALESCE(p.was_correct, fs.outcome) IS NOT NULL
              AND p.calibrated_prob IS NOT NULL
              {since_clause}
            ORDER BY p.predicted_at ASC
            """,
            params,
        ).fetchall()

        cols = ["calibrated_prob", "p_model_raw", "p_market_raw",
                "blend_strategy", "model_id", "predicted_at",
                "domain", "country", "outcome"]
        return [dict(zip(cols, r)) for r in rows]
    except Exception as e:
        logger.warning("_load_resolved failed: %s", e)
        return []


# ── Report sections ───────────────────────────────────────────────────────────

def _section(records: list[dict], label: str) -> dict:
    if not records:
        return {"n": 0, "label": label}
    probs   = [_clip(r["calibrated_prob"]) for r in records]
    outcomes = [int(r["outcome"]) for r in records]
    base    = sum(outcomes) / len(outcomes)
    b       = _brier(probs, outcomes)
    return {
        "label":       label,
        "n":           len(records),
        "base_rate":   round(base, 4),
        "brier":       round(b, 4),
        "brier_skill": _brier_skill(b, base),
        "log_loss":    round(_log_loss(probs, outcomes), 4),
        "ece":         _ece(probs, outcomes),
        "auc":         _auc(probs, outcomes),
    }


def run(
    since: Optional[datetime] = None,
    by: str = "all",
    save: bool = True,
) -> dict:
    records = _load_resolved(since)

    if not records:
        msg = "No resolved predictions found. Seed data with: python scripts/seed_from_markets.py"
        print(msg)
        return {"n": 0, "message": msg}

    logger.info("evaluating %d resolved predictions", len(records))

    report: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "since":        since.isoformat() if since else "all-time",
        "n_total":      len(records),
    }

    # ── Overall ───────────────────────────────────────────────────────────────
    report["overall"] = _section(records, "Overall")

    # ── By domain ─────────────────────────────────────────────────────────────
    domains: dict[str, list] = {}
    for r in records:
        domains.setdefault(r["domain"], []).append(r)
    report["by_domain"] = {
        d: _section(recs, d)
        for d, recs in sorted(domains.items())
    }

    # ── By blend strategy ─────────────────────────────────────────────────────
    strategies: dict[str, list] = {}
    for r in records:
        strat = r.get("blend_strategy") or "model_only"
        strategies.setdefault(strat, []).append(r)
    report["by_strategy"] = {
        s: _section(recs, s)
        for s, recs in sorted(strategies.items())
    }

    # ── Model-only vs market-blended ─────────────────────────────────────────
    model_only = [r for r in records if (r.get("blend_strategy") or "model_only") == "model_only"]
    blended    = [r for r in records if (r.get("blend_strategy") or "model_only") != "model_only"]
    if model_only:
        report["model_only"]  = _section(model_only, "model_only")
    if blended:
        report["blended"]     = _section(blended, "blended")

    # ── Calibration curve (10 bins) ───────────────────────────────────────────
    probs   = [_clip(r["calibrated_prob"]) for r in records]
    outcomes = [int(r["outcome"]) for r in records]
    n_bins  = 10
    bins    = [i / n_bins for i in range(n_bins + 1)]
    calib_rows = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        idx = [j for j, p in enumerate(probs) if lo <= p < hi]
        if not idx:
            continue
        mean_p  = round(sum(probs[j] for j in idx) / len(idx), 4)
        mean_y  = round(sum(outcomes[j] for j in idx) / len(idx), 4)
        calib_rows.append({
            "bin": f"[{lo:.1f},{hi:.1f})",
            "n": len(idx),
            "mean_pred": mean_p,
            "mean_actual": mean_y,
            "gap": round(mean_p - mean_y, 4),
        })
    report["calibration_curve"] = calib_rows

    if save:
        _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        _REPORT_PATH.write_text(json.dumps(report, indent=2))
        logger.info("saved eval report to %s", _REPORT_PATH)

    _print_report(report)
    return report


# ── Pretty print ──────────────────────────────────────────────────────────────

def _fmt(v, default="-") -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return default
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _print_report(r: dict) -> None:
    W = 72

    def row(s): return f"│ {s:<{W}} │"
    def sep(): return "├─" + "─" * W + "─┤"

    lines = [
        "┌─" + "─" * W + "─┐",
        row("  ORACLE EVALUATION REPORT"),
        row(f"  {r.get('since','all-time')} | {r['n_total']} resolved predictions"),
        sep(),
    ]

    # Overall
    ov = r.get("overall", {})
    lines += [
        row("  OVERALL"),
        row(f"  n={ov.get('n',0):>4}  base_rate={_fmt(ov.get('base_rate'))}  "
            f"brier={_fmt(ov.get('brier'))}  skill={_fmt(ov.get('brier_skill'))}"),
        row(f"  log_loss={_fmt(ov.get('log_loss'))}  ece={_fmt(ov.get('ece'))}  "
            f"auc={_fmt(ov.get('auc'))}"),
        sep(),
    ]

    # By domain
    if r.get("by_domain"):
        lines.append(row("  BY DOMAIN"))
        lines.append(row(f"  {'domain':<18} {'n':>5}  {'brier':>7}  {'skill':>7}  {'auc':>7}  {'ece':>7}"))
        for d, s in r["by_domain"].items():
            if s.get("n", 0) == 0:
                continue
            lines.append(row(
                f"  {d:<18} {s['n']:>5}  "
                f"{_fmt(s.get('brier')):>7}  "
                f"{_fmt(s.get('brier_skill')):>7}  "
                f"{_fmt(s.get('auc')):>7}  "
                f"{_fmt(s.get('ece')):>7}"
            ))
        lines.append(sep())

    # By strategy
    if r.get("by_strategy"):
        lines.append(row("  BY BLEND STRATEGY"))
        lines.append(row(f"  {'strategy':<28} {'n':>5}  {'brier':>7}  {'skill':>7}"))
        for s_name, s in r["by_strategy"].items():
            if s.get("n", 0) == 0:
                continue
            lines.append(row(
                f"  {s_name:<28} {s['n']:>5}  "
                f"{_fmt(s.get('brier')):>7}  "
                f"{_fmt(s.get('brier_skill')):>7}"
            ))
        lines.append(sep())

    # Calibration curve
    if r.get("calibration_curve"):
        lines.append(row("  CALIBRATION CURVE  (gap = mean_pred - mean_actual)"))
        lines.append(row(f"  {'bin':<13} {'n':>5}  {'pred':>6}  {'actual':>6}  {'gap':>6}  {'flag':<4}"))
        for c in r["calibration_curve"]:
            flag = "OVER" if c["gap"] > 0.05 else ("UNDER" if c["gap"] < -0.05 else "ok  ")
            lines.append(row(
                f"  {c['bin']:<13} {c['n']:>5}  "
                f"{c['mean_pred']:>6.3f}  {c['mean_actual']:>6.3f}  "
                f"{c['gap']:>+6.3f}  {flag}"
            ))

    lines.append("└─" + "─" * W + "─┘")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate oracle predictions")
    parser.add_argument("--since", help="Only include predictions after this date (YYYY-MM-DD)")
    parser.add_argument("--by", default="all", choices=["all", "domain", "strategy"])
    parser.add_argument("--no-save", action="store_true", help="Don't write eval_report.json")
    args = parser.parse_args()

    since = None
    if args.since:
        try:
            since = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc)
        except ValueError:
            logger.error("invalid --since date: %s", args.since)
            sys.exit(1)

    run(since=since, by=args.by, save=not args.no_save)


if __name__ == "__main__":
    main()
