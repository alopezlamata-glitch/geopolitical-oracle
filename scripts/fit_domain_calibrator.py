"""
Fit per-domain isotonic calibrators and update feature weights for non-conflict domains.

Two operations:

1. ISOTONIC CALIBRATION (always run if data exists)
   Reads training_ready_snapshots WHERE event_family = <domain>.
   Fits sklearn IsotonicRegression on (raw_prob → outcome).
   Saves calibrator to data/model/calibrator_{event_family}.pkl.

2. WEIGHT FITTING (optional, --fit-weights)
   Reads feature_snapshots with labeled outcomes.
   Fits logistic regression coefficients for domain feature columns.
   Updates data/feature_weights.json with learned weights.

Run after auto_resolve.py accumulates labeled examples:
  python scripts/fit_domain_calibrator.py --domain political
  python scripts/fit_domain_calibrator.py --all-domains --fit-weights

Minimum examples: 30 per domain for calibration, 50 for weight fitting.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fit_domain_calibrator")

_MIN_EXAMPLES_CALIBRATION = 30
_MIN_EXAMPLES_WEIGHTS = 50

_DOMAIN_FEATURES = {
    "political": [
        "pol_resignation_signals", "pol_approval_pressure", "pol_coalition_stability",
        "pol_electoral_proximity", "pol_judicial_pressure",
        "avg_polarity_7d", "llm_escalation", "llm_threat_level",
    ],
    "economic": [
        "eco_rate_change_prob", "eco_gdp_momentum", "eco_debt_stress",
        "eco_market_volatility", "eco_policy_uncertainty",
        "avg_polarity_7d", "llm_threat_level",
    ],
    "legal": [
        "leg_arrest_probability", "leg_extradition_risk", "leg_evidence_strength",
        "leg_jurisdictional_support", "leg_precedent_match",
        "llm_escalation", "llm_threat_level",
    ],
    "entertainment": [
        "llm_event_certainty", "llm_threat_level", "llm_escalation",
    ],
}


def _fit_calibrator_for_domain(db, domain: str) -> bool:
    """Fit isotonic calibrator for a single domain. Returns True if successful."""
    rows = db.execute(
        """
        SELECT fs.features_json, qr.outcome
        FROM training_ready_snapshots trs
        JOIN feature_snapshots fs ON fs.snapshot_id = trs.snapshot_id
        JOIN question_resolutions qr ON qr.question_id = trs.question_id
        JOIN questions q ON q.question_id = trs.question_id
        WHERE q.event_family = ?
          AND qr.is_ambiguous = FALSE
          AND qr.outcome IN (0, 1)
        """,
        [domain],
    ).fetchall()

    if len(rows) < _MIN_EXAMPLES_CALIBRATION:
        logger.info(
            "fit_calibrator[%s]: only %d examples (need %d) — skipping",
            domain, len(rows), _MIN_EXAMPLES_CALIBRATION,
        )
        return False

    logger.info("fit_calibrator[%s]: fitting on %d examples", domain, len(rows))

    # We need the raw probability (pre-calibration) and the true outcome.
    # Since base_rate_predictor stores raw_prob in features_json... not yet.
    # Fallback: use a proxy feature score if raw_prob not in features_json.
    # TODO: store raw_prob in feature_snapshots once base_rate_predictor is running.
    raw_probs = []
    outcomes = []
    for features_json, outcome in rows:
        try:
            feats = json.loads(features_json) if isinstance(features_json, str) else {}
        except Exception:
            feats = {}
        # Use stored raw_prob if present, else compute a proxy from features
        raw_p = feats.get("_raw_prob")
        if raw_p is None:
            # Simple proxy: mean of nonzero domain features, defaulting to 0.5
            domain_cols = _DOMAIN_FEATURES.get(domain, [])
            vals = [feats.get(c, 0.0) for c in domain_cols if feats.get(c, 0.0) > 0]
            raw_p = float(sum(vals) / len(vals)) if vals else 0.5
        raw_probs.append(max(0.01, min(0.99, float(raw_p))))
        outcomes.append(int(outcome))

    try:
        from sklearn.isotonic import IsotonicRegression
        cal = IsotonicRegression(out_of_bounds="clip")
        cal.fit(raw_probs, outcomes)

        from model.domain_calibrator import save_domain_calibrator
        path = save_domain_calibrator(cal, domain)
        logger.info("fit_calibrator[%s]: saved to %s", domain, path)
        return True
    except Exception as e:
        logger.error("fit_calibrator[%s]: fitting failed: %s", domain, e)
        return False


def _fit_weights_for_domain(db, domain: str) -> bool:
    """Fit logistic regression weights for domain features. Updates feature_weights.json."""
    feature_cols = _DOMAIN_FEATURES.get(domain, [])
    if not feature_cols:
        logger.info("fit_weights[%s]: no feature columns defined", domain)
        return False

    rows = db.execute(
        """
        SELECT fs.features_json, qr.outcome
        FROM training_ready_snapshots trs
        JOIN feature_snapshots fs ON fs.snapshot_id = trs.snapshot_id
        JOIN question_resolutions qr ON qr.question_id = trs.question_id
        JOIN questions q ON q.question_id = trs.question_id
        WHERE q.event_family = ?
          AND qr.is_ambiguous = FALSE
          AND qr.outcome IN (0, 1)
        """,
        [domain],
    ).fetchall()

    if len(rows) < _MIN_EXAMPLES_WEIGHTS:
        logger.info(
            "fit_weights[%s]: only %d examples (need %d) — skipping",
            domain, len(rows), _MIN_EXAMPLES_WEIGHTS,
        )
        return False

    logger.info("fit_weights[%s]: fitting logistic regression on %d examples", domain, len(rows))

    import numpy as np
    X = []
    y = []
    for features_json, outcome in rows:
        try:
            feats = json.loads(features_json) if isinstance(features_json, str) else {}
        except Exception:
            feats = {}
        row = [feats.get(c, 0.0) for c in feature_cols]
        X.append(row)
        y.append(int(outcome))

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    try:
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(
            C=1.0,
            max_iter=500,
            solver="lbfgs",
            class_weight="balanced",
        )
        lr.fit(X, y)
        coefs = lr.coef_[0]
        logger.info("fit_weights[%s]: coefs = %s", domain, dict(zip(feature_cols, coefs.round(3))))

        # Update feature_weights.json
        weights_path = Path(__file__).parent.parent / "data" / "feature_weights.json"
        weights = json.loads(weights_path.read_text())
        domain_weights = weights.setdefault(domain, {})
        for fname, coef in zip(feature_cols, coefs):
            domain_weights[fname] = round(float(coef), 4)
        weights_path.write_text(json.dumps(weights, indent=2))
        logger.info("fit_weights[%s]: updated feature_weights.json", domain)
        return True
    except Exception as e:
        logger.error("fit_weights[%s]: fitting failed: %s", domain, e)
        return False


def run(domains: list[str], fit_weights: bool = False) -> None:
    from data_layer.db import get_db, init_schema
    init_schema()
    db = get_db()

    for domain in domains:
        logger.info("--- Processing domain: %s ---", domain)
        _fit_calibrator_for_domain(db, domain)
        if fit_weights:
            _fit_weights_for_domain(db, domain)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit domain calibrators and/or feature weights")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--domain", type=str, help="Single domain to fit (political/economic/legal/entertainment)")
    group.add_argument("--all-domains", action="store_true", help="Fit all known non-conflict domains")
    parser.add_argument(
        "--fit-weights", action="store_true",
        help="Also fit logistic regression weights (requires >= 50 examples per domain)",
    )
    args = parser.parse_args()

    if args.all_domains:
        domains = list(_DOMAIN_FEATURES.keys())
    else:
        domains = [args.domain]

    run(domains=domains, fit_weights=args.fit_weights)


if __name__ == "__main__":
    main()
