"""
Session-level coherence tracker for prediction consistency.

Problems it solves:
  1. Negation violation:  P(war) + P(peace) > 1 for same entity
  2. Entailment gap:      P(Russia invades Poland) high but P(NATO Article 5) low
  3. Cross-entity drift:  Ukraine prediction ignores Russia's military buildup

Coherence is maintained at two levels:
  A. Session cache (in-process): fast, within a single Python process
  B. DuckDB history (persistent): checks against predictions from the last 24h

Architecture:
  check(entity, predicate, deadline, probability)
    → returns CoherenceReport with violations and suggested adjustments

  register(entity, predicate, deadline, probability, prediction_id)
    → stores prediction in session cache + flags to DuckDB

The coherence layer is advisory, not corrective. It never changes the
model's probability — it annotates the output with consistency flags
so the user (and future models) can reason about contradictions.
"""
from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# In-process session cache: (entity_name, predicate, deadline_str) → probability
_session_cache: dict[tuple[str, str, str], float] = {}
_session_ids: dict[tuple[str, str, str], str] = {}

# Known negation pairs: if we predict P(A) we can infer P(¬A) = 1 - P(A)
_NEGATION_PAIRS: dict[str, str] = {
    "military_escalation":   "ceasefire",
    "ceasefire":             "military_escalation",
    "resign":                "survive_politically",
    "survive_politically":   "resign",
    "default":               "debt_repayment",
    "debt_repayment":        "default",
    "coup":                  "political_stability",
    "political_stability":   "coup",
    "invasion":              "peace",
    "peace":                 "invasion",
    "arrest":                "avoid_arrest",
    "avoid_arrest":          "arrest",
}

# Known entailment links: if P(A) is high, P(B) should also be elevated
_ENTAILMENT_LINKS: list[dict] = [
    {"if_entity": "*",       "if_predicate": "invasion",          "then_entity": "*",       "then_predicate": "military_escalation", "min_ratio": 0.7},
    {"if_entity": "Russia",  "if_predicate": "military_escalation","then_entity": "Ukraine", "then_predicate": "military_escalation", "min_ratio": 0.6},
    {"if_entity": "China",   "if_predicate": "military_escalation","then_entity": "Taiwan",  "then_predicate": "military_escalation", "min_ratio": 0.5},
    {"if_entity": "*",       "if_predicate": "coup",               "then_entity": "*",       "then_predicate": "political_instability","min_ratio": 0.8},
    {"if_entity": "*",       "if_predicate": "default",            "then_entity": "*",       "then_predicate": "eco_debt_stress",    "min_ratio": 0.5},
]


@dataclass
class CoherenceViolation:
    violation_type: str           # 'negation' | 'entailment' | 'drift'
    description: str
    severity: str                 # 'warning' | 'error'
    related_entity: Optional[str] = None
    related_predicate: Optional[str] = None
    related_probability: Optional[float] = None
    suggested_range: Optional[tuple[float, float]] = None


@dataclass
class CoherenceReport:
    entity_name: str
    predicate: str
    probability: float
    deadline: Optional[date]

    violations: list[CoherenceViolation] = field(default_factory=list)
    implications: list[dict] = field(default_factory=list)
    is_consistent: bool = True
    consistency_score: float = 1.0  # 1.0 = fully consistent, 0.0 = deeply contradictory

    def summary(self) -> str:
        if self.is_consistent:
            return f"consistent (score={self.consistency_score:.2f})"
        parts = [f"{v.severity}: {v.description}" for v in self.violations]
        return " | ".join(parts)


def _cache_key(entity: str, predicate: str, deadline: Optional[date]) -> tuple:
    return (entity.lower(), predicate.lower(), str(deadline) if deadline else "none")


def register(
    entity_name: str,
    predicate: str,
    probability: float,
    deadline: Optional[date] = None,
    prediction_id: Optional[str] = None,
) -> None:
    """Store a prediction in the session cache for future consistency checks."""
    key = _cache_key(entity_name, predicate, deadline)
    _session_cache[key] = probability
    if prediction_id:
        _session_ids[key] = prediction_id

    # Also register implied negation
    neg_predicate = _NEGATION_PAIRS.get(predicate)
    if neg_predicate:
        neg_key = _cache_key(entity_name, neg_predicate, deadline)
        if neg_key not in _session_cache:
            _session_cache[neg_key] = round(1.0 - probability, 4)

    logger.debug("coherence: registered %s/%s p=%.3f", entity_name, predicate, probability)


def check(
    entity_name: str,
    predicate: str,
    probability: float,
    deadline: Optional[date] = None,
    event_family: Optional[str] = None,
) -> CoherenceReport:
    """
    Check probability for consistency against session cache and recent DB history.
    Returns a CoherenceReport with any violations found.
    """
    report = CoherenceReport(
        entity_name=entity_name,
        predicate=predicate,
        probability=probability,
        deadline=deadline,
    )

    violations: list[CoherenceViolation] = []
    penalty = 0.0

    # ── 1. Negation check ────────────────────────────────────────────────────
    neg_pred = _NEGATION_PAIRS.get(predicate)
    if neg_pred:
        neg_key = _cache_key(entity_name, neg_pred, deadline)
        neg_p = _session_cache.get(neg_key)
        if neg_p is not None:
            total = probability + neg_p
            if total > 1.05:  # allow small slack for rounding
                gap = total - 1.0
                violations.append(CoherenceViolation(
                    violation_type="negation",
                    description=(
                        f"P({predicate})={probability:.3f} + P({neg_pred})={neg_p:.3f} "
                        f"= {total:.3f} > 1.0 (gap={gap:.3f})"
                    ),
                    severity="error" if gap > 0.15 else "warning",
                    related_predicate=neg_pred,
                    related_probability=neg_p,
                    suggested_range=(
                        max(0.01, probability - gap / 2),
                        min(0.99, probability + gap / 2),
                    ),
                ))
                penalty += gap * 0.5  # each 0.10 excess → -0.05 score

    # ── 2. Entailment check ──────────────────────────────────────────────────
    for link in _ENTAILMENT_LINKS:
        if_ent  = link["if_entity"]
        if_pred = link["if_predicate"]
        then_ent  = link["then_entity"]
        then_pred = link["then_predicate"]
        min_ratio = link["min_ratio"]

        # Does this link apply to the current prediction?
        if if_ent != "*" and if_ent.lower() != entity_name.lower():
            continue
        if if_pred.lower() != predicate.lower():
            continue

        # Check if the implied prediction exists in session
        implied_entity = entity_name if then_ent == "*" else then_ent
        implied_key = _cache_key(implied_entity, then_pred, deadline)
        implied_p = _session_cache.get(implied_key)
        if implied_p is None:
            continue

        # High P(A) should imply elevated P(B)
        expected_min = probability * min_ratio
        if probability > 0.6 and implied_p < expected_min:
            gap = expected_min - implied_p
            violations.append(CoherenceViolation(
                violation_type="entailment",
                description=(
                    f"P({entity_name}/{predicate})={probability:.3f} implies "
                    f"P({implied_entity}/{then_pred}) >= {expected_min:.3f} "
                    f"but got {implied_p:.3f} (gap={gap:.3f})"
                ),
                severity="warning",
                related_entity=implied_entity,
                related_predicate=then_pred,
                related_probability=implied_p,
                suggested_range=(expected_min, min(0.99, implied_p + gap)),
            ))
            penalty += gap * 0.2

    # ── 3. Cross-entity drift check (via DB recent predictions) ─────────────
    try:
        db_violations = _check_db_consistency(entity_name, predicate, probability, deadline)
        violations.extend(db_violations)
        penalty += len(db_violations) * 0.05
    except Exception as e:
        logger.debug("coherence DB check skipped: %s", e)

    # ── 4. Causal implications (informational, not a violation) ──────────────
    try:
        from world_state.entity_graph import get_causal_implications
        implications = get_causal_implications(entity_name, predicate, probability)
        report.implications = implications
    except Exception:
        pass

    # ── Assemble report ───────────────────────────────────────────────────────
    report.violations = violations
    report.is_consistent = len([v for v in violations if v.severity == "error"]) == 0
    report.consistency_score = max(0.0, round(1.0 - penalty, 3))

    if violations:
        logger.info(
            "coherence[%s/%s p=%.3f]: %d violation(s) score=%.2f",
            entity_name, predicate, probability,
            len(violations), report.consistency_score,
        )

    return report


def _check_db_consistency(
    entity_name: str,
    predicate: str,
    probability: float,
    deadline: Optional[date],
) -> list[CoherenceViolation]:
    """
    Check against DuckDB predictions from the last 24h for the same entity.
    Flags if a recent prediction for the negated predicate contradicts this one.
    """
    violations = []
    neg_pred = _NEGATION_PAIRS.get(predicate)
    if not neg_pred:
        return violations

    try:
        from data_layer.db import get_db, table_exists
        if not table_exists("predictions") or not table_exists("questions"):
            return violations

        db = get_db()
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)

        rows = db.execute("""
            SELECT p.calibrated_prob, q.predicate, q.subject
            FROM predictions p
            JOIN questions q ON p.question_id = q.question_id
            WHERE q.subject ILIKE ?
              AND q.predicate = ?
              AND p.predicted_at >= ?
            ORDER BY p.predicted_at DESC
            LIMIT 3
        """, [entity_name, neg_pred, cutoff]).fetchall()

        for (neg_p, neg_predicate, subject) in rows:
            if neg_p is None:
                continue
            total = probability + float(neg_p)
            if total > 1.10:
                violations.append(CoherenceViolation(
                    violation_type="negation",
                    description=(
                        f"DB: recent P({subject}/{neg_predicate})={neg_p:.3f} "
                        f"+ current P={probability:.3f} = {total:.3f} > 1.0"
                    ),
                    severity="warning",
                    related_entity=subject,
                    related_predicate=neg_predicate,
                    related_probability=float(neg_p),
                ))
    except Exception as e:
        logger.debug("_check_db_consistency failed: %s", e)

    return violations


def correct_probability(probability: float, report: CoherenceReport) -> float:
    """
    Soft auto-correction: nudge `probability` toward coherence.

    Strategy per violation type:
      negation:   split the excess equally → nudge DOWN by half the gap
      entailment: implied probability too low → nudge UP by half the gap

    Never moves by more than MAX_CORRECTION = 0.12 in total.
    Correction is weighted by severity (error → 0.5 scale, warning → 0.25).

    Returns corrected probability in [0.01, 0.99].
    """
    MAX_CORRECTION = 0.12

    total_nudge = 0.0
    for v in report.violations:
        if v.suggested_range is None:
            continue

        lo, hi = v.suggested_range
        midpoint = (lo + hi) / 2.0
        raw_nudge = midpoint - probability
        scale = 0.5 if v.severity == "error" else 0.25
        total_nudge += raw_nudge * scale

    total_nudge = max(-MAX_CORRECTION, min(MAX_CORRECTION, total_nudge))

    if abs(total_nudge) > 0.005:
        logger.debug(
            "coherence: auto-correct %s/%s %.3f → %.3f (nudge=%+.3f)",
            report.entity_name, report.predicate,
            probability, probability + total_nudge, total_nudge,
        )

    return round(max(0.01, min(0.99, probability + total_nudge)), 4)


def format_coherence_output(report: CoherenceReport) -> str:
    """Format coherence report for CLI output."""
    if report.is_consistent and not report.violations:
        return ""

    lines = [f"\n  Coherence (score={report.consistency_score:.2f}):"]
    for v in report.violations:
        icon = "⚠" if v.severity == "warning" else "✗"
        lines.append(f"    {icon} [{v.violation_type}] {v.description}")
        if v.suggested_range:
            lo, hi = v.suggested_range
            lines.append(f"      → suggested range: [{lo:.3f}, {hi:.3f}]")

    if report.implications:
        lines.append("  Causal implications:")
        for imp in report.implications[:3]:
            lines.append(
                f"    → {imp['target_entity']}/{imp['target_feature']}: "
                f"Δ={imp['implied_delta']:+.3f}"
            )

    return "\n".join(lines)
