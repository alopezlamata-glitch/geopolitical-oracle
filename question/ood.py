"""
Out-of-Distribution (OOD) detector for the geopolitical oracle.

The current model (XGBoost trained on ICEWS conflict/escalation windows)
is only valid for a narrow class of questions. Answering out-of-domain
questions with a number is worse than refusing — it creates false confidence.

This module:
  1. Assesses whether a ParsedQuestion is in-domain for the CURRENT model.
  2. Returns an OODAssessment with:
       - in_domain (bool): whether to proceed with prediction
       - confidence (float): 0=clearly OOD, 1=clearly in-domain
       - reason (str): human-readable explanation
       - suggested_action (str): what to do instead

Current model domain:
  event_family: "conflict"
  subject_type: "country"
  horizon: <= 90 days
  predicate: military escalation, coup, ceasefire, nuclear event

Future: as new domain-specific models are added, register them here
and the router will pick the right one per question.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

from question.parser import ParsedQuestion, EventFamily


# ── Registered model domains ──────────────────────────────────────────────────
# Each entry describes what a model can answer.
# When a new domain model is added, register it here.

@dataclass
class ModelDomain:
    model_id: str
    description: str
    valid_families: set[EventFamily]
    valid_subject_types: set[str]
    max_horizon_days: int
    valid_predicates: set[str]         # empty = any predicate in family
    notes: str = ""


_REGISTERED_DOMAINS: list[ModelDomain] = [
    ModelDomain(
        model_id="xgb_conflict_v3",
        description="XGBoost conflict/escalation classifier (ICEWS 2012-2022)",
        valid_families={"conflict"},
        valid_subject_types={"country"},
        max_horizon_days=365,
        valid_predicates={
            "military_escalation", "coup", "ceasefire", "nuclear_event",
        },
        notes=(
            "Trained on ICEWS country-month windows labeled by military intensity. "
            "Not valid for political, legal, entertainment, or economic questions."
        ),
    ),
    ModelDomain(
        model_id="ollama_reasoning_v1",
        description="Ollama LLM reasoning predictor for political, economic, and legal events",
        valid_families={"political", "economic", "legal"},
        valid_subject_types={"person", "country", "organization", "unknown", "other"},
        max_horizon_days=730,
        valid_predicates=set(),   # accepts any predicate in the family
        notes=(
            "Uses local Ollama LLM (llama3.x/qwen2.5) for superforecaster-style reasoning. "
            "Non-fatal: falls back to market prior if Ollama unavailable. "
            "Accumulates labeled outcomes for future XGBoost domain models."
        ),
    ),
    ModelDomain(
        model_id="ollama_reasoning_v1",
        description="Entertainment events via Ollama reasoning",
        valid_families={"entertainment"},
        valid_subject_types={"artist", "person", "unknown", "other"},
        max_horizon_days=365,
        valid_predicates={"perform", "visit", "music_release", "retire", "tour"},
        notes="Ollama reasoning for entertainment/cultural events.",
    ),
]


# ── Assessment ────────────────────────────────────────────────────────────────

@dataclass
class OODAssessment:
    """Result of OOD check for a ParsedQuestion."""
    in_domain: bool
    matched_model: Optional[str]       # model_id if in_domain else None
    ood_score: float                   # 0.0 = clearly in domain, 1.0 = clearly OOD
    reason: str
    suggested_action: str
    domain_gap: list[str]              # specific mismatches found


def assess_ood(pq: ParsedQuestion, reference_date: Optional[date] = None) -> OODAssessment:
    """
    Determine whether this question is in-domain for any registered model.

    Returns OODAssessment. If in_domain=True, matched_model identifies which
    model to use. If in_domain=False, do not predict — return OOD message.
    """
    if reference_date is None:
        reference_date = date.today()

    # Compute horizon in days
    if pq.deadline is not None:
        horizon_days = max(0, (pq.deadline - reference_date).days)
    else:
        horizon_days = None

    best_score = 0.0
    best_model = None
    best_gaps: list[str] = []

    for domain in _REGISTERED_DOMAINS:
        score, gaps = _score_domain_match(pq, domain, horizon_days)
        if score > best_score:
            best_score = score
            best_model = domain
            best_gaps = gaps

    # In domain if score >= 0.60 and no hard failures
    in_domain = best_score >= 0.60 and best_model is not None and len(best_gaps) == 0

    if in_domain:
        return OODAssessment(
            in_domain=True,
            matched_model=best_model.model_id,
            ood_score=round(1.0 - best_score, 2),
            reason=f"Question matches domain of {best_model.model_id}.",
            suggested_action="proceed",
            domain_gap=[],
        )

    # Build human-readable OOD explanation
    reason, action = _build_ood_explanation(pq, best_gaps, best_score, reference_date)

    return OODAssessment(
        in_domain=False,
        matched_model=None,
        ood_score=round(1.0 - best_score, 2),
        reason=reason,
        suggested_action=action,
        domain_gap=best_gaps,
    )


def _score_domain_match(
    pq: ParsedQuestion,
    domain: ModelDomain,
    horizon_days: Optional[int],
) -> tuple[float, list[str]]:
    """
    Score how well a question matches a model domain.
    Returns (score 0-1, list_of_mismatches).
    Higher = better match. score >= 0.60 with no gaps = in domain.
    """
    score = 0.0
    gaps: list[str] = []

    # ── Family check (hard requirement) ───────────────────────────────────────
    if pq.event_family in domain.valid_families:
        score += 0.40
    else:
        gaps.append(
            f"event_family='{pq.event_family}' not in {domain.valid_families}"
        )

    # ── Subject type check (hard requirement) ─────────────────────────────────
    if pq.subject_type in domain.valid_subject_types:
        score += 0.25
    else:
        gaps.append(
            f"subject_type='{pq.subject_type}' not in {domain.valid_subject_types}"
        )

    # ── Predicate check (soft) ────────────────────────────────────────────────
    if not domain.valid_predicates or pq.predicate in domain.valid_predicates:
        score += 0.20
    else:
        gaps.append(
            f"predicate='{pq.predicate}' not in {domain.valid_predicates}"
        )

    # ── Horizon check (soft) ──────────────────────────────────────────────────
    if horizon_days is None:
        # No deadline: mild penalty
        score += 0.05
        gaps.append("no deadline specified — prediction horizon unknown")
    elif horizon_days <= domain.max_horizon_days:
        score += 0.15
    else:
        gaps.append(
            f"horizon={horizon_days}d exceeds model max {domain.max_horizon_days}d"
        )
        score += 0.05  # partial credit — still might be useful

    return score, gaps


def _build_ood_explanation(
    pq: ParsedQuestion,
    gaps: list[str],
    best_score: float,
    reference_date: date,
) -> tuple[str, str]:
    """Build human-readable reason and suggested_action for OOD result."""

    family = pq.event_family
    subject_type = pq.subject_type

    # Specific messages per domain gap
    if family == "political" and subject_type == "person":
        reason = (
            f"This question asks about a political event ('{pq.predicate}') "
            f"involving a person ('{pq.subject}'). "
            "The current model is trained only on country-level conflict/escalation "
            "and has no features for political survival (coalitions, approval, "
            "judicial pressure, electoral calendars)."
        )
        action = (
            "A political survival model would require: parliamentary approval data, "
            "coalition stability indicators, judicial pressure scores, and electoral "
            "calendar features. Data sources: ParlGov, V-Dem, news sentiment by leader."
        )

    elif family == "entertainment":
        reason = (
            f"This question asks about an entertainment event ('{pq.predicate}') "
            f"involving '{pq.subject}'. "
            "The current model has no features for tour schedules, venue bookings, "
            "music releases, or artist activity patterns."
        )
        action = (
            "An entertainment events model would require: historical tour data, "
            "venue reservation signals, official artist announcements, and "
            "ticketing platform data (Songkick, Bandsintown, Ticketmaster API)."
        )

    elif family == "legal":
        reason = (
            f"This question asks about a legal event ('{pq.predicate}') "
            f"involving '{pq.subject}'. "
            "The current model has no features for judicial processes, extradition "
            "status, arrest warrants, or bilateral legal agreements."
        )
        action = (
            "A legal events model would require: INTERPOL notice status, bilateral "
            "extradition treaty data, judicial calendar, DOJ/official court records, "
            "and news on legal proceedings."
        )

    elif family == "economic":
        reason = (
            f"This question asks about an economic event ('{pq.predicate}'). "
            "The current model has no macroeconomic features, central bank calendars, "
            "or financial market signals."
        )
        action = (
            "An economic events model would require: central bank meeting calendars, "
            "inflation data, interest rate forwards, GDP growth estimates, "
            "and financial market indicators."
        )

    elif family == "conflict" and subject_type != "country":
        reason = (
            f"This question is about a conflict event but involves a "
            f"'{subject_type}' ('{pq.subject}') rather than a country. "
            "The current model predicts conflict at country level only."
        )
        action = (
            "Rephrase the question at country level, e.g. "
            f"'Will there be military escalation in [country] before [date]?'"
        )

    elif not pq.has_deadline:
        reason = (
            "No deadline found in the question. "
            "Binary forecasting requires a specific resolution date to be meaningful."
        )
        action = (
            "Add a specific deadline, e.g. 'before 1 July 2026' or 'in 2026'."
        )

    else:
        reason = (
            f"Question does not match any registered model domain. "
            f"Detected: event_family='{family}', predicate='{pq.predicate}', "
            f"subject_type='{subject_type}'. "
            f"Mismatches: {'; '.join(gaps)}."
        )
        action = (
            "The current system only handles country-level conflict/escalation questions. "
            "Consider refining the question or waiting for domain-specific models."
        )

    return reason, action
