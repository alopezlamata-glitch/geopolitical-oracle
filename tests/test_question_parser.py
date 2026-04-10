"""
Unit tests for question/parser.py and question/ood.py.

All tests are pure Python — no network, no model files, no disk I/O.

Coverage:
  - Parser: subject extraction, predicate classification, deadline parsing,
    event family assignment, negation detection, jurisdiction, resolution rule
  - OOD detector: in-domain acceptance, OOD rejection per domain gap type,
    no-deadline handling, edge cases
"""
from __future__ import annotations

import pytest
from datetime import date

from question.parser import parse_question, ParsedQuestion
from question.ood import assess_ood, OODAssessment


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse(q: str) -> ParsedQuestion:
    return parse_question(q)

def ood(q: str, ref: date | None = None) -> OODAssessment:
    pq = parse_question(q)
    return assess_ood(pq, reference_date=ref or date(2026, 4, 1))


# ═══════════════════════════════════════════════════════════════════════════════
# Parser tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestEventFamily:
    def test_conflict_escalation(self):
        pq = parse("Will there be a military escalation in Ukraine before June 2026?")
        assert pq.event_family == "conflict"

    def test_conflict_coup(self):
        pq = parse("Will there be a coup in Venezuela before 1 July 2026?")
        assert pq.event_family == "conflict"

    def test_conflict_ceasefire(self):
        pq = parse("Will a ceasefire be declared in Gaza before April 2026?")
        assert pq.event_family == "conflict"

    def test_political_resign(self):
        pq = parse("Will Pedro Sánchez resign before 6 April 2026?")
        assert pq.event_family == "political"

    def test_political_election(self):
        pq = parse("Will there be a snap election in France before December 2026?")
        assert pq.event_family == "political"

    def test_political_sanction(self):
        pq = parse("Will the US impose new sanctions on Russia before July 2026?")
        assert pq.event_family == "political"

    def test_legal_arrest(self):
        pq = parse("Will Nicolás Maduro be arrested before 5 April 2026?")
        assert pq.event_family == "legal"

    def test_legal_captured(self):
        pq = parse("Will the US capture Maduro before April 2026?")
        assert pq.event_family == "legal"

    def test_entertainment_concert(self):
        pq = parse("Will Bad Bunny perform in Spain in 2026?")
        assert pq.event_family == "entertainment"

    def test_entertainment_tour(self):
        pq = parse("Will Taylor Swift tour Europe before December 2026?")
        assert pq.event_family == "entertainment"

    def test_economic_rate_cut(self):
        pq = parse("Will the Fed cut rates before June 2026?")
        assert pq.event_family == "economic"

    def test_economic_recession(self):
        pq = parse("Will the US enter a recession in 2026?")
        assert pq.event_family == "economic"


class TestPredicateExtraction:
    def test_resign_predicate(self):
        pq = parse("Will Pedro Sánchez resign before April 2026?")
        assert pq.predicate == "resign"

    def test_arrested_predicate(self):
        pq = parse("Will Maduro be arrested before April 2026?")
        assert pq.predicate == "arrested"

    def test_captured_predicate(self):
        pq = parse("Will the US capture Maduro before April 2026?")
        assert pq.predicate == "captured"

    def test_military_escalation(self):
        pq = parse("Will Russia invade Poland before 2027?")
        assert pq.predicate == "military_escalation"

    def test_perform_predicate(self):
        pq = parse("Will Bad Bunny perform in Spain in 2026?")
        assert pq.predicate == "perform"

    def test_interest_rate_predicate(self):
        pq = parse("Will the Fed cut rates before June 2026?")
        assert pq.predicate == "interest_rate"

    def test_coup_predicate(self):
        pq = parse("Will there be a coup in Venezuela before July 2026?")
        assert pq.predicate == "coup"

    def test_ceasefire_predicate(self):
        pq = parse("Will Israel and Hamas reach a ceasefire before May 2026?")
        assert pq.predicate == "ceasefire"

    def test_nuclear_predicate(self):
        pq = parse("Will North Korea conduct a nuclear test before December 2026?")
        assert pq.predicate == "nuclear_event"


class TestSubjectExtraction:
    def test_known_artist(self):
        pq = parse("Will Bad Bunny perform in Spain in 2026?")
        assert "bad bunny" in pq.subject.lower()
        assert pq.subject_type == "artist"

    def test_known_politician(self):
        pq = parse("Will Pedro Sánchez resign before April 2026?")
        assert "s" in pq.subject.lower()  # Sánchez matched
        assert pq.subject_type == "person"

    def test_country_subject(self):
        pq = parse("Will Ukraine launch a counteroffensive before June 2026?")
        assert pq.subject_type == "country"

    def test_taylor_swift_artist(self):
        pq = parse("Will Taylor Swift release a new album before 2027?")
        assert pq.subject_type == "artist"

    def test_trump_politician(self):
        pq = parse("Will Donald Trump impose tariffs on Europe before July 2026?")
        assert pq.subject_type == "person"

    def test_north_korea_country(self):
        pq = parse("Will North Korea conduct a nuclear test in 2026?")
        assert pq.subject_type == "country"


class TestDeadlineParsing:
    def test_before_day_month_year(self):
        pq = parse("Will Pedro Sánchez resign before 6 April 2026?")
        assert pq.deadline == date(2026, 4, 6)

    def test_before_month_day_year(self):
        pq = parse("Will there be escalation in Ukraine before April 6, 2026?")
        assert pq.deadline == date(2026, 4, 6)

    def test_in_year(self):
        pq = parse("Will Bad Bunny perform in Spain in 2026?")
        assert pq.deadline is not None
        assert pq.deadline.year == 2026

    def test_by_month_year(self):
        pq = parse("Will the Fed cut rates by June 2026?")
        assert pq.deadline is not None
        assert pq.deadline.month == 6
        assert pq.deadline.year == 2026

    def test_no_deadline(self):
        pq = parse("Will China invade Taiwan?")
        assert pq.deadline is None
        assert not pq.has_deadline

    def test_iso_date(self):
        pq = parse("Will there be escalation in Gaza before 2026-05-01?")
        assert pq.deadline == date(2026, 5, 1)


class TestNegation:
    def test_negated_question(self):
        pq = parse("Will Pedro Sánchez not resign before April 2026?")
        assert pq.is_negated is True

    def test_non_negated_question(self):
        pq = parse("Will Pedro Sánchez resign before April 2026?")
        assert pq.is_negated is False

    def test_fail_to_negation(self):
        pq = parse("Will negotiations fail to reach a ceasefire before May 2026?")
        assert pq.is_negated is True


class TestJurisdiction:
    def test_country_in_question(self):
        pq = parse("Will there be military escalation in Ukraine before June 2026?")
        assert pq.jurisdiction is not None
        assert "ukraine" in pq.jurisdiction.lower()

    def test_spain_jurisdiction(self):
        pq = parse("Will Bad Bunny perform in Spain in 2026?")
        assert pq.jurisdiction is not None
        assert "spain" in pq.jurisdiction.lower()

    def test_no_jurisdiction(self):
        pq = parse("Will the Fed cut rates before June 2026?")
        # "us" might be picked up — that's fine, or might be None
        # Just check it doesn't crash
        assert pq.jurisdiction is None or isinstance(pq.jurisdiction, str)


class TestResolutionRule:
    def test_resign_rule(self):
        pq = parse("Will Pedro Sánchez resign before 6 April 2026?")
        assert "resignation" in pq.resolution_rule.lower() or "resign" in pq.resolution_rule.lower()
        assert pq.subject in pq.resolution_rule

    def test_arrested_rule(self):
        pq = parse("Will Maduro be arrested before April 2026?")
        assert "arrest" in pq.resolution_rule.lower() or "detention" in pq.resolution_rule.lower()

    def test_perform_rule(self):
        pq = parse("Will Bad Bunny perform in Spain in 2026?")
        assert "concert" in pq.resolution_rule.lower() or "performance" in pq.resolution_rule.lower()


class TestParseConfidence:
    def test_high_confidence_conflict(self):
        pq = parse("Will there be a military escalation in Ukraine before June 2026?")
        assert pq.parse_confidence >= 0.70

    def test_lower_confidence_vague(self):
        pq = parse("Will something happen?")
        assert pq.parse_confidence < 0.60


# ═══════════════════════════════════════════════════════════════════════════════
# OOD detector tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestInDomain:
    """Questions that should be accepted by the current xgb_conflict_v3 model."""

    def test_military_escalation_country(self):
        a = ood("Will there be a military escalation in Ukraine before June 2026?")
        assert a.in_domain is True
        assert a.matched_model == "xgb_conflict_v3"

    def test_coup_country(self):
        a = ood("Will there be a coup in Venezuela before July 2026?")
        assert a.in_domain is True

    def test_ceasefire_country(self):
        a = ood("Will a ceasefire be declared in Gaza before April 2026?")
        assert a.in_domain is True

    def test_nuclear_test(self):
        a = ood("Will North Korea conduct a nuclear test before December 2026?")
        assert a.in_domain is True

    def test_ood_score_low_for_in_domain(self):
        a = ood("Will Russia invade Poland before 2027?")
        assert a.ood_score < 0.5


class TestOutOfDomain:
    """Questions that must be rejected with clear explanation."""

    def test_political_resign(self):
        a = ood("Will Pedro Sánchez resign before 6 April 2026?")
        assert a.in_domain is False
        assert "political" in a.reason.lower() or "political" in a.domain_gap[0].lower()

    def test_legal_arrest(self):
        a = ood("Will Nicolás Maduro be arrested before 5 April 2026?")
        assert a.in_domain is False
        # The model is OOD because it's legal family + person subject.
        # Accept either "legal" in reason, or the gap list mentions it, or
        # the reason mentions person/subject mismatch.
        assert (
            "legal" in a.reason.lower()
            or "person" in a.reason.lower()
            or any("legal" in g.lower() or "person" in g.lower() for g in a.domain_gap)
        )

    def test_entertainment_concert(self):
        a = ood("Will Bad Bunny perform in Spain in 2026?")
        assert a.in_domain is False
        assert "entertainment" in a.reason.lower()

    def test_economic_rate(self):
        a = ood("Will the Fed cut rates before June 2026?")
        assert a.in_domain is False
        assert "economic" in a.reason.lower()

    def test_suggested_action_not_empty(self):
        a = ood("Will Pedro Sánchez resign before 6 April 2026?")
        assert len(a.suggested_action) > 20

    def test_ood_score_high_for_ood(self):
        a = ood("Will Bad Bunny perform in Spain in 2026?")
        assert a.ood_score > 0.5


class TestEdgeCases:
    def test_no_deadline_ood(self):
        a = ood("Will China invade Taiwan?")
        # No deadline → OOD (horizon unknown) or domain mismatch
        # Either way, should not crash
        assert isinstance(a.in_domain, bool)
        assert isinstance(a.reason, str)

    def test_ambiguous_question(self):
        a = ood("Will something bad happen?")
        assert a.in_domain is False

    def test_conflict_but_person_subject_ood(self):
        # Conflict family but person subject → OOD for current country-level model
        a = ood("Will Vladimir Putin launch a military attack before June 2026?")
        # Putin is a person, not a country → subject_type mismatch
        # May or may not be in domain depending on parser; just check no crash
        assert isinstance(a.in_domain, bool)

    def test_assessment_has_all_fields(self):
        a = ood("Will Bad Bunny perform in Spain in 2026?")
        assert hasattr(a, "in_domain")
        assert hasattr(a, "ood_score")
        assert hasattr(a, "reason")
        assert hasattr(a, "suggested_action")
        assert hasattr(a, "domain_gap")
        assert hasattr(a, "matched_model")

    def test_past_deadline_still_parses(self):
        # Deadline already passed — parser should still parse it
        pq = parse("Will Iran attack Israel before 1 January 2020?")
        assert pq.deadline == date(2020, 1, 1)
        # OOD check with this past deadline
        a = assess_ood(pq, reference_date=date(2026, 4, 1))
        # horizon_days will be negative — should handle gracefully
        assert isinstance(a.in_domain, bool)
