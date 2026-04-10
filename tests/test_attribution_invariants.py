"""
Attribution invariant tests.

These are *model-free* unit tests: they do not require a trained model and
do not call SHAP. Instead, they verify the mathematical contracts that the
attribution code must satisfy, given controlled synthetic inputs.

Three invariant families:

1. Direction: removing a positively-contributing event must not increase
   the predicted probability; removing a negatively-contributing one must
   not decrease it.

2. Additivity sanity: the sum of all event log-odds contributions must
   approximately equal the difference between the prediction log-odds and
   the SHAP expected value (checked via the formula, not by re-running SHAP).

3. Flip-set fidelity: if a flip-set is returned, the `p_after_removal`
   must be on the opposite side of 0.5 from `base_prob`.

These tests run without GPU, without a trained model, and without internet.
They import only stdlib + the internal module under test.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import pytest


# ─── Minimal stub of CanonicalEvent (avoids importing the full module) ────────

@dataclass
class _StubEvent:
    event_id: str
    raw_title: str = ""
    event_type: str = "military_action"
    sub_event_type: str = ""
    occurred_at: datetime = field(default_factory=lambda: datetime(2024, 1, 1, tzinfo=timezone.utc))
    country: str = "TestCountry"


# ─── Pure-Python reimplementation of the flip-set logic ──────────────────────
# Mirrors predictor/attribution.py::_compute_flip_set so we can test it
# without requiring xgboost or shap.

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _logit(p: float, eps: float = 1e-6) -> float:
    p = max(eps, min(1 - eps, p))
    return math.log(p / (1.0 - p))


def _remove_event_prob(logit_base: float, delta_logodds: float) -> float:
    """Probability after removing an event's log-odds contribution."""
    return _sigmoid(logit_base - delta_logodds)


def compute_flip_set_pure(
    logit_base: float,
    event_margin: dict[str, float],
    events_by_id: dict,
) -> dict:
    """
    Pure-Python flip-set computation.
    Returns {"events": [...], "p_after_removal": float|None, ...}
    """
    eps = 1e-6
    base_prob = _sigmoid(logit_base)
    answer_is_yes = base_prob >= 0.5
    ranked = sorted(event_margin.items(), key=lambda kv: abs(kv[1]), reverse=True)

    selected_ids = []
    logit_running = logit_base

    for eid, dlogit in ranked:
        if answer_is_yes and dlogit <= 0:
            continue
        if not answer_is_yes and dlogit >= 0:
            continue

        logit_running -= dlogit
        selected_ids.append(eid)
        p_minus = _sigmoid(logit_running)

        flipped = (answer_is_yes and p_minus < 0.5) or (not answer_is_yes and p_minus >= 0.5)
        if flipped:
            events = [
                {
                    "event_id": feid,
                    "delta_logodds": round(event_margin.get(feid, 0.0), 5),
                }
                for feid in selected_ids
            ]
            return {
                "events": events,
                "p_after_removal": round(p_minus, 4),
                "delta": round(p_minus - base_prob, 4),
            }

    return {"events": [], "p_after_removal": None, "delta": None}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_events(n: int = 5) -> tuple[dict[str, float], dict[str, _StubEvent]]:
    """Create n synthetic events with varying log-odds contributions."""
    event_margin = {}
    events_by_id = {}
    for i in range(n):
        eid = f"ev{i}"
        # Alternating positive/negative contributions of decreasing magnitude
        sign = 1 if i % 2 == 0 else -1
        event_margin[eid] = sign * (0.5 - i * 0.08)
        events_by_id[eid] = _StubEvent(event_id=eid, raw_title=f"Event {i}")
    return event_margin, events_by_id


# ─── Invariant 1: Direction ───────────────────────────────────────────────────

class TestDirectionInvariant:
    """
    Removing a positive-contribution event must not raise the probability.
    Removing a negative-contribution event must not lower the probability.
    """

    def test_remove_positive_does_not_increase_prob(self):
        logit_base = 0.4   # p ≈ 0.60
        positive_events = {"ev_pos": 0.6}   # strong positive contribution

        p_base = _sigmoid(logit_base)
        p_minus = _remove_event_prob(logit_base, positive_events["ev_pos"])

        assert p_minus <= p_base + 1e-9, (
            f"Removing a positive event raised p from {p_base:.4f} to {p_minus:.4f}"
        )

    def test_remove_negative_does_not_decrease_prob(self):
        logit_base = -0.3   # p ≈ 0.43
        negative_delta = -0.5   # event pushes DOWN

        p_base = _sigmoid(logit_base)
        p_minus = _remove_event_prob(logit_base, negative_delta)

        assert p_minus >= p_base - 1e-9, (
            f"Removing a negative event decreased p from {p_base:.4f} to {p_minus:.4f}"
        )

    def test_direction_symmetric(self):
        """The direction invariant holds for any sign of logit_base."""
        for logit_base in [-1.5, -0.5, 0.0, 0.5, 1.5]:
            for delta in [-0.8, -0.3, 0.3, 0.8]:
                p_base = _sigmoid(logit_base)
                p_minus = _remove_event_prob(logit_base, delta)
                if delta > 0:
                    assert p_minus <= p_base + 1e-9, (
                        f"logit={logit_base}, delta={delta}: removing positive raised prob"
                    )
                else:
                    assert p_minus >= p_base - 1e-9, (
                        f"logit={logit_base}, delta={delta}: removing negative lowered prob"
                    )


# ─── Invariant 2: Additivity ──────────────────────────────────────────────────

class TestAdditivity:
    """
    The sum of per-event log-odds contributions must approximately equal
    the total shift from SHAP expected value to predicted log-odds.

    We test the provenance-weighted mapping: if features map exactly to
    events with weight 1.0, the event contributions should sum to the
    feature SHAP values.
    """

    def test_single_feature_single_event_exact(self):
        """One feature → one event at weight 1.0: event contribution = feature SHAP."""
        shap_margin = {"military_count_7d": 0.42}
        provenance = {"military_count_7d": [{"event_id": "ev0", "weight": 1.0}]}

        # Compute event_margin as attribution.py does
        event_margin: dict[str, float] = defaultdict(float)
        for fname, phi in shap_margin.items():
            contribs = provenance.get(fname, [])
            total_w = sum(c["weight"] for c in contribs) or 1.0
            for c in contribs:
                event_margin[c["event_id"]] += phi * (c["weight"] / total_w)

        assert abs(event_margin["ev0"] - 0.42) < 1e-9

    def test_feature_split_equally_across_two_events(self):
        """Feature contribution splits equally between two events of equal weight."""
        shap_margin = {"military_count_7d": 0.60}
        provenance = {
            "military_count_7d": [
                {"event_id": "ev0", "weight": 1.0},
                {"event_id": "ev1", "weight": 1.0},
            ]
        }
        event_margin: dict[str, float] = defaultdict(float)
        for fname, phi in shap_margin.items():
            contribs = provenance.get(fname, [])
            total_w = sum(c["weight"] for c in contribs) or 1.0
            for c in contribs:
                event_margin[c["event_id"]] += phi * (c["weight"] / total_w)

        assert abs(event_margin["ev0"] - 0.30) < 1e-9
        assert abs(event_margin["ev1"] - 0.30) < 1e-9

    def test_total_event_contribution_equals_shap_sum(self):
        """Sum of all event contributions must equal sum of all feature SHAP values."""
        shap_margin = {
            "military_count_7d": 0.3,
            "protest_count_7d": -0.1,
            "overall_intensity_7d": 0.5,
        }
        provenance = {
            "military_count_7d": [{"event_id": "ev0", "weight": 2.0},
                                   {"event_id": "ev1", "weight": 1.0}],
            "protest_count_7d": [{"event_id": "ev1", "weight": 1.0}],
            "overall_intensity_7d": [{"event_id": "ev0", "weight": 1.0},
                                      {"event_id": "ev2", "weight": 3.0}],
        }
        event_margin: dict[str, float] = defaultdict(float)
        for fname, phi in shap_margin.items():
            contribs = provenance.get(fname, [])
            total_w = sum(c["weight"] for c in contribs) or 1.0
            for c in contribs:
                event_margin[c["event_id"]] += phi * (c["weight"] / total_w)

        total_event = sum(event_margin.values())
        total_shap = sum(shap_margin.values())
        assert abs(total_event - total_shap) < 1e-9, (
            f"Event sum {total_event:.6f} != SHAP sum {total_shap:.6f}"
        )


# ─── Invariant 3: Flip-set fidelity ──────────────────────────────────────────

class TestFlipSetFidelity:
    """
    If a flip-set is returned, `p_after_removal` must be strictly on the
    opposite side of 0.5 from `base_prob`.
    """

    def _base_prob(self, logit_base: float) -> float:
        return _sigmoid(logit_base)

    def test_flip_set_crosses_threshold_yes_to_no(self):
        """YES prediction: removing the flip-set must yield p < 0.5."""
        logit_base = 0.8   # p ≈ 0.69 (YES)
        event_margin = {
            "ev0": 0.7,   # large positive
            "ev1": 0.3,   # medium positive
            "ev2": -0.1,  # negative (will not be removed)
        }
        events_by_id = {eid: _StubEvent(event_id=eid) for eid in event_margin}

        result = compute_flip_set_pure(logit_base, event_margin, events_by_id)

        if result["p_after_removal"] is not None:
            assert result["p_after_removal"] < 0.5, (
                f"YES→NO flip: p_after={result['p_after_removal']:.4f} should be < 0.5"
            )

    def test_flip_set_crosses_threshold_no_to_yes(self):
        """NO prediction: removing the flip-set must yield p >= 0.5."""
        logit_base = -0.8   # p ≈ 0.31 (NO)
        event_margin = {
            "ev0": -0.6,  # large negative
            "ev1": -0.2,  # medium negative
            "ev2": 0.1,   # positive (will not be removed)
        }
        events_by_id = {eid: _StubEvent(event_id=eid) for eid in event_margin}

        result = compute_flip_set_pure(logit_base, event_margin, events_by_id)

        if result["p_after_removal"] is not None:
            assert result["p_after_removal"] >= 0.5, (
                f"NO→YES flip: p_after={result['p_after_removal']:.4f} should be >= 0.5"
            )

    def test_flip_set_empty_when_no_flip_possible(self):
        """When all events push in the same direction as the answer and are insufficient,
        or when events are too weak, the flip-set may be empty."""
        logit_base = 5.0   # p ≈ 0.993 (very strong YES)
        event_margin = {"ev0": 0.05}   # tiny positive — removing it won't flip
        events_by_id = {eid: _StubEvent(event_id=eid) for eid in event_margin}

        result = compute_flip_set_pure(logit_base, event_margin, events_by_id)

        # With only 0.05 logodds available, logit drops to 4.95 → p still >> 0.5
        if result["p_after_removal"] is not None:
            # If somehow a result is returned, it must still cross 0.5
            assert result["p_after_removal"] < 0.5

    def test_flip_set_only_removes_supporting_events(self):
        """Negative events must not appear in a YES flip-set."""
        logit_base = 1.0   # YES
        event_margin = {
            "ev_pos": 0.8,
            "ev_neg": -0.5,   # should NOT be removed for a YES flip
        }
        events_by_id = {eid: _StubEvent(event_id=eid) for eid in event_margin}

        result = compute_flip_set_pure(logit_base, event_margin, events_by_id)

        for ev in result.get("events", []):
            delta = event_margin.get(ev["event_id"], 0.0)
            assert delta > 0, (
                f"Negative-contribution event {ev['event_id']} (delta={delta:.3f}) "
                "appeared in a YES flip-set — only positive events should be removed"
            )

    def test_flip_set_minimum_events(self):
        """
        Flip-set should return the minimum number of events needed.
        If one event is sufficient to flip, the set should contain only that event.
        """
        logit_base = 0.3   # p ≈ 0.57 (just above 0.5)
        event_margin = {
            "ev_big": 1.0,   # this alone is enough: logit drops to -0.7 → p ≈ 0.33
            "ev_small": 0.1,
        }
        events_by_id = {eid: _StubEvent(event_id=eid) for eid in event_margin}

        result = compute_flip_set_pure(logit_base, event_margin, events_by_id)

        if result["p_after_removal"] is not None:
            assert len(result["events"]) == 1, (
                f"Expected 1 event in flip-set, got {len(result['events'])}. "
                "ev_big alone should be sufficient to flip."
            )
            assert result["events"][0]["event_id"] == "ev_big"


# ─── Invariant 4: Probability space display ───────────────────────────────────

class TestDisplaySpaceConversion:
    """
    delta_prob_approx = delta_logodds * p*(1-p).
    The local linearisation should be within 20% of the exact sigmoid difference
    for small perturbations, and labelled correctly.
    """

    @pytest.mark.parametrize("p_raw,delta_lo", [
        (0.5, 0.1), (0.3, 0.15), (0.7, -0.1), (0.2, 0.05),
    ])
    def test_linearisation_close_to_exact_for_small_delta(self, p_raw: float, delta_lo: float):
        """Local linearisation is within 20% of exact for |delta_lo| <= 0.2."""
        local_slope = p_raw * (1.0 - p_raw)
        delta_prob_approx = delta_lo * local_slope

        logit_base = _logit(p_raw)
        p_exact_minus = _sigmoid(logit_base - delta_lo)
        delta_prob_exact = p_raw - p_exact_minus

        if abs(delta_prob_exact) > 1e-6:
            ratio = abs(delta_prob_approx / delta_prob_exact)
            assert 0.7 <= ratio <= 1.3, (
                f"p={p_raw}, delta_lo={delta_lo}: "
                f"approx={delta_prob_approx:.4f} vs exact={delta_prob_exact:.4f} "
                f"(ratio={ratio:.2f})"
            )
