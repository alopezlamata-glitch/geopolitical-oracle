"""
Seed synthetic training data to bootstrap XGBoost before real labeled predictions accumulate.

Generated examples cover a range of geopolitical scenarios calibrated to realistic
feature distributions. Each example has a ground-truth outcome (0/1) that follows
logistic relationships with the features.

Run: python scripts/seed_training_data.py
"""
from __future__ import annotations

import json
import math
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

_TRAINING_DIR = Path(__file__).parent.parent / "data" / "training"
_N = 80  # number of synthetic examples to generate
random.seed(42)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _logit(p: float) -> float:
    p = max(0.001, min(0.999, p))
    return math.log(p / (1 - p))


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ── Feature generators for two broad classes ─────────────────────────────────

def _sample_high_risk() -> dict:
    """Scenario: active conflict / high escalation probability."""
    return {
        "military_count_7d": random.randint(3, 15),
        "military_count_30d": random.randint(8, 40),
        "protest_count_7d": random.randint(1, 8),
        "protest_count_30d": random.randint(2, 20),
        "diplomatic_count_7d": random.randint(0, 3),
        "ceasefire_count_7d": random.randint(0, 1),
        "sanction_count_7d": random.randint(1, 5),
        "political_crisis_count_7d": random.randint(1, 4),
        "military_intensity_7d": random.uniform(2.0, 8.0),
        "protest_intensity_7d": random.uniform(0.5, 3.0),
        "overall_intensity_7d": random.uniform(3.0, 12.0),
        "military_accel": random.uniform(1.5, 5.0),
        "protest_accel": random.uniform(0.8, 3.0),
        "overall_accel": random.uniform(1.2, 4.0),
        "avg_polarity_7d": random.uniform(-0.9, -0.4),
        "avg_polarity_30d": random.uniform(-0.8, -0.3),
        "tone_trend": random.uniform(-0.4, 0.0),
        "source_diversity_7d": random.uniform(0.4, 1.0),
        "avg_independent_sources": random.uniform(1.5, 3.0),
        "avg_contradiction_score": random.uniform(0.0, 0.3),
        "fatalities_7d": random.randint(10, 500),
        "has_military_7d": 1.0,
        "has_ceasefire_7d": 0.0,
        "escalation_index": random.uniform(2.0, 8.0),
        "metaculus_p": _clamp(random.gauss(0.72, 0.1), 0.05, 0.95),
        "polymarket_p": _clamp(random.gauss(0.68, 0.12), 0.05, 0.95),
        "market_available": 1.0,
    }


def _sample_low_risk() -> dict:
    """Scenario: stable / diplomatic / ceasefire-trending."""
    return {
        "military_count_7d": random.randint(0, 2),
        "military_count_30d": random.randint(0, 5),
        "protest_count_7d": random.randint(0, 3),
        "protest_count_30d": random.randint(0, 8),
        "diplomatic_count_7d": random.randint(2, 8),
        "ceasefire_count_7d": random.randint(1, 4),
        "sanction_count_7d": random.randint(0, 2),
        "political_crisis_count_7d": random.randint(0, 1),
        "military_intensity_7d": random.uniform(0.0, 1.0),
        "protest_intensity_7d": random.uniform(0.0, 0.8),
        "overall_intensity_7d": random.uniform(0.2, 2.0),
        "military_accel": random.uniform(0.0, 0.8),
        "protest_accel": random.uniform(0.0, 1.0),
        "overall_accel": random.uniform(0.2, 1.2),
        "avg_polarity_7d": random.uniform(-0.2, 0.5),
        "avg_polarity_30d": random.uniform(-0.1, 0.4),
        "tone_trend": random.uniform(0.0, 0.4),
        "source_diversity_7d": random.uniform(0.1, 0.5),
        "avg_independent_sources": random.uniform(1.0, 2.0),
        "avg_contradiction_score": random.uniform(0.0, 0.2),
        "fatalities_7d": random.randint(0, 5),
        "has_military_7d": float(random.random() < 0.15),
        "has_ceasefire_7d": float(random.random() < 0.6),
        "escalation_index": random.uniform(-1.0, 0.5),
        "metaculus_p": _clamp(random.gauss(0.22, 0.1), 0.05, 0.95),
        "polymarket_p": _clamp(random.gauss(0.25, 0.12), 0.05, 0.95),
        "market_available": float(random.random() < 0.7),
    }


def _sample_uncertain() -> dict:
    """Scenario: mixed signals — could go either way."""
    return {
        "military_count_7d": random.randint(1, 5),
        "military_count_30d": random.randint(3, 15),
        "protest_count_7d": random.randint(1, 5),
        "protest_count_30d": random.randint(2, 12),
        "diplomatic_count_7d": random.randint(1, 5),
        "ceasefire_count_7d": random.randint(0, 2),
        "sanction_count_7d": random.randint(0, 3),
        "political_crisis_count_7d": random.randint(0, 2),
        "military_intensity_7d": random.uniform(0.5, 3.0),
        "protest_intensity_7d": random.uniform(0.3, 2.0),
        "overall_intensity_7d": random.uniform(1.0, 5.0),
        "military_accel": random.uniform(0.5, 2.0),
        "protest_accel": random.uniform(0.5, 2.0),
        "overall_accel": random.uniform(0.5, 2.0),
        "avg_polarity_7d": random.uniform(-0.5, 0.1),
        "avg_polarity_30d": random.uniform(-0.4, 0.2),
        "tone_trend": random.uniform(-0.2, 0.2),
        "source_diversity_7d": random.uniform(0.2, 0.7),
        "avg_independent_sources": random.uniform(1.0, 2.5),
        "avg_contradiction_score": random.uniform(0.1, 0.5),
        "fatalities_7d": random.randint(0, 30),
        "has_military_7d": float(random.random() < 0.5),
        "has_ceasefire_7d": float(random.random() < 0.3),
        "escalation_index": random.uniform(-0.5, 2.0),
        "metaculus_p": _clamp(random.gauss(0.48, 0.15), 0.05, 0.95),
        "polymarket_p": _clamp(random.gauss(0.50, 0.15), 0.05, 0.95),
        "market_available": float(random.random() < 0.5),
    }


def _compute_outcome(features: dict) -> int:
    """
    Simulate a ground-truth outcome using a logistic model of the features.
    This ensures the training data has a learnable signal.
    """
    log_odds = (
        0.8  * features["military_intensity_7d"]
        + 0.5  * features["military_count_7d"]
        + 0.6  * features["military_accel"]
        + 0.4  * features["overall_accel"]
        - 0.7  * features["avg_polarity_7d"]       # negative polarity → higher risk
        - 0.9  * features["ceasefire_count_7d"]
        + 0.5  * features["sanction_count_7d"]
        + 0.3  * features["political_crisis_count_7d"]
        + 0.0003 * features["fatalities_7d"]
        + 0.4  * features["has_military_7d"]
        - 0.5  * features["has_ceasefire_7d"]
        - 3.0  # intercept (base rate ~4% if all features = 0)
    )

    # Add market signal if available
    if features["market_available"] > 0.5:
        market_avg = (
            (features["metaculus_p"] + features["polymarket_p"]) / 2
            if features["metaculus_p"] > 0 and features["polymarket_p"] > 0
            else max(features["metaculus_p"], features["polymarket_p"])
        )
        log_odds += 0.7 * _logit(market_avg)

    p = _sigmoid(log_odds)
    # Inject noise: outcome is stochastic given the features
    return int(random.random() < p)


def _sample_questions() -> list[str]:
    return [
        "Will there be a military escalation between Iran and Israel?",
        "Will Russia launch a major offensive in Eastern Ukraine?",
        "Will North Korea conduct a nuclear test?",
        "Will China begin military exercises near Taiwan?",
        "Will the ceasefire in Gaza hold for 30 days?",
        "Will Sudan's civil war spread to neighboring countries?",
        "Will there be a coup attempt in Venezuela?",
        "Will sanctions on Iran be lifted in the next quarter?",
        "Will the Houthis attack Red Sea shipping again?",
        "Will Turkey and Greece reach a maritime agreement?",
        "Will there be mass protests in Iran over the regime?",
        "Will the US impose new tariffs on Chinese semiconductors?",
        "Will Ethiopia and Eritrea resume armed conflict?",
        "Will Pakistan and India engage in cross-border fire?",
        "Will there be a diplomatic breakthrough in Syria?",
        "Will the Wagner Group expand operations in Africa?",
        "Will the UN Security Council pass a Gaza ceasefire resolution?",
        "Will there be a military confrontation in the South China Sea?",
        "Will Saudi Arabia and Iran restore full diplomatic relations?",
        "Will there be a major terrorist attack in Europe?",
    ]


def main():
    _TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    questions = _sample_questions()

    # Distribution: 30% high-risk, 30% low-risk, 40% uncertain
    scenarios = (
        [("high", _sample_high_risk)] * int(_N * 0.30)
        + [("low", _sample_low_risk)] * int(_N * 0.30)
        + [("uncertain", _sample_uncertain)] * int(_N * 0.40)
    )
    random.shuffle(scenarios)

    base_time = datetime.now(timezone.utc) - timedelta(days=90)
    outcomes = {0: 0, 1: 0}

    for i, (scenario_name, sampler) in enumerate(scenarios):
        features = sampler()
        outcome = _compute_outcome(features)
        outcomes[outcome] += 1

        ts = (base_time + timedelta(days=i)).strftime("%Y%m%dT%H%M%S")
        question = questions[i % len(questions)]
        slug = question.lower().replace(" ", "_").replace("?", "")[:50]

        record = {
            "question": question,
            "timestamp": (base_time + timedelta(days=i)).isoformat(),
            "scenario_type": scenario_name,
            "features": features,
            "outcome": outcome,
        }

        path = _TRAINING_DIR / f"{ts}_{slug}.json"
        path.write_text(json.dumps(record, indent=2))

    total = len(scenarios)
    print(f"Generated {total} synthetic training examples → {_TRAINING_DIR}")
    print(f"  YES outcomes: {outcomes[1]} ({outcomes[1]/total:.0%})")
    print(f"  NO  outcomes: {outcomes[0]} ({outcomes[0]/total:.0%})")
    print(f"\nRun: python main.py train")


if __name__ == "__main__":
    main()
