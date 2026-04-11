from __future__ import annotations

from scripts.baseline_v1 import run_baseline_v1


PREDICT_KEYS = {
    "raw_prob",
    "calibrated_prob",
    "ci_lo",
    "ci_hi",
    "ci_method",
    "answer",
    "untrained",
    "market_override",
}


def test_public_interface_run_baseline_v1_stable(monkeypatch):
    def fake_predict(_features, metaculus_p=None, polymarket_p=None):
        return {
            "raw_prob": 0.5,
            "calibrated_prob": 0.5,
            "ci_lo": 0.2,
            "ci_hi": 0.8,
            "ci_method": "heuristic",
            "answer": "YES",
            "untrained": False,
            "market_override": bool(metaculus_p or polymarket_p),
        }

    monkeypatch.setattr("scripts.baseline_v1.predict", fake_predict)

    out = run_baseline_v1({"x": 1.0})

    assert out["baseline"] == "v1"
    assert set(out.keys()) == {"baseline", "prediction", "variants", "market_coverage", "omitted_variants"}
    assert PREDICT_KEYS.issubset(set(out["prediction"].keys()))


def test_output_contract_compatible_with_predict(monkeypatch):
    def fake_predict(_features, metaculus_p=None, polymarket_p=None):
        return {
            "raw_prob": 0.4,
            "calibrated_prob": 0.45,
            "ci_lo": 0.2,
            "ci_hi": 0.7,
            "ci_method": "conformal",
            "answer": "NO",
            "untrained": False,
            "market_override": bool(metaculus_p or polymarket_p),
        }

    monkeypatch.setattr("scripts.baseline_v1.predict", fake_predict)
    out = run_baseline_v1({"x": 1.0}, metaculus_p=0.6)

    assert "model_only" in out["variants"]
    assert "market_blend" in out["variants"]
    for variant in out["variants"].values():
        assert PREDICT_KEYS <= set(variant.keys())


def test_no_market_variant_does_not_break_and_reports_zero_coverage(monkeypatch):
    def fake_predict(_features, metaculus_p=None, polymarket_p=None):
        return {
            "raw_prob": 0.3,
            "calibrated_prob": 0.3,
            "ci_lo": 0.1,
            "ci_hi": 0.6,
            "ci_method": "heuristic",
            "answer": "NO",
            "untrained": False,
            "market_override": False,
        }

    monkeypatch.setattr("scripts.baseline_v1.predict", fake_predict)
    out = run_baseline_v1({"x": 1.0}, metaculus_p=None, polymarket_p=None)

    assert out["market_coverage"] == 0.0
    assert "market_blend" not in out["variants"]
    assert "market_blend" in out["omitted_variants"]
