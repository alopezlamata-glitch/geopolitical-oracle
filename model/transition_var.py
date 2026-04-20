"""
VAR (Vector Autoregression) transition model for world state.

Architecture:
  One VAR per (entity, domain) fitted on world_state_history.
  Domains: conflict | political | economic
  Each VAR maps state[t-p..t] → state[t+1]

  With < MIN_HISTORY days of data:
    Falls back to a random-walk model using the single-step delta observed.
    This gives honest wide CIs when history is short.

  With >= MIN_HISTORY days:
    Fits statsmodels VAR with automatically selected lag order (AIC, max_lag=7).
    Stores fitted model as pickle at data/model/var_{entity_id}_{domain}.pkl.

Usage:
  fit_all()                              # fit for all entities in world_state_history
  predict(entity_id, domain, horizon)   # forecast next `horizon` days
"""
from __future__ import annotations

import hashlib
import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_MODEL_DIR = Path(__file__).parent.parent / "data" / "model" / "var"
_MODEL_DIR.mkdir(parents=True, exist_ok=True)

MIN_HISTORY = 14   # days of history required to fit a real VAR
MAX_LAG     = 7    # maximum VAR lag order tested
MIN_OBS_FOR_VAR = 20  # statsmodels VAR is unstable below this

# ── Feature groups per domain ─────────────────────────────────────────────────

DOMAIN_FEATURES: dict[str, list[str]] = {
    "conflict": [
        "military_count_7d",
        "protest_count_7d",
        "ceasefire_count_7d",
        "escalation_index",
        "military_intensity_7d",
        "event_velocity_7d",
        "military_accel",
        "military_share_7d",
        "avg_polarity_7d",
        "tone_trend",
    ],
    "political": [
        "pol_resignation_signals",
        "pol_approval_pressure",
        "pol_coalition_stability",
        "pol_electoral_proximity",
        "pol_judicial_pressure",
    ],
    "economic": [
        "eco_rate_change_prob",
        "eco_gdp_momentum",
        "eco_debt_stress",
        "eco_market_volatility",
        "eco_policy_uncertainty",
        "fred_vix",
        "fred_yield_spread",
    ],
}

ALL_DOMAINS = list(DOMAIN_FEATURES.keys())


# ── Persistence helpers ───────────────────────────────────────────────────────

def _model_path(entity_id: str, domain: str) -> Path:
    return _MODEL_DIR / f"var_{entity_id}_{domain}.pkl"


def _meta_path(entity_id: str, domain: str) -> Path:
    return _MODEL_DIR / f"var_{entity_id}_{domain}.json"


def save_model(entity_id: str, domain: str, model_obj: dict) -> None:
    path = _model_path(entity_id, domain)
    with open(path, "wb") as f:
        pickle.dump(model_obj, f)
    meta_path = _meta_path(entity_id, domain)
    meta = {k: v for k, v in model_obj.items() if k != "fitted_model"}
    meta_path.write_text(json.dumps(meta, default=str))


def load_model(entity_id: str, domain: str) -> Optional[dict]:
    path = _model_path(entity_id, domain)
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.warning("var: failed to load %s/%s: %s", entity_id, domain, e)
        return None


# ── History loading ───────────────────────────────────────────────────────────

def load_history(entity_id: str, n_days: int = 180) -> Optional[np.ndarray]:
    """
    Load world_state_history for entity_id, sorted by date ascending.
    Returns (n_obs, n_all_features) array or None if DB unavailable.
    """
    from datetime import date, timedelta

    cutoff = date.today() - timedelta(days=n_days)
    try:
        from data_layer.db import get_db, table_exists
        if not table_exists("world_state_history"):
            return None
        db = get_db()
        rows = db.execute("""
            SELECT features
            FROM world_state_history
            WHERE entity_id = ?
              AND as_of_date >= ?
            ORDER BY as_of_date ASC
        """, [entity_id, cutoff]).fetchall()
    except Exception as e:
        logger.debug("var: load_history failed: %s", e)
        return None

    if not rows:
        return None

    # Parse JSON feature dicts
    records = []
    for (features_json,) in rows:
        try:
            d = json.loads(features_json) if isinstance(features_json, str) else features_json
            records.append(d)
        except Exception:
            continue

    return records  # list[dict]


def _records_to_matrix(records: list[dict], features: list[str]) -> np.ndarray:
    """Convert list of feature dicts to (n_obs, n_features) float matrix."""
    mat = []
    for rec in records:
        row = [float(rec.get(f, 0.0) or 0.0) for f in features]
        mat.append(row)
    return np.array(mat, dtype=float)


# ── Model fitting ─────────────────────────────────────────────────────────────

def fit(entity_id: str, domain: str, n_days: int = 180) -> Optional[dict]:
    """
    Fit a VAR model for (entity_id, domain).

    Returns a model_obj dict with either:
      - type='var': statsmodels fitted VAR result + metadata
      - type='rw':  random-walk fallback using observed deltas
      - None: not enough data even for random-walk
    """
    features = DOMAIN_FEATURES[domain]
    records = load_history(entity_id, n_days=n_days)

    if records is None or len(records) < 1:
        logger.debug("var: %s/%s — no history", entity_id, domain)
        return None

    X = _records_to_matrix(records, features)
    n_obs = len(X)

    # With only 1 observation, use a "static" model: no drift, covariance from
    # feature value spread as a rough uncertainty estimate.
    if n_obs == 1:
        variance_est = (X[0] * 0.1) ** 2 + 1e-6  # 10% of current value as uncertainty
        model_obj = {
            "type":        "rw",
            "entity_id":   entity_id,
            "domain":      domain,
            "features":    features,
            "n_obs":       n_obs,
            "lag_order":   1,
            "last_obs":    X[-1:].tolist(),
            "delta_mean":  [0.0] * len(features),
            "delta_cov":   np.diag(variance_est).tolist(),
            "feature_means": X[0].tolist(),
            "feature_stds":  np.sqrt(variance_est).tolist(),
        }
        save_model(entity_id, domain, model_obj)
        logger.info("var: fitted %s/%s as static (1 obs)", entity_id, domain)
        return model_obj

    if n_obs >= MIN_OBS_FOR_VAR:
        # Fit real VAR
        try:
            from statsmodels.tsa.vector_ar.var_model import VAR
            model = VAR(X)
            # Select lag order by AIC, cap at MAX_LAG (and at most n_obs // 3)
            max_lag = min(MAX_LAG, n_obs // 3)
            if max_lag < 1:
                max_lag = 1
            results = model.fit(maxlags=max_lag, ic="aic", verbose=False)
            lag_order = results.k_ar

            model_obj = {
                "type":       "var",
                "entity_id":  entity_id,
                "domain":     domain,
                "features":   features,
                "n_obs":      n_obs,
                "lag_order":  lag_order,
                "fitted_model": results,
                "last_obs":   X[-lag_order:].tolist(),   # needed for forecasting
                "feature_means": X.mean(axis=0).tolist(),
                "feature_stds":  X.std(axis=0).tolist(),
            }
            save_model(entity_id, domain, model_obj)
            logger.info(
                "var: fitted %s/%s  lag=%d  n_obs=%d",
                entity_id, domain, lag_order, n_obs,
            )
            return model_obj
        except Exception as e:
            logger.warning("var: VAR fit failed for %s/%s: %s", entity_id, domain, e)

    # Fallback: random-walk model (mean + covariance of 1-day deltas)
    if n_obs >= 2:
        deltas = X[1:] - X[:-1]
        delta_mean = deltas.mean(axis=0)
        # Covariance of deltas (diagonal if n < features)
        if n_obs >= len(features) + 2:
            delta_cov = np.cov(deltas.T)
        else:
            delta_cov = np.diag(deltas.var(axis=0) + 1e-8)

        model_obj = {
            "type":        "rw",
            "entity_id":   entity_id,
            "domain":      domain,
            "features":    features,
            "n_obs":       n_obs,
            "lag_order":   1,
            "last_obs":    X[-1:].tolist(),
            "delta_mean":  delta_mean.tolist(),
            "delta_cov":   delta_cov.tolist() if hasattr(delta_cov, 'tolist') else list(delta_cov),
            "feature_means": X.mean(axis=0).tolist(),
            "feature_stds":  X.std(axis=0).tolist(),
        }
        save_model(entity_id, domain, model_obj)
        logger.info(
            "var: fitted %s/%s as random-walk  n_obs=%d",
            entity_id, domain, n_obs,
        )
        return model_obj

    return None


def fit_all(entity_ids: Optional[list[str]] = None) -> dict[str, int]:
    """
    Fit VAR models for all (entity, domain) pairs in world_state_history.
    Returns {entity_id: n_domains_fitted}.
    """
    try:
        from data_layer.db import get_db, table_exists
        if not table_exists("world_state_history"):
            return {}
        db = get_db()
        if entity_ids:
            rows = [(eid,) for eid in entity_ids]
        else:
            rows = db.execute(
                "SELECT DISTINCT entity_id FROM world_state_history"
            ).fetchall()
    except Exception as e:
        logger.error("var: fit_all DB query failed: %s", e)
        return {}

    results: dict[str, int] = {}
    for (eid,) in rows:
        n_fitted = 0
        for domain in ALL_DOMAINS:
            m = fit(eid, domain)
            if m is not None:
                n_fitted += 1
        results[eid] = n_fitted
        if n_fitted:
            logger.info("var: %s → %d/%d domains fitted", eid, n_fitted, len(ALL_DOMAINS))

    return results


# ── Forecasting ───────────────────────────────────────────────────────────────

def predict(
    entity_id: str,
    domain: str,
    horizon_days: int,
    n_samples: int = 100,
    seed: Optional[int] = None,
) -> Optional[np.ndarray]:
    """
    Forecast the domain feature vector for the next `horizon_days` days.

    Returns array of shape (horizon_days, n_features) representing the
    expected trajectory, or None if no model exists.

    For stochastic models, returns the mean of `n_samples` draws.
    Use predict_distribution() for full posterior samples.
    """
    model_obj = load_model(entity_id, domain)
    if model_obj is None:
        return None

    features = model_obj["features"]
    n_feat = len(features)

    if model_obj["type"] == "var":
        try:
            fitted = model_obj["fitted_model"]
            last_obs = np.array(model_obj["last_obs"])
            # statsmodels VAR forecast returns (horizon, n_features) point forecast
            fc = fitted.forecast(last_obs, steps=horizon_days)
            # Clip to plausible ranges
            fc = np.clip(fc, -5.0, 100.0)
            return fc
        except Exception as e:
            logger.warning("var: forecast failed for %s/%s: %s", entity_id, domain, e)
            # Fall through to random-walk

    # Random-walk: mean drift + noise
    rng = np.random.default_rng(seed)
    delta_mean = np.array(model_obj.get("delta_mean", [0.0] * n_feat))
    delta_cov_raw = model_obj.get("delta_cov", np.eye(n_feat).tolist())
    delta_cov = np.array(delta_cov_raw)
    if delta_cov.ndim == 1:
        delta_cov = np.diag(delta_cov)

    current = np.array(model_obj["last_obs"][-1])

    trajectories = []
    for _ in range(n_samples):
        traj = []
        state = current.copy()
        for _ in range(horizon_days):
            noise = rng.multivariate_normal(delta_mean, delta_cov)
            state = state + noise
            state = np.clip(state, -5.0, 100.0)
            traj.append(state.copy())
        trajectories.append(traj)

    return np.mean(trajectories, axis=0)  # (horizon, n_features)


def predict_distribution(
    entity_id: str,
    domain: str,
    horizon_days: int,
    n_samples: int = 200,
    seed: Optional[int] = None,
) -> Optional[np.ndarray]:
    """
    Sample `n_samples` trajectories from the VAR posterior.
    Returns array of shape (n_samples, horizon_days, n_features).
    """
    model_obj = load_model(entity_id, domain)
    if model_obj is None:
        return None

    features = model_obj["features"]
    n_feat = len(features)
    rng = np.random.default_rng(seed)

    if model_obj["type"] == "var":
        try:
            fitted = model_obj["fitted_model"]
            last_obs = np.array(model_obj["last_obs"])
            sigma_u = fitted.sigma_u  # residual covariance (n_feat, n_feat)

            samples = []
            for _ in range(n_samples):
                traj = []
                history = last_obs.copy()
                # Propagate one step at a time, adding residual noise
                for h in range(horizon_days):
                    point = fitted.forecast(history, steps=1)[0]
                    noise = rng.multivariate_normal(np.zeros(n_feat), sigma_u)
                    step = np.clip(point + noise, -5.0, 100.0)
                    traj.append(step)
                    # Advance history window
                    history = np.vstack([history[1:], step])
                samples.append(traj)
            return np.array(samples)  # (n_samples, horizon, n_features)
        except Exception as e:
            logger.warning("var: distribution forecast failed for %s/%s: %s", entity_id, domain, e)

    # Random-walk fallback
    delta_mean = np.array(model_obj.get("delta_mean", [0.0] * n_feat))
    delta_cov_raw = model_obj.get("delta_cov", np.eye(n_feat).tolist())
    delta_cov = np.array(delta_cov_raw)
    if delta_cov.ndim == 1:
        delta_cov = np.diag(delta_cov)

    current = np.array(model_obj["last_obs"][-1])
    samples = []
    for _ in range(n_samples):
        traj = []
        state = current.copy()
        for _ in range(horizon_days):
            noise = rng.multivariate_normal(delta_mean, delta_cov)
            state = state + noise
            state = np.clip(state, -5.0, 100.0)
            traj.append(state.copy())
        samples.append(traj)
    return np.array(samples)  # (n_samples, horizon, n_features)


def list_fitted_models() -> list[dict]:
    """List all fitted VAR models with their metadata."""
    models = []
    for meta_path in sorted(_MODEL_DIR.glob("var_*.json")):
        try:
            meta = json.loads(meta_path.read_text())
            models.append(meta)
        except Exception:
            continue
    return models
