# Conformal Prediction Intervals

## What we guarantee

The oracle outputs an 80% prediction interval `[lo, hi]` for the calibrated probability.
Under the **exchangeability assumption** (see below), these intervals have marginal coverage:

```
P(Y ∈ C(X)) ≥ 1 - α = 0.80
```

This means that over many predictions drawn from the same distribution as the calibration set,
at least 80% of the true outcomes fall inside the reported interval.

## Method: asymmetric split conformal

We use **split conformal prediction** (Papadopoulos et al. 2002; Vovk et al. 2005).

The calibration set is the temporal holdout (newest 20% of labeled training examples).
Nonconformity scores are computed per class:

| Label | Score | Interpretation |
|-------|-------|----------------|
| y = 1 | s = 1 - p_cal | how much we underestimated a true positive |
| y = 0 | s = p_cal     | how much we overestimated a true negative |

At inference, with calibrated probability `p`:

```
q_pos = ceil((n_pos + 1) * (1 - α)) / n_pos -th order statistic of scores_pos
q_neg = ceil((n_neg + 1) * (1 - α)) / n_neg -th order statistic of scores_neg

lo = p - q_neg
hi = p + q_pos
```

The asymmetric construction lets the interval be narrower on one side when the model
is more consistently right in one label direction.

## Assumptions and limitations

### 1. Exchangeability (not i.i.d.)
Coverage holds under exchangeability of (X, Y) pairs — a weaker condition than i.i.d.
It does **not** hold if the test distribution drifts from the calibration distribution.
The drift monitor (Z-score on feature history) will flag when this assumption is at risk.

### 2. No conditional coverage
These are **marginal** intervals: averaged over all inputs X.
They do not guarantee coverage conditional on subgroups (e.g., "high-conflict countries only").
To get conditional coverage, Mondrian conformal prediction would be needed, which requires
substantially larger calibration sets than we currently have.

### 3. Calibration set size
With `n_pos + n_neg` calibration examples, the quantile estimate has standard error
roughly `√(α(1-α)/n)`. With n = 65 (current), SE ≈ 0.05 — so the nominal 80% CI
is accurate to ±5 percentage points of coverage. More training data directly tightens this.

### 4. Fallback behaviour
- If n_pos < 5 or n_neg < 5: falls back to symmetric conformal using all scores.
- If fewer than 5 scores total: falls back to beta distribution heuristic with
  effective N = 20 (deliberately conservative, no coverage guarantee).

The `ci_method` field in every prediction records which branch was used.

## References
- Vovk, Gammerman, Shafer (2005). *Algorithmic Learning in a Random World.*
- Angelopoulos & Bates (2023). *A Gentle Introduction to Conformal Prediction.*
- Barber et al. (2021). *Predictive Inference with the Jackknife+.*
