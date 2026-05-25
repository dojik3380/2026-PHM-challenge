"""Stage 2: convert per-window HI predictions to RUL via curve extrapolation.

The neural network in model.py predicts a Health Indicator in [0, 1] per
window. This module fits that HI sequence to a simple curve and extrapolates
to HI_FAILURE_THRESHOLD to estimate the failure time, from which RUL is
computed as t_failure - t_now.

Two strategies, ordered from simple to robust:

  - linear_tail : robust linear fit on the most-recent K windows. Best when
                  the recent HI slope is informative (typical late-life
                  behaviour).
  - power_fit   : power-law HI(t) = a * (t/T)^b fit on the full history. Good
                  when HI labels were generated with HI_LABEL_MODE='power' and
                  long history is available.

The library defaults to linear_tail because it survives best with noisy
predictions on tiny data.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from config import HI_FAILURE_THRESHOLD


_LINEAR_TAIL_K = 16        # how many trailing windows to fit
_MIN_SLOPE = 1e-9          # below this we treat HI as flat -> very large RUL


def _linear_tail_t_failure(
    times: np.ndarray,
    hi: np.ndarray,
    threshold: float,
    k: int = _LINEAR_TAIL_K,
) -> float:
    """Robust linear extrapolation of HI to threshold.

    Returns t_failure. If the slope is non-positive (HI not growing), returns
    a large finite number so RUL ~ remaining_horizon and stays optimistic.
    """
    n = len(times)
    if n < 2:
        # No trend -> assume HI grows from current to threshold over the
        # remaining typical lifetime. Without history just return a large
        # number so RUL = t_failure - t_now stays positive.
        return float(times[-1] + 1e6)
    use = min(k, n)
    t = times[-use:].astype(np.float64)
    h = np.clip(hi[-use:].astype(np.float64), 0.0, 1.0)

    # Simple least-squares slope/intercept.
    t_mean = t.mean()
    h_mean = h.mean()
    denom = ((t - t_mean) ** 2).sum()
    if denom < 1e-12:
        return float(t[-1] + 1e6)
    slope = ((t - t_mean) * (h - h_mean)).sum() / denom
    intercept = h_mean - slope * t_mean

    if slope <= _MIN_SLOPE:
        # HI not increasing in recent window -> can't extrapolate meaningfully.
        # Use the most recent hi to estimate "fraction-of-life-remaining"
        # under a uniform-progress assumption: t_failure = t_now / max(hi_now, eps).
        hi_now = max(float(h[-1]), 1e-3)
        if hi_now >= threshold:
            return float(t[-1])
        return float(t[-1] * threshold / hi_now)

    t_fail = (threshold - intercept) / slope
    return float(max(t_fail, t[-1]))


def _power_t_failure(
    times: np.ndarray,
    hi: np.ndarray,
    threshold: float,
) -> float:
    """Fit HI(t) = a * t^b on positive values; extrapolate to threshold."""
    t = times.astype(np.float64)
    h = np.clip(hi.astype(np.float64), 1e-6, 0.9999)
    mask = (t > 0) & (h > 1e-6)
    if mask.sum() < 3:
        return _linear_tail_t_failure(times, hi, threshold)
    lt = np.log(t[mask])
    lh = np.log(h[mask])
    # log h = log a + b * log t
    A = np.vstack([np.ones_like(lt), lt]).T
    try:
        coef, *_ = np.linalg.lstsq(A, lh, rcond=None)
    except np.linalg.LinAlgError:
        return _linear_tail_t_failure(times, hi, threshold)
    log_a, b = float(coef[0]), float(coef[1])
    if b <= _MIN_SLOPE:
        return _linear_tail_t_failure(times, hi, threshold)
    # threshold = a * t_fail^b -> t_fail = (threshold / a) ^ (1/b)
    return float(np.exp((np.log(threshold) - log_a) / b))


def hi_sequence_to_rul(
    times: np.ndarray,
    hi: np.ndarray,
    t_now: float,
    case_max: Optional[float] = None,
    threshold: float = HI_FAILURE_THRESHOLD,
    method: str = "linear_tail",
) -> float:
    """Return predicted t_failure given an HI history. RUL = t_failure - t_now.

    times, hi: per-window arrays for a single case, in ascending time order.
    t_now    : current time (typically times[-1]).
    case_max : optional, used as a sanity bound for t_failure.
    method   : 'linear_tail' (default) or 'power'.
    """
    times = np.asarray(times, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    if method == "power":
        t_fail = _power_t_failure(times, hi, threshold)
    else:
        t_fail = _linear_tail_t_failure(times, hi, threshold)

    if case_max is not None and case_max > 0:
        # Bound the prediction: a flat HI history would otherwise produce
        # absurd RUL values. 1.5x is conservative: even the longest training
        # case in this dataset is ~1.4x the population mean, so capping at 1.5x
        # avoids catastrophic over-prediction without disallowing realistic
        # outliers. The asymmetric metric punishes over-prediction much harder
        # than under-prediction, so the cap should err conservative.
        t_fail = min(t_fail, max(t_now, case_max * 1.5))

    return float(max(t_fail, t_now))


def calibrate_hi(hi_pred: np.ndarray, anchors: dict) -> np.ndarray:
    """Linear map from the model's compressed output range to the true HI scale.

    The HI regressor trained with MSE on 7 cases typically produces predictions
    in a narrow band (e.g. [0.4, 0.7]) even though the truth labels span
    [0, ~0.9]. This is the small-data mean-fallback failure mode: ranking is
    preserved but absolute scale collapses. We undo it post-hoc by anchoring:
        pred_healthy -> true_healthy
        pred_failure -> true_failure
    and linearly mapping in between. Anchors are computed at training time
    from first/last windows of training cases and saved with the checkpoint.
    """
    pred_h = anchors["pred_healthy"]
    pred_f = anchors["pred_failure"]
    true_h = anchors["true_healthy"]
    true_f = anchors["true_failure"]
    span = pred_f - pred_h
    if abs(span) < 1e-6:
        return np.clip(np.asarray(hi_pred, dtype=np.float64), 0.0, 1.0)
    cal = (np.asarray(hi_pred, dtype=np.float64) - pred_h) / span * (true_f - true_h) + true_h
    return np.clip(cal, 0.0, 1.0)


def predict_rul_per_window(
    times: np.ndarray,
    hi: np.ndarray,
    case_max: Optional[float] = None,
    method: str = "linear_tail",
) -> np.ndarray:
    """For each window i, predict RUL using HI history [0..i] only."""
    times = np.asarray(times, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    order = np.argsort(times)
    t_sorted = times[order]
    h_sorted = hi[order]
    rul_sorted = np.zeros_like(t_sorted)
    for i in range(len(t_sorted)):
        t_fail = hi_sequence_to_rul(
            t_sorted[: i + 1], h_sorted[: i + 1],
            t_now=float(t_sorted[i]),
            case_max=case_max, method=method,
        )
        rul_sorted[i] = max(0.0, t_fail - t_sorted[i])
    out = np.zeros_like(rul_sorted)
    out[order] = rul_sorted
    return out
