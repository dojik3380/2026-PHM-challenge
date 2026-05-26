"""Stage 2: HI 궤적 지수 피팅 → RUL 외삽.

피팅 모델: f(t) = a·e^(b·t) + c   (scipy.optimize.curve_fit, TRF)

실패 시점:
    f(T_failure) = 1.0  →  T_failure = (1/b)·ln((1.0 - c) / a)
    RUL = max(0, T_failure - t_current)
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

from config import HI_FAILURE_THRESHOLD

# ---------------------------------------------------------------------------
# 전역 상수
# ---------------------------------------------------------------------------
_FALLBACK_RUL_CAP  = 200_000.0   
_RUL_BEFORE_FDP    = 30_000.0    
_FDP_THRESHOLD     = 0.15        
_ROLLING_WINDOW    = 30          
_MIN_HISTORY       = 20          


def ema(x: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    """Exponential Moving Average trajectory smoothing."""
    y = np.zeros_like(x)
    if len(x) == 0:
        return y
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i-1]
    return y


def _exp_model(t: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """f(t) = a·e^(b·t) + c"""
    return a * np.exp(b * t) + c


def _t_failure(a: float, b: float, c: float, threshold: float = 1.0) -> float:
    """f(T) = threshold → T = (1/b)·ln((threshold - c) / a)"""
    inner = (threshold - c) / a
    if inner <= 0.0 or b <= 0.0:
        return np.nan
    return float(np.log(inner) / b)


def _fit_and_validate(t_fit: np.ndarray, h_fit: np.ndarray, t_now: float, threshold: float) -> float:
    """Core fitting and validation logic for a rolling window of t and h.
    Returns valid RUL or np.nan if rejected.
    """
    if len(t_fit) < 3:
        return np.nan

    t_scale = max(t_now, 1.0)
    t_norm  = t_fit / t_scale

    bounds = (
        [0.0, 1e-6, -0.2],
        [1.0, 1.0, 0.99]
    )
    p0 = [0.05, 1e-3, 0.0]

    try:
        popt, _ = curve_fit(
            _exp_model, t_norm, h_fit,
            p0=p0, bounds=bounds,
            method="trf",
            maxfev=20000
        )
        a, b, c = popt
        
        # 6. Invalid Fit Rejection
        if abs(b) < 1e-5:
            return np.nan
            
        t_fail_norm = _t_failure(a, b, c, threshold)
        if np.isnan(t_fail_norm):
            return np.nan
            
        t_fail = t_fail_norm * t_scale
        if t_fail < t_now:
            return np.nan
            
        max_allowed_failure_time = 3 * t_now
        if t_fail > max_allowed_failure_time:
            return np.nan
            
        # 7. Fit Quality Validation
        h_pred_fit = _exp_model(t_norm, a, b, c)
        if np.var(h_fit) < 1e-8:
            r2 = 0.0
        else:
            r2 = r2_score(h_fit, h_pred_fit)
            
        if r2 < 0.8:
            return np.nan

        return float(max(0.0, t_fail - t_now))

    except Exception:
        return np.nan


def fit_stage2_rul(
    times: np.ndarray,
    hi_preds: np.ndarray,
    failure_threshold: float = HI_FAILURE_THRESHOLD,
    fallback_rul_cap: float = _FALLBACK_RUL_CAP,
) -> float:
    """Predicts a single scalar RUL using the provided history.
    """
    times    = np.asarray(times, dtype=np.float64)
    hi_preds = np.asarray(hi_preds, dtype=np.float64)

    order = np.argsort(times)
    t = times[order]
    
    # 1. EMA Smoothing
    h_ema = ema(hi_preds[order], alpha=0.2)
    
    # 2. Hard Monotonic Inference Filter
    h = np.maximum.accumulate(h_ema)

    if len(t) == 0:
        return fallback_rul_cap

    t_now = float(t[-1])
    h_now = float(h[-1])

    # 3. Early-Life Extrapolation Ban
    if h_now < _FDP_THRESHOLD:
        return _RUL_BEFORE_FDP
    
    if len(t) < _MIN_HISTORY:
        return fallback_rul_cap

    if h_now >= failure_threshold:
        return 0.0

    # 4. Rolling-Window Fitting
    t_fit = t[-_ROLLING_WINDOW:]
    h_fit = h[-_ROLLING_WINDOW:]

    rul = _fit_and_validate(t_fit, h_fit, t_now, failure_threshold)
    
    if np.isnan(rul):
        return fallback_rul_cap
    return min(rul, fallback_rul_cap)


def compute_stage2_trajectory(
    times: np.ndarray,
    hi_preds: np.ndarray,
    failure_threshold: float = HI_FAILURE_THRESHOLD,
    fallback_rul_cap: float = _FALLBACK_RUL_CAP,
) -> np.ndarray:
    """Calculates online RUL trajectory for a full case sequence."""
    times    = np.asarray(times, dtype=np.float64)
    hi_preds = np.asarray(hi_preds, dtype=np.float64)

    N     = len(times)
    order = np.argsort(times)
    t_s   = times[order]
    
    # Apply EMA and Monotonic filtering over the sequence
    h_ema = ema(hi_preds[order], alpha=0.2)
    h_mono = np.maximum.accumulate(h_ema)
    
    rul_s = np.zeros(N, dtype=np.float64)
    
    last_valid_rul = fallback_rul_cap
    last_valid_time = 0.0

    for i in range(N):
        h_now = float(h_mono[i])
        t_now = float(t_s[i])

        if h_now < _FDP_THRESHOLD:
            rul_s[i] = _RUL_BEFORE_FDP
            last_valid_rul = _RUL_BEFORE_FDP
            last_valid_time = t_now
            continue

        if h_now >= failure_threshold:
            rul_s[i] = 0.0
            continue

        if i + 1 < _MIN_HISTORY:
            rul_s[i] = fallback_rul_cap
            last_valid_rul = fallback_rul_cap
            last_valid_time = t_now
            continue

        start = max(0, i + 1 - _ROLLING_WINDOW)
        t_fit = t_s[start : i + 1]
        h_fit = h_mono[start : i + 1]

        rul = _fit_and_validate(t_fit, h_fit, t_now, failure_threshold)
        
        if np.isnan(rul):
            # Fallback to the last valid RUL, minus the time elapsed since then
            elapsed = t_now - last_valid_time
            rul_s[i] = max(0.0, last_valid_rul - elapsed)
        else:
            rul_s[i] = min(rul, fallback_rul_cap)
            last_valid_rul = rul_s[i]
            last_valid_time = t_now

    rul_out = np.zeros(N, dtype=np.float64)
    rul_out[order] = rul_s
    return rul_out
