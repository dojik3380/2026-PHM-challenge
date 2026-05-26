"""Stage 2: HI 궤적 → RUL.

B4 파이프라인 (Nguyen et al. 2025, IEEE Access — DOI 10.1109/ACCESS.2025.3643521):
  1. Joseph-form Kalman filter 로 HI 노이즈 제거 (positive-definite covariance 보존).
  2. np.maximum.accumulate 로 단조성 강제.
  3. FDP Trigger: KF_Filtered_HI > 0.15 일 때부터 피팅 시작.
  4. scipy.optimize.curve_fit 로 지수 가중치 곡선 피팅 (a*e^(bt)+c).
  5. 점 추정 RUL 반환.

설계 의도:
  - 기존 엑스포넨셜 곡선에 가중치를 부여하여 가장 최근 데이터에 피팅 가중치를 둡니다.
  - 수학적 폭발 방지를 위해 bounds 를 제한합니다.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning
import warnings

warnings.filterwarnings("ignore", category=OptimizeWarning)

from config import (
    HI_FAILURE_THRESHOLD, 
    STAGE2_FALLBACK_RUL_CAP,
    STAGE2_RUL_BEFORE_FDP,
    STAGE2_KF_Q,
    STAGE2_KF_R,
    STAGE2_KF_P0,
    STAGE2_KF_OUTLIER_LO,
    STAGE2_KF_OUTLIER_STATE,
    STAGE2_WLS_RECENCY_DECAY,
    STAGE2_WLS_ROLLING_WINDOW,
    STAGE2_FDP_THRESHOLD,
)

# ---------------------------------------------------------------------------
# 상수 (imported from config.py for backward compatibility in this file)
# ---------------------------------------------------------------------------
_FALLBACK_RUL_CAP   = STAGE2_FALLBACK_RUL_CAP
_RUL_BEFORE_FDP     = STAGE2_RUL_BEFORE_FDP

_KF_Q               = STAGE2_KF_Q
_KF_R               = STAGE2_KF_R
_KF_P0              = STAGE2_KF_P0
_KF_OUTLIER_LO      = STAGE2_KF_OUTLIER_LO
_KF_OUTLIER_STATE   = STAGE2_KF_OUTLIER_STATE

_WLS_RECENCY_DECAY  = STAGE2_WLS_RECENCY_DECAY
_WLS_ROLLING_WINDOW = STAGE2_WLS_ROLLING_WINDOW
_FDP_THRESHOLD      = STAGE2_FDP_THRESHOLD


# ---------------------------------------------------------------------------
# 1) Joseph-form Kalman filter (1D state)
# ---------------------------------------------------------------------------
def kalman_filter_hi(
    hi_raw: np.ndarray,
    Q: float = _KF_Q,
    R: float = _KF_R,
    P0: float = _KF_P0,
    outlier_lo: float = _KF_OUTLIER_LO,
    outlier_state: float = _KF_OUTLIER_STATE,
) -> np.ndarray:
    """1차원 Joseph-form KF + dip outlier rejection."""
    n = len(hi_raw)
    out = np.zeros(n, dtype=np.float64)
    if n == 0:
        return out

    x_prev = float(hi_raw[0])
    P_prev = float(P0)
    out[0] = x_prev
    for k in range(1, n):
        # Predict
        x_pred = x_prev
        P_pred = P_prev + Q
        obs = float(hi_raw[k])
        # Outlier dip: state 가 충분히 올라온 뒤 raw 가 갑자기 떨어지면 무시
        if x_prev > outlier_state and obs < outlier_lo:
            # measurement skip — state 와 P 는 predict 결과 유지
            x_prev = x_pred
            P_prev = P_pred
        else:
            Kg = P_pred / (P_pred + R)
            P_prev = (1.0 - Kg) ** 2 * P_pred + Kg ** 2 * R
            x_prev = x_pred + Kg * (obs - x_pred)
        out[k] = x_prev
    return out


# ---------------------------------------------------------------------------
# 2) Curve Fitting
# ---------------------------------------------------------------------------
def _fit_log_linear_rul(
    times: np.ndarray,
    hi_filtered: np.ndarray,
    t_now: float,
    failure_threshold: float,
    fallback_rul_cap: float,
) -> float:
    n = len(times)
    if n < 3:
        return fallback_rul_cap

    # Macro Fitting: 300스텝(약 50분) 고정 광역 윈도우로 단순화하여 노이즈 스파이크 저항력 확보
    rolling = min(300, n)
        
    t_window = times[-rolling:]
    hi_window = hi_filtered[-rolling:]
    
    # Local Monotonic Accumulate (윈도우 내부에서만 강제 단조 증가)
    hi_window = np.maximum.accumulate(hi_window)
    
    # Normalized time for fitting stability
    t_offset = t_window[0]
    t_norm = t_window - t_offset

    # Exponential weighting for WLS linear regression
    w = np.exp(_WLS_RECENCY_DECAY * np.arange(rolling, dtype=np.float64) / max(rolling - 1, 1))

    # Log-linear transformation (y = ln(HI))
    # Add epsilon to prevent log(0)
    epsilon = 1e-6
    y_log = np.log(np.clip(hi_window, epsilon, None))
    
    try:
        # np.polyfit with degree 1: minimizes sum(w * (y - (b1*x + b0))^2)
        # Returns [slope, intercept] = [b1, b0]
        coeffs = np.polyfit(t_norm, y_log, 1, w=w)
        b1, b0 = coeffs[0], coeffs[1]
        
        # Degradation implies HI should increase over time -> b1 should be positive
        if b1 <= 1e-8:
            return fallback_rul_cap
            
        # Extrapolate to threshold
        y_target = np.log(failure_threshold)
        if y_target <= b0:
            return 0.0 # Already failed
            
        t_fail_norm = (y_target - b0) / b1
        t_fail = t_fail_norm + t_offset
        rul = float(max(t_fail - t_now, 0.0))
        
        cap = max(2.0 * t_now, 1.0)
        return min(rul, cap, fallback_rul_cap)
        
    except (RuntimeError, ValueError, TypeError, np.linalg.LinAlgError):
        return fallback_rul_cap


# ---------------------------------------------------------------------------
# 3) Public API: 단일 RUL, 전체 trajectory
# ---------------------------------------------------------------------------
def fit_stage2_rul(
    times: np.ndarray,
    hi_preds: np.ndarray,
    failure_threshold: float = HI_FAILURE_THRESHOLD,
    fallback_rul_cap: float = _FALLBACK_RUL_CAP,
) -> float:
    """현재까지의 HI 시퀀스 → 단일 RUL 점 추정. inference 진입점."""
    times = np.asarray(times, dtype=np.float64)
    hi = np.asarray(hi_preds, dtype=np.float64)
    if len(times) == 0:
        return fallback_rul_cap

    order = np.argsort(times)
    t = times[order]
    h = np.clip(hi[order], 0.0, 1.0)

    hi_kf = kalman_filter_hi(h)
    hi_f = np.maximum.accumulate(hi_kf)
    
    t_now = float(t[-1])
    
    if hi_f[-1] <= _FDP_THRESHOLD:
        # Bayesian Linear Regression (Data-driven + Population Prior)
        lam = 1e11
        slope_prior = 0.5 / 72000.0
        sum_t_hi = np.sum(t * hi_kf)
        sum_t2 = np.sum(t**2)
        
        slope = (sum_t_hi + lam * slope_prior) / (sum_t2 + lam)
        slope = max(slope, 1e-8)
        t_max = 0.5 / slope
        return float(max(t_max - t_now, 0.0))
        
    rul = _fit_log_linear_rul(t, hi_kf, t_now, failure_threshold, fallback_rul_cap)
    return rul


def compute_stage2_trajectory(
    times: np.ndarray,
    hi_preds: np.ndarray,
    failure_threshold: float = HI_FAILURE_THRESHOLD,
    fallback_rul_cap: float = _FALLBACK_RUL_CAP,
) -> np.ndarray:
    """전체 시퀀스에 대해 online RUL trajectory 생성.
    
    각 timestep i 에서 [0..i] 데이터만 사용 → causal.
    """
    times = np.asarray(times, dtype=np.float64)
    hi = np.asarray(hi_preds, dtype=np.float64)
    N = len(times)
    if N == 0:
        return np.zeros(0, dtype=np.float64)

    order = np.argsort(times)
    t_s = times[order]
    h_s = np.clip(hi[order], 0.0, 1.0)

    hi_kf = kalman_filter_hi(h_s)
    hi_f = np.maximum.accumulate(hi_kf)

    rul_s = np.full(N, fallback_rul_cap, dtype=np.float64)
    last_valid_rul = float(_RUL_BEFORE_FDP)
    last_valid_time = 0.0

    for i in range(N):
        t_now = float(t_s[i])

        if hi_f[i] <= _FDP_THRESHOLD:
            # Bayesian Linear Regression (Data-driven + Population Prior)
            t_healthy = t_s[: i + 1]
            hi_healthy = hi_kf[: i + 1]
            
            lam = 1e11
            slope_prior = 0.5 / 72000.0
            sum_t_hi = np.sum(t_healthy * hi_healthy)
            sum_t2 = np.sum(t_healthy**2)
            
            slope = (sum_t_hi + lam * slope_prior) / (sum_t2 + lam)
            slope = max(slope, 1e-8)
            t_max = 0.5 / slope
            rul = max(t_max - t_now, 0.0)
                
            rul_s[i] = rul
            last_valid_rul = rul
            last_valid_time = t_now
            continue
            
        rul = _fit_log_linear_rul(t_s[: i + 1], hi_kf[: i + 1], t_now, failure_threshold, fallback_rul_cap)
        
        dt = t_now - last_valid_time
        if last_valid_rul not in (_FALLBACK_RUL_CAP, _RUL_BEFORE_FDP):
            expected_rul = max(last_valid_rul - dt, 0.0)
            margin = max(expected_rul * 0.05, 50.0)
            if rul > expected_rul + margin:
                rul = expected_rul
                
        rul_s[i] = rul
        last_valid_rul = rul
        last_valid_time = t_now

    rul_out = np.zeros(N, dtype=np.float64)
    rul_out[order] = rul_s
    return rul_out
