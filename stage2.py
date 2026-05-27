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
    STAGE2_FDP_BASELINE_N,
    STAGE2_FDP_K_SIGMA,
    STAGE2_FDP_MIN,
    STAGE2_FDP_MAX,
    STAGE2_FALLBACK_FLOOR,
    STAGE2_LIFETIME_PRIOR_QUANTILE,
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
# 1.5) Dynamic FDP threshold (3σ rule, 학계 표준)
# ---------------------------------------------------------------------------
def dynamic_fdp_threshold(
    hi_kf: np.ndarray,
    baseline_n: int = STAGE2_FDP_BASELINE_N,
    k_sigma: float = STAGE2_FDP_K_SIGMA,
    min_t: float = STAGE2_FDP_MIN,
    max_t: float = STAGE2_FDP_MAX,
) -> float:
    """케이스 첫 N window 의 HI 통계로 동적 FDP 임계.

    학계 표준 (3σ rule, Wang & Xiang 2021, Nguyen 2025): healthy baseline 의
    noise level + k·std 위로 올라온 시점부터 degradation 진입. 정적 0.15 보다
    fold 별 HI 모델 출력 분포 변동에 robust.
    """
    n = min(baseline_n, len(hi_kf))
    if n < 5:
        return float(min_t)
    base = np.asarray(hi_kf[:n], dtype=np.float64)
    thresh = float(np.mean(base) + k_sigma * np.std(base))
    return float(np.clip(thresh, min_t, max_t))


# ---------------------------------------------------------------------------
# 1.6) Lognormal Conditional Quantile Residual Life (CQRL) (학계 보수적 Prior)
# ---------------------------------------------------------------------------
def lognormal_cqrl(t_now: float, mu: float, sigma: float, quantile: float = STAGE2_LIFETIME_PRIOR_QUANTILE) -> float:
    """t_q - t_now | T > t_now for Lognormal(μ, σ).

    Extreme Lifetime Outlier를 다루기 위해 평균(Mean) 대신 보수적인 분위수(Quantile)를 
    사용하여 비대칭 패널티(Over-estimation) 위험을 회피(Risk-averse)합니다.

    수식: F(t_q) = F(t_now) + q * (1 - F(t_now))
          t_q = exp(μ + σ * Φ⁻¹(F(t_q)))
          CQRL(t_now) = t_q - t_now
    """
    from scipy.stats import norm
    t_now = max(float(t_now), 1.0)
    log_t = np.log(t_now)
    
    # F(t_now) = P(T <= t_now)
    f_t_now = norm.cdf((log_t - mu) / sigma)
    
    # Target CDF probability for the quantile condition
    f_t_q = f_t_now + quantile * (1.0 - f_t_now)
    
    if f_t_q >= 1.0 - 1e-9:
        return float(STAGE2_FALLBACK_FLOOR)
        
    # Inverse CDF (Percent Point Function) to find t_q
    t_q = np.exp(mu + sigma * norm.ppf(f_t_q))
    
    cqrl = t_q - t_now
    return float(max(cqrl, STAGE2_FALLBACK_FLOOR))


def _fallback_rul(t_now: float, lifetime_prior: dict | None) -> float:
    """Track 1 의 FDP 미통과 fallback. lifetime_prior 있으면 CQRL, 없으면 legacy 60000."""
    if lifetime_prior is not None:
        mu = float(lifetime_prior["mu"])
        sigma = float(lifetime_prior["sigma"])
        return lognormal_cqrl(t_now, mu, sigma, STAGE2_LIFETIME_PRIOR_QUANTILE)
    # Legacy: static cap (인위적, 사용자 지적)
    return 60_000.0


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
    lifetime_prior: dict | None = None,
) -> float:
    """현재까지의 HI 시퀀스 → 단일 RUL 점 추정. inference 진입점.

    lifetime_prior: {"mu": μ, "sigma": σ} of training fold lifetime lognormal fit.
                    None 이면 legacy 60000s static cap 사용.
    """
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
    fdp_thresh = dynamic_fdp_threshold(hi_f)

    if hi_f[-1] <= fdp_thresh:
        # Track 1: FDP 미통과. Global Linear Fit + lifetime_prior MRL fallback
        cap = _fallback_rul(t_now, lifetime_prior)
        t_mean = np.mean(t)
        hi_mean = np.mean(hi_kf)
        t_centered = t - t_mean
        hi_centered = hi_kf - hi_mean

        sum_t2_centered = np.sum(t_centered ** 2)
        if sum_t2_centered < 1e-6:
            return cap

        slope = np.sum(t_centered * hi_centered) / sum_t2_centered
        if slope < 1e-8:
            rul = cap
        else:
            intercept = hi_mean - slope * t_mean
            t_fail = (HI_FAILURE_THRESHOLD - intercept) / slope
            rul = max(t_fail - t_now, 0.0)
            rul = min(rul, cap)   # lifetime-prior MRL 로 cap

        return float(rul)

    rul = _fit_log_linear_rul(t, hi_kf, t_now, failure_threshold, fallback_rul_cap)
    return rul


def compute_stage2_trajectory(
    times: np.ndarray,
    hi_preds: np.ndarray,
    failure_threshold: float = HI_FAILURE_THRESHOLD,
    fallback_rul_cap: float = _FALLBACK_RUL_CAP,
    lifetime_prior: dict | None = None,
) -> np.ndarray:
    """전체 시퀀스 → online RUL trajectory.  각 timestep i 에서 [0..i] 만 사용 (causal).

    lifetime_prior: dict(mu, sigma) of training-fold lognormal lifetime fit.
                    None 이면 legacy 60000 static cap.
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

    # Dynamic FDP threshold — case 의 첫 N window 통계로 결정 (한 번 계산, 모든 step 공통)
    fdp_thresh = dynamic_fdp_threshold(hi_f)

    rul_s = np.full(N, fallback_rul_cap, dtype=np.float64)
    last_valid_rul = float(_RUL_BEFORE_FDP)
    last_valid_time = 0.0

    for i in range(N):
        t_now = float(t_s[i])

        if hi_f[i] <= fdp_thresh:
            # Track 1: linear fit + lifetime_prior MRL cap
            cap = _fallback_rul(t_now, lifetime_prior)
            t_healthy = t_s[: i + 1]
            hi_healthy = hi_kf[: i + 1]

            t_mean = np.mean(t_healthy)
            hi_mean = np.mean(hi_healthy)
            t_centered = t_healthy - t_mean
            hi_centered = hi_healthy - hi_mean

            sum_t2_centered = np.sum(t_centered ** 2)
            if sum_t2_centered < 1e-6:
                rul = cap
            else:
                slope = np.sum(t_centered * hi_centered) / sum_t2_centered
                if slope < 1e-8:
                    rul = cap
                else:
                    intercept = hi_mean - slope * t_mean
                    t_fail = (HI_FAILURE_THRESHOLD - intercept) / slope
                    rul = max(t_fail - t_now, 0.0)
                    rul = min(rul, cap)

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
