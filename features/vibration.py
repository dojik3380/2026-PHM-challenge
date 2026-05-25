"""STFT 진동 특징 생성 (Physics-based, operation 제거).

Operation 관련 feature 추출을 완전 제거하고
STFT magnitude vector만 추출하는 vibration-only 파이프라인.
"""

import sys
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
from scipy.signal import find_peaks, stft

# 직접 실행하거나 패키지 외부에서 임포트할 때 프로젝트 루트를 sys.path에 추가
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config import (
    BEARING_BPFI_MULT,
    BEARING_BPFO_MULT,
    BEARING_BSF_MULT,
    BEARING_FTF_MULT,
    BEARING_RPM_DEFAULT,
    BEARING_RPM_MAX,
    BEARING_RPM_MIN,
    SAMPLING_RATE,
    STFT_FREQ_BINS,
    STFT_NOVERLAP,
    STFT_NPERSEG,
)


def _finite_array(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def vibration_statistics(signal: Iterable[float]) -> Dict[str, float]:
    """필요 시 사용할 수 있는 시간영역 통계 특징."""
    from scipy.stats import kurtosis, skew

    arr = _finite_array(signal)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "rms": 0.0, "kurtosis": 0.0, "skewness": 0.0}

    std = float(np.std(arr))
    return {
        "mean": float(np.mean(arr)),
        "std": std,
        "rms": float(np.sqrt(np.mean(arr ** 2))),
        "kurtosis": float(kurtosis(arr, fisher=False, bias=False)) if arr.size > 3 and std > 0 else 0.0,
        "skewness": float(skew(arr, bias=False)) if arr.size > 2 and std > 0 else 0.0,
    }


def _pad_or_trim(vec: np.ndarray, target: int) -> np.ndarray:
    if vec.size < target:
        return np.pad(vec, (0, target - vec.size))
    return vec[:target]


def stft_magnitude_vector(signal: Iterable[float]) -> np.ndarray:
    """
    채널 1개의 raw 진동 신호를 STFT 특징 벡터로 변환한다.

    출력: concat(mean(|Zxx|, axis=time), std(|Zxx|, axis=time))
    shape: (STFT_FREQ_BINS * 2,) = (1026,)

    mean: 평균 스펙트럼 — 어느 주파수가 활성화됐는지
    std : 시간적 변동성 — 임펄스성 결함 주파수 검출에 핵심
    """
    arr = _finite_array(signal)
    if arr.size == 0:
        return np.zeros(STFT_FREQ_BINS * 2, dtype=np.float32)

    if arr.size < STFT_NPERSEG:
        arr = np.pad(arr, (0, STFT_NPERSEG - arr.size))

    _, _, zxx = stft(
        arr,
        fs=SAMPLING_RATE,
        nperseg=STFT_NPERSEG,
        noverlap=STFT_NOVERLAP,
        boundary=None,
        padded=False,
    )
    magnitude = np.abs(zxx)
    freq_mean = _pad_or_trim(np.mean(magnitude, axis=1), STFT_FREQ_BINS)
    freq_std  = _pad_or_trim(np.std(magnitude,  axis=1), STFT_FREQ_BINS)

    return np.concatenate([freq_mean, freq_std]).astype(np.float32)


def _full_rfft(signal: np.ndarray, sampling_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Full-length one-sided FFT → (freqs, amplitude). Removes DC, no window needed for peak detection."""
    arr = np.asarray(signal, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size < 4:
        return np.array([0.0]), np.array([0.0])
    arr = arr - arr.mean()
    spectrum = np.abs(np.fft.rfft(arr))
    freqs = np.fft.rfftfreq(arr.size, d=1.0 / sampling_rate)
    return freqs, spectrum


def _spectral_peak_in_band(
    freqs: np.ndarray,
    spectrum: np.ndarray,
    target_hz: float,
    bandwidth_hz: float = 2.0,
) -> float:
    """Max spectral amplitude in [target - bw/2, target + bw/2], normalized by N."""
    if target_hz <= 0 or target_hz > freqs[-1]:
        return 0.0
    lo = target_hz - bandwidth_hz / 2.0
    hi = target_hz + bandwidth_hz / 2.0
    mask = (freqs >= lo) & (freqs <= hi)
    if not np.any(mask):
        idx = int(np.argmin(np.abs(freqs - target_hz)))
        return float(spectrum[idx])
    return float(np.max(spectrum[mask]))


def estimate_shaft_freq_hz(
    signal: np.ndarray,
    sampling_rate: int = SAMPLING_RATE,
    min_rpm: float = BEARING_RPM_MIN,
    max_rpm: float = BEARING_RPM_MAX,
    default_rpm: float = BEARING_RPM_DEFAULT,
) -> float:
    """Estimate shaft (1×) frequency in Hz from vibration spectrum using harmonic peak scoring.

    Mirrors estimationRPM.estimate_rpm_from_spectrum but operates on a single
    pre-read signal array and returns Hz (not RPM). Falls back to default_rpm
    when the signal is too short or no plausible peak is found.
    """
    freqs, spectrum = _full_rfft(signal, sampling_rate)
    if freqs.size < 2:
        return default_rpm / 60.0

    min_hz, max_hz = min_rpm / 60.0, max_rpm / 60.0
    mask = (freqs >= min_hz) & (freqs <= max_hz)
    candidate_freqs = freqs[mask]
    search_spectrum = spectrum[mask]

    if candidate_freqs.size == 0:
        return default_rpm / 60.0

    peak_indices, _ = find_peaks(search_spectrum)
    if peak_indices.size == 0:
        peak_indices = np.arange(candidate_freqs.size)

    best_freq = float(candidate_freqs[peak_indices[0]])
    best_score = -np.inf
    for pi in peak_indices:
        f = float(candidate_freqs[pi])
        score = (
            _spectral_peak_in_band(freqs, spectrum, f)
            + 0.5 * _spectral_peak_in_band(freqs, spectrum, 2.0 * f)
            + 0.25 * _spectral_peak_in_band(freqs, spectrum, 3.0 * f)
        )
        if score > best_score:
            best_score = score
            best_freq = f

    return best_freq


def bearing_fault_amplitudes(
    signal: np.ndarray,
    sampling_rate: int = SAMPLING_RATE,
) -> np.ndarray:
    """Estimate shaft RPM from signal FFT, compute bearing fault frequencies, return spectral amplitudes.

    30306 bearing multipliers: BPFI=8.40×, BPFO=5.58×, BSF=4.68×, FTF=0.402× shaft freq.

    Returns 6-dim float32: [BPFI_1X, BPFI_2X, BPFO_1X, BPFO_2X, BSF_1X, FTF_1X]
    Amplitudes are divided by signal length so different chunk sizes give comparable values.
    """
    arr = np.asarray(signal, dtype=np.float64)
    arr = arr[np.isfinite(arr)]

    # Need at least 1 second to resolve shaft frequency (min ~10 Hz at 600 RPM)
    if arr.size < sampling_rate:
        shaft_hz = BEARING_RPM_DEFAULT / 60.0
    else:
        shaft_hz = estimate_shaft_freq_hz(arr, sampling_rate)

    freqs, spectrum = _full_rfft(arr, sampling_rate)
    n = max(arr.size, 1)
    spectrum = spectrum / n  # normalize by signal length

    bpfi = BEARING_BPFI_MULT * shaft_hz
    bpfo = BEARING_BPFO_MULT * shaft_hz
    bsf  = BEARING_BSF_MULT  * shaft_hz
    ftf  = BEARING_FTF_MULT  * shaft_hz

    def amp(f_hz: float) -> float:
        return _spectral_peak_in_band(freqs, spectrum, f_hz)

    return np.array([
        amp(bpfi), amp(2.0 * bpfi),
        amp(bpfo), amp(2.0 * bpfo),
        amp(bsf),
        amp(ftf),
    ], dtype=np.float32)


def augment_stft_features(stft_matrix: np.ndarray, aug_prob: float = 0.3) -> np.ndarray:
    """STFT 특징에 노이즈 억제 중심 증강 적용 (대회 특성 고려)"""
    augmented = stft_matrix.copy()
    
    # 1. 가벼운 가우시안 노이즈 (대회 노이즈 시뮬레이션)
    if np.random.random() < aug_prob * 0.5:  # 확률 낮춤
        noise_std = 0.02  # 기존 0.05 → 0.02로 감소
        noise = np.random.normal(0, noise_std, augmented.shape)
        augmented += noise
    
    # 2. 밝기 조정만 (콘트라스트 제외 - 노이즈 증가 방지)
    if np.random.random() < aug_prob * 0.7:
        brightness = np.random.uniform(0.95, 1.05)  # 범위 좁힘
        augmented *= brightness
    
    # 3. SpecAugment-style 마스킹 (노이즈 패턴 학습용)
    if np.random.random() < aug_prob * 0.4:
        freq_bins, time_steps = augmented.shape
        
        # 빈도 마스킹 (좁은 범위)
        if freq_bins > 1:
            max_freq_width = max(1, freq_bins // 8)
            mask_freq_width = np.random.randint(1, min(freq_bins, max_freq_width) + 1)
            mask_freq_start = np.random.randint(0, freq_bins - mask_freq_width + 1)
            augmented[mask_freq_start:mask_freq_start + mask_freq_width, :] *= 0.1  # 완전 마스킹 X
        
        # 시간 마스킹 (좁은 범위)
        if time_steps > 1:
            max_time_width = max(1, time_steps // 8)
            mask_time_width = np.random.randint(1, min(time_steps, max_time_width) + 1)
            mask_time_start = np.random.randint(0, time_steps - mask_time_width + 1)
            augmented[:, mask_time_start:mask_time_start + mask_time_width] *= 0.1
    
    return np.maximum(augmented, 0)  # 음수 방지


def augment_sequence_level(X_vib: np.ndarray, X_aux: np.ndarray, y: float, aug_prob: float = 0.3) -> list:
    """시퀀스 레벨 증강.

    RUL은 시간적 인과관계가 있으므로 시퀀스 반전(flip) 금지.
    적용 증강:
      1. 유색 노이즈 추가 (SNR 변동 모사)
      2. 진폭 스케일링 (센서 감도 변동 모사)
      3. 랜덤 시간 이동 (window 위치 jitter)
    """
    augmented = [(X_vib, X_aux, y)]

    # 1. 유색 노이즈 (AR(1) colored noise — 실제 기계 배경 노이즈 모사)
    if np.random.random() < aug_prob:
        alpha = 0.3
        noise_std = np.random.uniform(0.005, 0.02)
        white = np.random.randn(*X_vib.shape).astype(np.float32) * noise_std
        # 시간축(axis=0)을 따라 IIR 필터 적용 (간이 AR)
        colored = white.copy()
        for t in range(1, X_vib.shape[0]):
            colored[t] = alpha * white[t] + (1 - alpha) * colored[t - 1]
        augmented.append((X_vib + colored, X_aux, y))

    # 2. 진폭 스케일링 (Uniform(0.85, 1.15))
    if np.random.random() < aug_prob:
        scale = np.random.uniform(0.85, 1.15)
        augmented.append((X_vib * scale, X_aux, y))

    # 3. 랜덤 시간 이동 (shift) — 경계를 edge로 채워 causality 보존
    if np.random.random() < aug_prob * 0.5:
        shift = np.random.randint(-4, 5)
        if shift != 0:
            vib_shifted = np.roll(X_vib, shift, axis=0)
            aux_shifted = np.roll(X_aux, shift, axis=0)
            if shift > 0:
                vib_shifted[:shift]  = X_vib[0:1]
                aux_shifted[:shift]  = X_aux[0:1]
            else:
                vib_shifted[shift:]  = X_vib[-1:]
                aux_shifted[shift:]  = X_aux[-1:]
            augmented.append((vib_shifted, aux_shifted, y))

    return augmented
