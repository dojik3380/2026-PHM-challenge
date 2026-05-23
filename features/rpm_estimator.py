"""RPM trajectory estimation 및 auxiliary feature 추출 모듈.

TDMS raw vibration 신호에서 shaft rotational frequency를 추정한다.

RPM 추정 알고리즘 (Harmonic Scoring):
    1. 4채널 스펙트럼을 평균 → SNR 향상
    2. 600~1100 RPM (10~18.3 Hz) 구간에서 candidate peak 탐색
    3. 각 peak에 대해 1X + 0.5×2X + 0.25×3X harmonic score 계산
    4. 가장 높은 score의 peak → 실제 shaft frequency
    5. RPM = shaft_freq × 60
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from scipy.signal import detrend, find_peaks, windows

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config import SAMPLING_RATE, VIBRATION_CHANNELS

# 운전 RPM 범위 (600~1100은 실제 700~980에 충분한 마진)
RPM_MIN = 600.0
RPM_MAX = 1_100.0


def _channel_spectrum(
    signal: np.ndarray,
    fs: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """단일 채널 신호 → (freqs, magnitude spectrum)."""
    sig = np.asarray(signal, dtype=np.float64)
    sig = sig[np.isfinite(sig)]
    if sig.size < 4:
        return np.zeros(1), np.zeros(1)

    sig = detrend(sig, type="constant")
    win = windows.hann(sig.size, sym=False)
    spectrum = np.abs(np.fft.rfft(sig * win))
    freqs = np.fft.rfftfreq(sig.size, d=1.0 / fs)
    return freqs, spectrum


def _amplitude_at(freqs: np.ndarray, spectrum: np.ndarray, target_hz: float) -> float:
    if target_hz > freqs[-1] or freqs.size < 2:
        return 0.0
    idx = int(np.argmin(np.abs(freqs - target_hz)))
    return float(spectrum[idx])


def _combined_spectrum(
    channel_data: Dict[str, np.ndarray],
    fs: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """4채널 스펙트럼을 평균해 반환한다."""
    spectra = []
    freqs_ref = None
    normalized = {k.upper(): v for k, v in channel_data.items()}

    for ch in VIBRATION_CHANNELS:
        sig = normalized.get(ch)
        if sig is None or len(sig) == 0:
            continue
        f, s = _channel_spectrum(sig, fs)
        if freqs_ref is None:
            freqs_ref = f
        if s.size == freqs_ref.size:
            spectra.append(s)

    if not spectra:
        return np.zeros(1), np.zeros(1)

    return freqs_ref, np.mean(np.stack(spectra, axis=0), axis=0)


def estimate_rpm_trajectory(
    channel_data: Dict[str, np.ndarray],
    fs: int = SAMPLING_RATE,
) -> float:
    """4채널 진동 데이터에서 harmonic scoring으로 shaft RPM을 추정한다.

    Returns
    -------
    추정 RPM (float). 실패 시 0.0.
    """
    freqs, spectrum = _combined_spectrum(channel_data, fs)
    if freqs.size < 2:
        return 0.0

    min_hz = RPM_MIN / 60.0
    max_hz = RPM_MAX / 60.0
    mask = (freqs >= min_hz) & (freqs <= max_hz)
    if not mask.any():
        return 0.0

    cand_freqs = freqs[mask]
    cand_spec  = spectrum[mask]

    peak_indices, _ = find_peaks(cand_spec)
    if peak_indices.size == 0:
        peak_indices = np.arange(cand_freqs.size)

    best_freq  = float(cand_freqs[0])
    best_score = -np.inf

    for pi in peak_indices:
        f = float(cand_freqs[pi])
        score = (
            _amplitude_at(freqs, spectrum, f)
            + 0.5  * _amplitude_at(freqs, spectrum, 2.0 * f)
            + 0.25 * _amplitude_at(freqs, spectrum, 3.0 * f)
        )
        if score > best_score:
            best_score = score
            best_freq  = f

    return best_freq * 60.0


# 하위 호환 — 단일 채널 버전 (data_loader 등에서 직접 호출하는 경우)
def estimate_rpm_from_signal(
    signal: np.ndarray,
    fs: int = SAMPLING_RATE,
    search_range: tuple = (RPM_MIN / 60.0, RPM_MAX / 60.0),
    nperseg: int = 4096,
) -> float:
    """단일 채널 신호에서 harmonic scoring으로 RPM 추정."""
    ch_data = {"CH1": signal}
    return estimate_rpm_trajectory(ch_data, fs=fs)


def extract_rms_from_channels(
    channel_data: Dict[str, np.ndarray],
) -> float:
    """4채널 평균 RMS를 반환한다."""
    rms_values = []
    normalized = {k.upper(): v for k, v in channel_data.items()}
    for ch in VIBRATION_CHANNELS:
        sig = normalized.get(ch)
        if sig is None or len(sig) == 0:
            continue
        arr = np.asarray(sig, dtype=np.float64)
        rms = float(np.sqrt(np.mean(arr ** 2)))
        if np.isfinite(rms):
            rms_values.append(rms)
    return float(np.mean(rms_values)) if rms_values else 0.0


def extract_auxiliary_vector(
    channel_data: Dict[str, np.ndarray],
    fs: int = SAMPLING_RATE,
) -> np.ndarray:
    """[RPM, RMS] auxiliary feature vector (shape: (2,), float32)."""
    rpm = estimate_rpm_trajectory(channel_data, fs=fs)
    rms = extract_rms_from_channels(channel_data)
    return np.array([rpm, rms], dtype=np.float32)
