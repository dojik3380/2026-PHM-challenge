"""RPM trajectory estimation 및 auxiliary feature 추출 모듈.

TDMS raw vibration 신호에서 shaft rotational frequency를 추정하고,
RPM(t) trajectory 및 RMS feature를 생성한다.

RPM 추정 알고리즘:
    1. 4채널 raw signal 로드
    2. 채널별로 Welch PSD 계산
    3. RPM_SEARCH_RANGE(5~30 Hz) 대역에서 dominant peak 탐색
    4. 4채널 후보 중 median 선택 (이상치 제거)
    5. RPM = median_shaft_freq * 60
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import numpy as np
from scipy.signal import welch

# 직접 실행하거나 패키지 외부에서 임포트할 때 프로젝트 루트를 sys.path에 추가
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config import (
    RPM_ESTIMATION_WINDOW,
    RPM_SEARCH_RANGE,
    SAMPLING_RATE,
    VIBRATION_CHANNELS,
)


def estimate_rpm_from_signal(
    signal: np.ndarray,
    fs: int = SAMPLING_RATE,
    search_range: tuple[float, float] = RPM_SEARCH_RANGE,
    nperseg: int = RPM_ESTIMATION_WINDOW,
) -> float:
    """단일 채널 진동 신호에서 shaft rotational frequency를 추정한다.

    Welch PSD를 계산하여 search_range(기본 5~30Hz) 내 dominant peak를
    찾고, 이를 shaft frequency로 사용한다.

    Args:
        signal: 1D raw vibration signal.
        fs: Sampling rate (Hz).
        search_range: (min_freq, max_freq) shaft frequency 탐색 범위.
        nperseg: PSD 계산에 사용할 FFT window 크기.

    Returns:
        RPM 값 (shaft_freq * 60). 추정 실패 시 0.0.
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.size < nperseg:
        # 신호가 너무 짧으면 zero-pad
        if signal.size < 64:
            return 0.0
        nperseg = min(nperseg, signal.size)

    freqs, psd = welch(signal, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)

    # search_range 대역 필터링
    freq_min, freq_max = search_range
    mask = (freqs >= freq_min) & (freqs <= freq_max)
    if not mask.any():
        return 0.0

    band_freqs = freqs[mask]
    band_psd = psd[mask]

    # Dominant peak 탐색
    peak_idx = np.argmax(band_psd)
    shaft_freq = float(band_freqs[peak_idx])

    return shaft_freq * 60.0


def estimate_rpm_trajectory(
    channel_data: Dict[str, np.ndarray],
    fs: int = SAMPLING_RATE,
    search_range: tuple[float, float] = RPM_SEARCH_RANGE,
) -> float:
    """4채널 진동 데이터에서 대표 RPM을 추정한다.

    각 채널별로 RPM을 추정한 뒤 median을 취해 이상치를 제거한다.

    Args:
        channel_data: {channel_name: signal_array} 딕셔너리.
        fs: Sampling rate (Hz).
        search_range: shaft frequency 탐색 범위.

    Returns:
        대표 RPM 값 (float). 모든 채널 추정 실패 시 0.0.
    """
    rpm_candidates = []
    normalized = {name.upper(): values for name, values in channel_data.items()}

    for ch_name in VIBRATION_CHANNELS:
        signal = normalized.get(ch_name.upper())
        if signal is None or len(signal) == 0:
            continue
        rpm = estimate_rpm_from_signal(
            np.asarray(signal, dtype=np.float64), fs=fs, search_range=search_range
        )
        if rpm > 0:
            rpm_candidates.append(rpm)

    if not rpm_candidates:
        return 0.0

    return float(np.median(rpm_candidates))


def extract_rms_from_channels(
    channel_data: Dict[str, np.ndarray],
) -> float:
    """4채널 진동 데이터에서 평균 RMS를 계산한다.

    RMS = sqrt(mean(x^2))를 각 채널별로 계산 후 평균한다.

    Args:
        channel_data: {channel_name: signal_array} 딕셔너리.

    Returns:
        4채널 평균 RMS 값 (float).
    """
    rms_values = []
    normalized = {name.upper(): values for name, values in channel_data.items()}

    for ch_name in VIBRATION_CHANNELS:
        signal = normalized.get(ch_name.upper())
        if signal is None or len(signal) == 0:
            continue
        arr = np.asarray(signal, dtype=np.float64)
        rms = float(np.sqrt(np.mean(arr ** 2)))
        if np.isfinite(rms):
            rms_values.append(rms)

    if not rms_values:
        return 0.0

    return float(np.mean(rms_values))


def extract_auxiliary_vector(
    channel_data: Dict[str, np.ndarray],
    fs: int = SAMPLING_RATE,
) -> np.ndarray:
    """TDMS 채널 데이터에서 auxiliary feature vector [RPM, RMS]를 추출한다.

    Args:
        channel_data: {channel_name: signal_array} 딕셔너리.
        fs: Sampling rate (Hz).

    Returns:
        np.ndarray shape (2,) = [RPM, RMS], dtype float32.
    """
    rpm = estimate_rpm_trajectory(channel_data, fs=fs)
    rms = extract_rms_from_channels(channel_data)

    return np.array([rpm, rms], dtype=np.float32)
