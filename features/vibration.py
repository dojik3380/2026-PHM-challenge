"""STFT 진동 특징 생성 (Physics-based, operation 제거).

Operation 관련 feature 추출을 완전 제거하고
STFT magnitude vector만 추출하는 vibration-only 파이프라인.
"""

import sys
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
from scipy.signal import stft

# 직접 실행하거나 패키지 외부에서 임포트할 때 프로젝트 루트를 sys.path에 추가
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config import (
    SAMPLING_RATE,
    STFT_CACHE_DIR,
    STFT_CACHE_ENABLED,
    STFT_FREQ_BINS,
    STFT_NOVERLAP,
    STFT_NPERSEG,
    VIBRATION_CHANNELS,
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


def stft_magnitude_vector(signal: Iterable[float]) -> np.ndarray:
    """
    채널 1개의 raw 진동 신호를 STFT magnitude 벡터로 변환한다.
    STFT time 축은 평균내서 freq_bins 길이의 벡터로 만든다.
    """
    arr = _finite_array(signal)
    if arr.size == 0:
        return np.zeros(STFT_FREQ_BINS, dtype=np.float32)

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
    freq_vector = np.mean(magnitude, axis=1)

    if freq_vector.size < STFT_FREQ_BINS:
        freq_vector = np.pad(freq_vector, (0, STFT_FREQ_BINS - freq_vector.size))
    elif freq_vector.size > STFT_FREQ_BINS:
        freq_vector = freq_vector[:STFT_FREQ_BINS]

    return freq_vector.astype(np.float32)


def vibration_stft_timestep(
    tdms_file_path: str | Path,
) -> np.ndarray:
    """
    TDMS 파일 1개를 STFT timestep으로 변환 (캐시 지원).
    출력 shape: (4, STFT_FREQ_BINS) = (4, 513)

    Handcrafted features 제거 — STFT magnitude만 사용.
    """
    import hashlib
    import pickle
    from pathlib import Path

    tdms_path = Path(tdms_file_path)
    
    # 캐시 키 생성 (파일 경로 + 수정시간 + 'v2' 접미사로 기존 캐시와 분리)
    file_stat = tdms_path.stat()
    cache_key = hashlib.md5(f"{tdms_path}:{file_stat.st_mtime}:v3_stft_mean".encode()).hexdigest()
    cache_file = STFT_CACHE_DIR / f"{cache_key}.pkl"
    
    # 캐시 히트 시 로드
    if STFT_CACHE_ENABLED and cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception:
            pass  # 캐시 로드 실패 시 재계산
    
    # 캐시 미스 시 계산
    from data_loader import load_tdms_channels
    channel_data = load_tdms_channels(tdms_path)
    
    normalized = {name.upper(): values for name, values in channel_data.items()}
    stft_vectors = []
    
    for channel in VIBRATION_CHANNELS:
        signal = np.array(normalized.get(channel.upper(), []), dtype=np.float32)
        stft_vectors.append(stft_magnitude_vector(signal))
    
    # STFT only: (4, STFT_FREQ_BINS)
    result = np.stack(stft_vectors, axis=0).astype(np.float32)
    
    # 캐시 저장
    if STFT_CACHE_ENABLED:
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
        except Exception:
            pass  # 캐시 저장 실패 시 무시
    
    return result


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
    """시퀀스 레벨 증강 (시간축 조작).
    
    Auxiliary(RPM/RMS)는 물리적 값이므로 vibration과 동일하게
    시간축만 조작한다 (값 자체는 변경하지 않음).
    """
    augmented = [(X_vib, X_aux, y)]  # 원본 유지
    
    # 1. 가벼운 시간 반전 (물리적 의미 유지)
    if np.random.random() < aug_prob * 0.6:
        vib_reversed = np.flip(X_vib, axis=1)
        aux_reversed = np.flip(X_aux, axis=0)
        augmented.append((vib_reversed, aux_reversed, y))
    
    # 2. 가벼운 시간 이동 (shift) 증강
    if np.random.random() < aug_prob * 0.4:
        shift = np.random.randint(-4, 5)
        if shift != 0:
            vib_shifted = np.roll(X_vib, shift, axis=0)
            aux_shifted = np.roll(X_aux, shift, axis=0)
            if shift > 0:
                vib_shifted[:shift] = X_vib[0:1]
                aux_shifted[:shift] = X_aux[0:1]
            else:
                vib_shifted[shift:] = X_vib[-1:]
                aux_shifted[shift:] = X_aux[-1:]
            augmented.append((vib_shifted, aux_shifted, y))
    
    return augmented
