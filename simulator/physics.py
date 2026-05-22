"""
simulator/physics.py

물리 기반 베어링 결함 임펄스 생성 엔진.
수식 출처: KIMM PHM Challenge 슬라이드 및 관련 논문.

핵심 수식:
  단일 임펄스 응답 h(t):
    t < 0: exp(-zeta_L / sqrt(1 - zeta_L^2) * (2*pi*fn*t)^2) * cos(2*pi*fn*t)
    t >= 0: exp(-zeta_R / sqrt(1 - zeta_R^2) * (2*pi*fn*t)^2) * cos(2*pi*fn*t)

  임펄스 트레인 y[i]:
    y[i] = sum_k h(i*Delta_t - k*T - Delta*T)
    Delta: [-0.1, 0.1] 범위의 랜덤 슬립 (±10% 타이밍 지터)
"""
from __future__ import annotations

import numpy as np
from numpy.random import Generator


# ============================================================
# 1. 베어링 물리 규격 (30306 Bearing)
# ============================================================
BEARING_SPECS = {
    "model": "30306",
    "sampling_rate": 25_600,
    # 1000 RPM 기준 결함 주파수 (Hz)
    "fault_frequencies": {
        "BPFI": 140.0,
        "BPFO": 93.0,
        "BSF":  78.0,
        "Cage": 6.7,
    },
    # 실험 데이터에서 추정된 f_n, zeta 통계 (사진 슬라이드 기반)
    "fn_mean": 2404.0,
    "fn_std":  137.45,
    "fn_min":  2053.46,
    "fn_max":  2622.31,
    "zeta_mean": 0.007,
    "zeta_std":  0.005,
    "zeta_min":  0.0003,
    "zeta_max":  0.0259,
}


# ============================================================
# 2. 단일 임펄스 응답 h(t) 생성
# ============================================================

def single_impulse_response(
    fn: float,
    zeta_L: float,
    zeta_R: float,
    fs: int = 25_600,
    duration_ms: float = 5.0,
) -> np.ndarray:
    """
    비대칭 감쇠 가우시안 포락선(Asymmetric Gaussian Envelope) 모델로
    단일 결함 임펄스 응답 h(t)를 생성한다.

    Args:
        fn: 고유 진동수 (Natural frequency, Hz)
        zeta_L: 좌측(t<0) 감쇠비 (Damping ratio)
        zeta_R: 우측(t>=0) 감쇠비 (Damping ratio)
        fs: 샘플링 레이트 (Hz)
        duration_ms: 임펄스 창 길이 (ms, 좌우 대칭)

    Returns:
        h: 단일 임펄스 응답 배열 (shape: [N_impulse_samples])
    """
    n_samples = int(fs * duration_ms / 1000.0)
    t = np.linspace(-duration_ms / 2000.0, duration_ms / 2000.0, 2 * n_samples + 1)

    # 분모 안전 처리 (감쇠비가 1에 가까우면 sqrt가 0에 수렴)
    sqrt_L = np.sqrt(max(1.0 - zeta_L ** 2, 1e-8))
    sqrt_R = np.sqrt(max(1.0 - zeta_R ** 2, 1e-8))

    envelope = np.where(
        t < 0,
        np.exp(-zeta_L / sqrt_L * (2 * np.pi * fn * t) ** 2),
        np.exp(-zeta_R / sqrt_R * (2 * np.pi * fn * t) ** 2),
    )
    h = envelope * np.cos(2 * np.pi * fn * t)
    return h.astype(np.float32)


# ============================================================
# 3. 결함 주파수 기반 임펄스 트레인 생성 y[i]
# ============================================================

def build_impulse_train(
    signal_length: int,
    fault_freq: float,
    fn: float,
    zeta_L: float,
    zeta_R: float,
    amplitude: float,
    slip_range: float = 0.10,
    fs: int = 25_600,
    rng: Generator | None = None,
) -> np.ndarray:
    """
    주기 T = 1 / fault_freq 마다 임펄스를 삽입하여 신호 y[i]를 만든다.
    각 임펄스 타이밍에는 ±slip_range*100% 의 랜덤 지터를 추가한다.

    수식: y[i] = sum_k h(i*Delta_t - k*T - Delta*T)
          Delta ~ Uniform(-slip_range, +slip_range)

    Args:
        signal_length: 출력 신호 길이 (샘플 수)
        fault_freq: 결함 주파수 (Hz)
        fn, zeta_L, zeta_R: 임펄스 응답 파라미터
        amplitude: 임펄스 진폭 (열화도에 따라 외부에서 조절)
        slip_range: 타이밍 지터 비율 (0.10 = ±10%)
        fs: 샘플링 레이트
        rng: NumPy 랜덤 제너레이터

    Returns:
        y: 신호 배열 (shape: [signal_length])
    """
    if rng is None:
        rng = np.random.default_rng()

    y = np.zeros(signal_length, dtype=np.float32)

    # 단일 임펄스 응답 커널
    h = single_impulse_response(fn, zeta_L, zeta_R, fs=fs)
    h_len = len(h)
    h_center = h_len // 2  # 커널의 t=0 위치

    # 결함 주기 (샘플 단위)
    T_samples = fs / fault_freq

    # 임펄스 발생 위치 결정
    k = 0
    while True:
        # 슬립(Timing Jitter) 적용: Delta ~ Uniform(-slip_range, +slip_range)
        delta = rng.uniform(-slip_range, slip_range)
        impulse_center = int(round(k * T_samples * (1.0 + delta)))

        if impulse_center - h_center >= signal_length:
            break

        # 커널을 신호에 오버랩-추가(Overlap-Add)
        start_sig = impulse_center - h_center
        end_sig = start_sig + h_len
        start_ker = 0
        end_ker = h_len

        # 신호 경계 클리핑
        if end_sig <= 0 or start_sig >= signal_length:
            k += 1
            continue
        if start_sig < 0:
            start_ker -= start_sig
            start_sig = 0
        if end_sig > signal_length:
            end_ker -= (end_sig - signal_length)
            end_sig = signal_length

        y[start_sig:end_sig] += amplitude * h[start_ker:end_ker]
        k += 1

    return y


# ============================================================
# 4. 배경 진동 및 노이즈 합성
# ============================================================

def make_background_vibration(
    signal_length: int,
    rpm: float,
    noise_std: float,
    fs: int = 25_600,
    rng: Generator | None = None,
) -> np.ndarray:
    """
    축 회전 주파수(1X RPM)와 그 하모닉(2X, 3X) + 가우시안 백색 노이즈를 합성한다.

    Args:
        signal_length: 출력 신호 길이 (샘플 수)
        rpm: 회전 속도 (RPM)
        noise_std: 가우시안 노이즈 표준편차
        fs: 샘플링 레이트
        rng: NumPy 랜덤 제너레이터

    Returns:
        bg: 배경 진동 신호 배열 (shape: [signal_length])
    """
    if rng is None:
        rng = np.random.default_rng()

    t = np.arange(signal_length) / fs
    rotation_freq = rpm / 60.0

    bg = np.zeros(signal_length, dtype=np.float32)
    # 1X, 2X, 3X 하모닉 (진폭은 차수에 반비례)
    for harmonic in [1, 2, 3]:
        amp = rng.uniform(0.005, 0.02) / harmonic
        phase = rng.uniform(0, 2 * np.pi)
        bg += amp * np.sin(2 * np.pi * harmonic * rotation_freq * t + phase).astype(np.float32)

    # 유색 노이즈(Colored Noise): 낮은 주파수를 강조하여 실제 기계 진동 모사
    from scipy.signal import lfilter
    alpha = 0.3
    white = rng.standard_normal(signal_length).astype(np.float32) * noise_std
    # y[i] = alpha*x[i] + (1-alpha)*y[i-1]  →  IIR: b=[alpha], a=[1, -(1-alpha)]
    colored = lfilter([alpha], [1.0, -(1.0 - alpha)], white).astype(np.float32)

    bg += colored
    return bg


# ============================================================
# 5. 파라미터 랜덤 샘플링 헬퍼
# ============================================================

def sample_fn_zeta(
    rng: Generator,
    specs: dict = BEARING_SPECS,
) -> tuple[float, float, float]:
    """
    실험 데이터 통계(mean, std, min, max)를 바탕으로
    자연 진동수 fn, 좌/우 감쇠비 zeta_L, zeta_R를 샘플링한다.

    Returns:
        fn, zeta_L, zeta_R
    """
    fn = float(np.clip(
        rng.normal(specs["fn_mean"], specs["fn_std"]),
        specs["fn_min"], specs["fn_max"]
    ))
    zeta = float(np.clip(
        rng.normal(specs["zeta_mean"], specs["zeta_std"]),
        specs["zeta_min"], specs["zeta_max"]
    ))
    # 좌/우 감쇠비를 약간 다르게 설정 (비대칭 포락선)
    zeta_L = float(np.clip(zeta * rng.uniform(0.8, 1.0), specs["zeta_min"], specs["zeta_max"]))
    zeta_R = float(np.clip(zeta * rng.uniform(1.0, 1.3), specs["zeta_min"], specs["zeta_max"]))
    return fn, zeta_L, zeta_R


def get_fault_freq_for_rpm(fault_type: str, rpm: float) -> float:
    """
    기준 RPM(1000)에서의 결함 주파수를 실제 RPM에 맞게 선형 스케일링하여 반환.
    """
    base_rpm = 1000.0
    base_freqs = BEARING_SPECS["fault_frequencies"]
    if fault_type not in base_freqs:
        raise ValueError(f"Unknown fault type: {fault_type}. Choose from {list(base_freqs.keys())}")
    return base_freqs[fault_type] * (rpm / base_rpm)

def get_all_fault_freqs_for_rpm(rpm: float) -> dict[str, float]:
    """
    기준 RPM(1000)에서의 결함 주파수를 실제 RPM에 맞게 스케일링하여 모두 반환.
    """
    base_rpm = 1000.0
    base_freqs = BEARING_SPECS["fault_frequencies"]
    return {k: v * (rpm / base_rpm) for k, v in base_freqs.items()}
