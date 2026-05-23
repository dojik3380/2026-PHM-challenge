"""
simulator/physics.py

물리 기반 베어링 결함 임펄스 생성 엔진.

핵심 수식:
  단일 임펄스 응답 h(t):
    t < 0: exp(-zeta_L / sqrt(1 - zeta_L^2) * (2*pi*fn*t)^2) * cos(2*pi*fn*t)
    t >= 0: exp(-zeta_R / sqrt(1 - zeta_R^2) * (2*pi*fn*t)^2) * cos(2*pi*fn*t)

  임펄스 트레인 y[i]:
    y[i] = sum_k h(i*Delta_t - k*T - Delta*T)
    Delta: [-slip, +slip] 범위의 랜덤 타이밍 지터
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
    # 다중모달 공진 주파수 (실제 TDMS에서 4kHz / 12kHz 대역 prominence 관찰)
    # 형식: (mean_Hz, std_Hz, prob)
    "fn_modes": [
        (4000.0,  400.0, 0.70),   # 저주파 housing resonance
        (12000.0, 700.0, 0.30),   # 고주파 structure resonance
    ],
    # JSON calibration fallback 용 단일 모달 통계 (build_bearing_specs_from_json 호환)
    "fn_mean": 4000.0,
    "fn_std":  500.0,
    "fn_min":  2000.0,
    "fn_max":  15000.0,
    # 감쇠비: Uniform(0.01, 0.04) — 음수 방지, 현실적 decay 길이 확보
    "zeta_min": 0.01,
    "zeta_max": 0.04,
    "zeta_mean": 0.025,   # JSON fallback 용
    "zeta_std":  0.008,
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
    비대칭 감쇠 Gaussian 포락선 모델로 단일 결함 임펄스 응답 h(t)를 생성한다.
    """
    n_samples = int(fs * duration_ms / 1000.0)
    t = np.linspace(-duration_ms / 2000.0, duration_ms / 2000.0, 2 * n_samples + 1)

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
# 3. 결함 주파수 기반 임펄스 트레인 생성
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
    주기 T = 1/fault_freq 마다 임펄스를 삽입하여 신호 y[i]를 만든다.
    타이밍 지터: Delta ~ Uniform(-slip_range, +slip_range)
    """
    if rng is None:
        rng = np.random.default_rng()

    y = np.zeros(signal_length, dtype=np.float32)
    h = single_impulse_response(fn, zeta_L, zeta_R, fs=fs)
    h_len = len(h)
    h_center = h_len // 2
    T_samples = fs / fault_freq

    k = 0
    while True:
        delta = rng.uniform(-slip_range, slip_range)
        impulse_center = int(round(k * T_samples * (1.0 + delta)))

        if impulse_center - h_center >= signal_length:
            break

        start_sig = impulse_center - h_center
        end_sig   = start_sig + h_len
        start_ker = 0
        end_ker   = h_len

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
# 4. AM 사이드밴드 모듈레이션
# ============================================================

def add_sideband_modulation(
    signal: np.ndarray,
    shaft_freq: float,
    fs: int = 25_600,
    rng: Generator | None = None,
) -> np.ndarray:
    """
    실제 베어링 결함 신호에 존재하는 AM 사이드밴드 구조를 추가한다.

    fault_freq ± shaft_freq 사이드밴드 생성.
    변조 심도: Uniform(0.10, 0.40)
    """
    if rng is None:
        rng = np.random.default_rng()

    t = np.arange(len(signal)) / fs
    depth = rng.uniform(0.10, 0.40)
    phase = rng.uniform(0.0, 2 * np.pi)
    modulator = 1.0 + depth * np.sin(2 * np.pi * shaft_freq * t + phase)
    return (signal * modulator.astype(np.float32))


# ============================================================
# 5. 배경 진동 및 노이즈 합성 (하모닉 랜덤화)
# ============================================================

def make_background_vibration(
    signal_length: int,
    rpm: float,
    noise_std: float,
    fs: int = 25_600,
    rng: Generator | None = None,
) -> np.ndarray:
    """
    축 회전 주파수(1X RPM)와 랜덤 차수 하모닉(1~5X) + 유색 노이즈를 합성한다.

    하모닉 리얼리즘:
    - 가시 차수: 1~5 랜덤
    - 각 차수 진폭: (base_amp / k) × Uniform(0.7, 1.3)
    """
    if rng is None:
        rng = np.random.default_rng()

    t = np.arange(signal_length) / fs
    rotation_freq = rpm / 60.0

    bg = np.zeros(signal_length, dtype=np.float32)
    # 가시 하모닉 수를 랜덤화 (1~5차)
    n_harmonics = rng.integers(1, 6)
    base_amp = rng.uniform(0.005, 0.02)
    for k in range(1, n_harmonics + 1):
        amp   = (base_amp / k) * rng.uniform(0.7, 1.3)
        phase = rng.uniform(0.0, 2 * np.pi)
        bg += amp * np.sin(2 * np.pi * k * rotation_freq * t + phase).astype(np.float32)

    # 유색 노이즈 (낮은 주파수 강조, IIR AR(1))
    from scipy.signal import lfilter
    alpha = 0.3
    white   = rng.standard_normal(signal_length).astype(np.float32) * noise_std
    colored = lfilter([alpha], [1.0, -(1.0 - alpha)], white).astype(np.float32)
    bg += colored
    return bg


# ============================================================
# 6. 파라미터 랜덤 샘플링 헬퍼
# ============================================================

def sample_fn_zeta(
    rng: Generator,
    specs: dict = BEARING_SPECS,
) -> tuple[float, float, float]:
    """
    다중모달 공진 주파수 fn, 좌/우 감쇠비 zeta_L/R를 샘플링한다.

    fn 샘플링:
    - specs에 'fn_modes' 키가 있으면 다중모달 (기본값)
    - 없으면 단일 N(fn_mean, fn_std) fallback (JSON calibration 호환)

    zeta 샘플링:
    - Uniform(zeta_min, zeta_max) — 음수 방지, 현실적 decay 확보
    """
    # fn: 다중모달 vs 단일모달
    if "fn_modes" in specs:
        modes = specs["fn_modes"]
        probs = np.array([m[2] for m in modes], dtype=np.float64)
        probs /= probs.sum()
        mode_idx = int(rng.choice(len(modes), p=probs))
        fn_mean, fn_std, _ = modes[mode_idx]
        fn_min = fn_mean * 0.50
        fn_max = fn_mean * 1.50
        fn = float(np.clip(rng.normal(fn_mean, fn_std), fn_min, fn_max))
    else:
        fn = float(np.clip(
            rng.normal(specs["fn_mean"], specs["fn_std"]),
            specs["fn_min"], specs["fn_max"],
        ))

    # zeta: Uniform (음수 완전 차단, 현실적 감쇠)
    z_min = specs.get("zeta_min", 0.01)
    z_max = specs.get("zeta_max", 0.04)
    zeta = float(rng.uniform(z_min, z_max))

    # 비대칭 감쇠비 (zeta_L < zeta < zeta_R)
    zeta_L = float(np.clip(zeta * rng.uniform(0.7, 1.0), z_min, z_max))
    zeta_R = float(np.clip(zeta * rng.uniform(1.0, 1.4), z_min, min(z_max * 1.5, 0.30)))
    return fn, zeta_L, zeta_R


def get_fault_freq_for_rpm(fault_type: str, rpm: float) -> float:
    """기준 RPM(1000)에서의 결함 주파수를 실제 RPM으로 선형 스케일링."""
    base_rpm = 1000.0
    base_freqs = BEARING_SPECS["fault_frequencies"]
    if fault_type not in base_freqs:
        raise ValueError(f"Unknown fault type: {fault_type}. Choose from {list(base_freqs.keys())}")
    return base_freqs[fault_type] * (rpm / base_rpm)


def get_all_fault_freqs_for_rpm(rpm: float) -> dict[str, float]:
    """모든 결함 타입의 주파수를 실제 RPM으로 스케일링하여 반환."""
    base_rpm = 1000.0
    base_freqs = BEARING_SPECS["fault_frequencies"]
    return {k: v * (rpm / base_rpm) for k, v in base_freqs.items()}