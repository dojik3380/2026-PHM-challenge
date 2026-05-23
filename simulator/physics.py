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
    # 다중모달 공진 주파수 — 실측 스펙트럼(4kHz sharp, 6-7kHz broad, 12kHz sharp) 반영.
    # 형식: (mean_Hz, std_Hz, weight)
    "fn_modes": [
        (4000.0,  400.0, 0.30),   # 1차 sharp 공진
        (6500.0,  800.0, 0.30),   # 2차 broad housing resonance
        (12000.0, 700.0, 0.40),   # 3차 sharp high-freq structure resonance
    ],
    # JSON calibration fallback 용 단일 모달 통계
    "fn_mean": 6500.0,
    "fn_std":  500.0,
    "fn_min":  2000.0,
    "fn_max":  15000.0,
    # 감쇠비: Uniform — 음수 방지, 현실적 decay 길이 확보
    "zeta_min": 0.01,
    "zeta_max": 0.04,
    "zeta_mean": 0.025,
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
    # 회전 하모닉을 high-order까지 확장 — 실측의 0-1.5kHz 풍부한 line spectrum 모사
    # RPM 880 기준 회전 주파수 14.7Hz → 100차 = 1470Hz, 30차부터 mechanical resonance로 amplified
    n_harmonics = rng.integers(40, 80)  # 2~8 → 40~80차 (저주파 line spectrum 풍부)
    base_amp = rng.uniform(0.03, 0.10)
    # 30차 이후 high-order는 mechanical resonance에 의해 boost되는 영역 모사
    for k in range(1, n_harmonics + 1):
        # k^0.6 decay: 매우 high-order도 충분히 살아남음
        amp = (base_amp / (k ** 0.6)) * rng.uniform(0.5, 1.5)
        # 30~80차 영역에 boost (mechanical 공진과의 결합 시뮬레이션)
        if 30 <= k <= 80:
            amp *= rng.uniform(1.5, 3.0)
        phase = rng.uniform(0.0, 2 * np.pi)
        bg += amp * np.sin(2 * np.pi * k * rotation_freq * t + phase).astype(np.float32)

    # 광대역(거의 white) 배경 노이즈 — 실측 spectrum의 전 대역 floor 반영
    white = rng.standard_normal(signal_length).astype(np.float32) * noise_std
    bg += white

    # 공진 대역 연속 excitation — mean spectrum에 공진 피크가 나타나도록
    # 700Hz/1100Hz mid-freq mechanical + 4kHz/6.5kHz/12kHz structure resonance
    resonance_excitation = 0.20
    for fn_center, weight, jitter_pct, n_lines in [
        (700.0,   0.25, 0.04, 5),   # mid-freq mechanical cluster #1 (gear/cage harmonics)
        (1100.0,  0.20, 0.05, 5),   # mid-freq mechanical cluster #2
        (4000.0,  0.30, 0.02, 3),   # 1차 sharp structure resonance
        (6500.0,  0.30, 0.05, 4),   # 2차 broad housing resonance
        (12000.0, 0.40, 0.02, 3),   # 3차 sharp high-freq resonance
    ]:
        for _ in range(n_lines):
            fn_jit = fn_center * (1.0 + rng.normal(0.0, jitter_pct))
            amp_jit = resonance_excitation * weight * rng.uniform(0.6, 1.4)
            phase = rng.uniform(0.0, 2 * np.pi)
            bg += amp_jit * np.sin(2 * np.pi * fn_jit * t + phase).astype(np.float32)

    # DC 제거 (저주파 누적 방지)
    bg = bg - bg.mean()
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


def sample_multi_fn_zeta(
    rng: Generator,
    specs: dict = BEARING_SPECS,
    max_modes: int = 3,
) -> list[tuple[float, float, float, float]]:
    """모든 fn_modes를 동시에 샘플링한다 — superposition용 다중 공진 발현.

    Returns:
        [(fn, zeta_L, zeta_R, weight), ...] — weight는 mode prob을 사용.
    """
    if "fn_modes" not in specs:
        fn, zL, zR = sample_fn_zeta(rng, specs)
        return [(fn, zL, zR, 1.0)]

    modes = specs["fn_modes"][:max_modes]
    results = []
    z_min = specs.get("zeta_min", 0.01)
    z_max = specs.get("zeta_max", 0.04)
    for fn_mean, fn_std, weight in modes:
        fn_lo = fn_mean * 0.50
        fn_hi = fn_mean * 1.50
        fn = float(np.clip(rng.normal(fn_mean, fn_std), fn_lo, fn_hi))
        zeta = float(rng.uniform(z_min, z_max))
        zeta_L = float(np.clip(zeta * rng.uniform(0.7, 1.0), z_min, z_max))
        zeta_R = float(np.clip(zeta * rng.uniform(1.0, 1.4), z_min, min(z_max * 1.5, 0.30)))
        results.append((fn, zeta_L, zeta_R, float(weight)))
    return results


def build_multi_resonance_impulse_train(
    signal_length: int,
    fault_freq: float,
    modes: list[tuple[float, float, float, float]],
    amplitude: float,
    slip_range: float = 0.10,
    fs: int = 25_600,
    rng: Generator | None = None,
) -> np.ndarray:
    """여러 공진 mode를 superposition한 임펄스 train을 생성한다.

    각 mode는 (fn, zeta_L, zeta_R, weight). 동일 시점에 각 mode의 impulse response를
    weight 비율로 합성한 뒤 동일 train으로 배치한다.
    """
    if rng is None:
        rng = np.random.default_rng()

    # 정규화 없이 weighted superposition — 각 mode peak이 spectrum에 그대로 나타남
    h_list = [(single_impulse_response(fn, zL, zR, fs=fs), w) for fn, zL, zR, w in modes]
    h_combined = np.zeros_like(h_list[0][0])
    for h, w in h_list:
        h_combined += w * h

    h_len = len(h_combined)
    h_center = h_len // 2
    T_samples = fs / fault_freq

    y = np.zeros(signal_length, dtype=np.float32)
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

        y[start_sig:end_sig] += amplitude * h_combined[start_ker:end_ker]
        k += 1

    return y


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