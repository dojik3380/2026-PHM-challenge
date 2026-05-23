"""
simulator/engine.py

열화 진행 엔진(Degradation Progression Engine).
0% (Healthy) → 100% (Failure) 까지의 전체 RUL Sequence를 생성한다.

열화 단계:
  0~30%   Healthy     : 임펄스 없음, 배경 노이즈만
  30~60%  Onset       : 약한 임펄스 시작 (선형 성장)
  60~85%  Growth      : 임펄스 + 하모닉 성장 (제곱 가속)
  85~100% Rapid Fail  : 지수 폭증

출력 텐서 shape: (seq_len, channels, signal_length)
  seq_len       : RUL 타임스텝 수 (예: 100)
  channels      : 진동 채널 수 (예: 4)
  signal_length : 타임스텝당 샘플 수 (예: 25600 = 1초 @ 25.6kHz)
"""
from __future__ import annotations

import numpy as np
from numpy.random import Generator
from dataclasses import dataclass, field
from typing import Literal

from .physics import (
    build_impulse_train,
    build_multi_resonance_impulse_train,
    make_background_vibration,
    add_sideband_modulation,
    sample_fn_zeta,
    sample_multi_fn_zeta,
    get_fault_freq_for_rpm,
    BEARING_SPECS,
)


# ============================================================
# 설정 데이터 클래스
# ============================================================

@dataclass
class SyntheticConfig:
    """시뮬레이터 전체 파라미터 설정."""

    # 신호 파라미터
    fs: int = 25_600                   # 샘플링 레이트 (Hz)
    signal_duration_sec: float = 1.0   # 타임스텝당 신호 길이 (초)

    # 열화 타임스텝
    seq_len: int = 100                 # 한 Run에서의 RUL 타임스텝 수

    # 채널 구성
    n_channels: int = 4                # 진동 채널 수

    # 운전 조건 (Switching RPM)
    rpm_low_range: tuple = (700.0, 760.0)   # low-speed regime
    rpm_high_range: tuple = (940.0, 980.0)  # high-speed regime
    rpm_switch_steps: int = 40              # regime당 유지 timestep 수 (≥ window_size 권장)
    rpm_transition_steps: int = 3           # 전환에 걸리는 timestep 수
    rpm_jitter: float = 10.0               # ±RPM 변동 (운전 불안정성)

    # 결함 설정
    fault_type: Literal["BPFI", "BPFO", "BSF"] = "BPFO"
    multi_fault: bool = False          # True이면 여러 결함 주파수를 혼합

    # 슬립(Jitter) 설정
    slip_range: float = 0.10           # ±10% 타이밍 지터

    # 노이즈 설정 — 실측 RMS(~0.25g)에 맞춰 ~10배 증폭
    healthy_noise_std: float = 0.15    # 0.02 → 0.15: 실측 noise floor 반영
    noise_growth_factor: float = 5.0   # 2.5 → 5.0: failure 시점 RMS 폭증 반영

    # 진폭 설정 (임펄스)
    onset_amplitude: float = 0.25     # 0.05 → 0.25
    max_amplitude: float = 8.0        # 3.0 → 8.0 (실측 max 11.7 고려)

    # 다중 공진 활성화 — 4kHz/6.5kHz/12kHz를 동시에 발현
    multi_resonance: bool = True

    # 채널 다양성
    channel_phase_jitter: float = 0.05  # 채널 간 위상차 (라디안 단위 노이즈로 표현)
    channel_amp_jitter: float = 0.1     # 채널 간 진폭 배율 변동 (±10%)

    # 임펄스 창 길이
    impulse_duration_ms: float = 3.0

    # RUL 스케일: 실제 데이터 수명 범위(초)에 맞게 설정
    # Train1=75251s, Train2=67979s, Train3=53225s, Train4=82613s → 범위 50000~90000
    total_life_seconds: float = 70000.0
    total_life_seconds_min: float = 50000.0  # run별 수명 랜덤화 하한
    total_life_seconds_max: float = 90000.0  # run별 수명 랜덤화 상한

    # 재현성
    seed: int | None = None

    # 베어링 물리 파라미터 (None이면 BEARING_SPECS 기본값 사용)
    bearing_specs: dict | None = None


# ============================================================
# 열화 진행 계산 함수
# ============================================================

def _degradation_amplitude(progress: float, config: SyntheticConfig) -> float:
    """
    열화 진행률(0.0 ~ 1.0)에 따라 임펄스 진폭을 계산한다.
    """
    if progress < 0.30:
        return 0.0
    elif progress < 0.60:
        # 선형 성장 (Onset)
        t = (progress - 0.30) / 0.30
        return config.onset_amplitude * t
    elif progress < 0.85:
        # 제곱 가속 (Growth)
        t = (progress - 0.60) / 0.25
        start = config.onset_amplitude
        end = config.max_amplitude * 0.40
        return start + (end - start) * (t ** 2)
    else:
        # 지수 폭증 (Rapid Degradation)
        t = (progress - 0.85) / 0.15
        start = config.max_amplitude * 0.40
        end = config.max_amplitude
        return start + (end - start) * (np.exp(3.0 * t) - 1.0) / (np.exp(3.0) - 1.0)


def _degradation_noise_std(progress: float, config: SyntheticConfig) -> float:
    """열화 진행에 따라 배경 노이즈 크기를 선형적으로 증가시킨다."""
    return config.healthy_noise_std * (
        1.0 + (config.noise_growth_factor - 1.0) * max(0.0, (progress - 0.30) / 0.70)
    )

# ============================================================
# RPM Trajectory 생성 함수
# ============================================================
def _generate_rpm_trajectory(config: SyntheticConfig, rng: Generator) -> np.ndarray:
    """Switching RPM(t) trajectory를 생성한다."""
    rpm_seq = np.zeros(config.seq_len, dtype=np.float32)
    
    current_state = rng.choice(["low", "high"])
    step = 0
    
    while step < config.seq_len:
        # 현재 regime 목표 RPM 샘플링
        if current_state == "low":
            target_rpm = rng.uniform(*config.rpm_low_range)
        else:
            target_rpm = rng.uniform(*config.rpm_high_range)
            
        # 유지 구간 설정
        hold_steps = config.rpm_switch_steps + rng.integers(-3, 4)
        end_hold = min(step + hold_steps, config.seq_len)
        rpm_seq[step:end_hold] = target_rpm
        step = end_hold
        
        # 전환 구간 (transition)
        if step < config.seq_len:
            transition_steps = config.rpm_transition_steps + rng.integers(0, 2)
            end_trans = min(step + transition_steps, config.seq_len)
            
            next_state = "high" if current_state == "low" else "low"
            if next_state == "low":
                next_rpm = rng.uniform(*config.rpm_low_range)
            else:
                next_rpm = rng.uniform(*config.rpm_high_range)
                
            # 선형 보간으로 부드럽게 전환
            if end_trans > step:
                rpm_seq[step:end_trans] = np.linspace(target_rpm, next_rpm, end_trans - step)
                
            current_state = next_state
            step = end_trans

    # 지터 추가
    rpm_seq += rng.uniform(-config.rpm_jitter, config.rpm_jitter, size=config.seq_len)
    return rpm_seq


# ============================================================
# 단일 채널 타임스텝 신호 생성
# ============================================================

def _generate_timestep_signal(
    progress: float,
    config: SyntheticConfig,
    resonance_modes: list,
    current_rpm: float,
    current_fault_freq: float,
    channel_amp_scale: float,
    rng: Generator,
) -> np.ndarray:
    """하나의 타임스텝, 하나의 채널에 해당하는 1차원 진동 신호를 생성한다.

    resonance_modes:
        config.multi_resonance=True 시 [(fn, zL, zR, weight), ...]
        False 시 [(fn, zL, zR, 1.0)] 단일 mode
    """
    signal_length = int(config.fs * config.signal_duration_sec)
    noise_std = _degradation_noise_std(progress, config)

    # Intermittent burst: Onset 이후 15% 확률로 순간 진폭 급증
    base_amplitude = _degradation_amplitude(progress, config) * channel_amp_scale
    if base_amplitude > 0 and progress > 0.45 and rng.uniform() < 0.15:
        burst_factor = float(rng.uniform(2.0, 5.0))
        amplitude = base_amplitude * burst_factor
    else:
        amplitude = base_amplitude

    # 1. 배경 진동 (RPM 기반, 랜덤 하모닉)
    signal = make_background_vibration(signal_length, current_rpm, noise_std, config.fs, rng)

    # 2. 임펄스 트레인 — multi-resonance superposition
    if amplitude > 0:
        impulse = build_multi_resonance_impulse_train(
            signal_length=signal_length,
            fault_freq=current_fault_freq,
            modes=resonance_modes,
            amplitude=amplitude,
            slip_range=config.slip_range,
            fs=config.fs,
            rng=rng,
        )
        shaft_freq = current_rpm / 60.0
        impulse = add_sideband_modulation(impulse, shaft_freq, config.fs, rng)
        signal += impulse

        if config.multi_fault and progress > 0.70:
            secondary_type = "BSF" if config.fault_type != "BSF" else "Cage"
            secondary_freq = get_fault_freq_for_rpm(secondary_type, current_rpm)
            secondary_amp = amplitude * rng.uniform(0.15, 0.35)
            secondary = build_multi_resonance_impulse_train(
                signal_length=signal_length,
                fault_freq=secondary_freq,
                modes=resonance_modes,
                amplitude=secondary_amp,
                slip_range=config.slip_range * 1.5,
                fs=config.fs,
                rng=rng,
            )
            secondary = add_sideband_modulation(secondary, shaft_freq, config.fs, rng)
            signal += secondary

    return signal


# ============================================================
# 전체 Run (한 베어링의 수명 전체) 생성
# ============================================================

def generate_run(
    config: SyntheticConfig,
    rng: Generator | None = None,
) -> dict:
    """
    한 베어링의 처음(Healthy)부터 고장(Failure)까지의
    전체 RUL 시퀀스를 생성한다.
    """
    if rng is None:
        rng = np.random.default_rng(config.seed)

    signal_length = int(config.fs * config.signal_duration_sec)
    vibration = np.zeros(
        (config.seq_len, config.n_channels, signal_length), dtype=np.float32
    )

    # 이 Run 전체에 걸친 물리 파라미터 샘플링 (베어링마다 다른 특성)
    specs = config.bearing_specs if config.bearing_specs is not None else BEARING_SPECS
    if config.multi_resonance:
        resonance_modes = sample_multi_fn_zeta(rng, specs)
    else:
        fn, zL, zR = sample_fn_zeta(rng, specs)
        resonance_modes = [(fn, zL, zR, 1.0)]
    # 대표 fn / zeta — 메타데이터용 (가장 큰 weight)
    fn, zeta_L, zeta_R, _ = max(resonance_modes, key=lambda m: m[3])

    # RPM trajectory 생성
    rpm_seq = _generate_rpm_trajectory(config, rng)

    # 채널별 진폭 스케일 (센서 위치 차이 모사)
    channel_amp_scales = np.clip(
        rng.normal(1.0, config.channel_amp_jitter, size=config.n_channels),
        0.6, 1.4,
    ).astype(np.float32)

    progress_arr = np.linspace(0.0, 1.0, config.seq_len, dtype=np.float64)
    # RUL을 초 단위로 생성해 실제 데이터(수만 초)와 log1p 스케일이 맞도록 한다
    rul_arr = np.linspace(config.total_life_seconds, 0.0, config.seq_len, dtype=np.float32)

    for step_idx, progress in enumerate(progress_arr):
        current_rpm = float(rpm_seq[step_idx])
        current_fault_freq = get_fault_freq_for_rpm(config.fault_type, current_rpm)
        
        for ch in range(config.n_channels):
            vibration[step_idx, ch, :] = _generate_timestep_signal(
                progress=float(progress),
                config=config,
                resonance_modes=resonance_modes,
                current_rpm=current_rpm,
                current_fault_freq=current_fault_freq,
                channel_amp_scale=float(channel_amp_scales[ch]),
                rng=rng,
            )

    return {
        "vibration": vibration,
        "rul": rul_arr,
        "progress": progress_arr.astype(np.float32),
        "rpm_trajectory": rpm_seq,
        "fault_type": config.fault_type,
        "fn": fn,
        "zeta_L": zeta_L,
        "zeta_R": zeta_R,
        "config": config,
    }


# ============================================================
# 데이터셋 대량 생성
# ============================================================

def generate_dataset(
    n_runs: int = 50,
    config: SyntheticConfig | None = None,
    fault_types: list[str] | None = None,
    verbose: bool = True,
) -> list[dict]:
    """
    N개의 가상 베어링 Run을 생성하여 리스트로 반환한다.
    fault_types를 지정하면 결함 타입을 순환하며 다양하게 생성한다.
    """
    if config is None:
        config = SyntheticConfig()
    if fault_types is None:
        fault_types = ["BPFO", "BPFI", "BSF"]

    runs = []
    master_rng = np.random.default_rng(config.seed)

    for i in range(n_runs):
        # 결함 타입 순환
        fault_type = fault_types[i % len(fault_types)]

        # 각 Run마다 독립 RNG (재현 가능)
        run_seed = int(master_rng.integers(0, 2**31))
        run_rng = np.random.default_rng(run_seed)

        # 실제 데이터처럼 run마다 수명을 랜덤화
        life_sec = float(run_rng.uniform(config.total_life_seconds_min,
                                         config.total_life_seconds_max))

        run_config = SyntheticConfig(
            **{
                **config.__dict__,
                "fault_type": fault_type,
                "seed": run_seed,
                "total_life_seconds": life_sec,
            }
        )
        run_data = generate_run(run_config, rng=run_rng)
        runs.append(run_data)

        if verbose and (i + 1) % 10 == 0:
            print(f"  [Engine] Generated {i + 1}/{n_runs} runs | "
                  f"fault={fault_type} | life={life_sec:.0f}s | "
                  f"fn={run_data['fn']:.1f}Hz | zeta_L={run_data['zeta_L']:.4f}")

    return runs
