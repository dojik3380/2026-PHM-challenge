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

    # 노이즈 설정 — floor를 낮춰 공진 피크가 도드라지게
    healthy_noise_std: float = 0.08    # 0.15 → 0.08: floor 절반 감소
    noise_growth_factor: float = 5.0   # failure 시점 RMS 폭증 유지

    # 진폭 설정 (임펄스) — 공진 피크가 spectrum에 prominent하게 발현되도록 증폭
    onset_amplitude: float = 0.50     # 0.25 → 0.50
    max_amplitude: float = 12.0       # 8.0 → 12.0 (실측 max 11.7에 근접)

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

def _degradation_amplitude(progress: float, config: SyntheticConfig,
                           phase_thresholds: tuple[float, float, float] | None = None) -> float:
    """
    열화 진행률(0.0 ~ 1.0)에 따라 임펄스 진폭을 계산한다.

    phase_thresholds = (onset, growth, rapid_fail) — Healthy~Onset 경계, Onset~Growth, Growth~Rapid.
    None이면 (0.30, 0.60, 0.85) default. 각 run마다 random sampling 가능.
    """
    if phase_thresholds is None:
        t1, t2, t3 = 0.30, 0.60, 0.85
    else:
        t1, t2, t3 = phase_thresholds

    if progress < t1:
        return 0.0
    elif progress < t2:
        t = (progress - t1) / max(t2 - t1, 1e-6)
        return config.onset_amplitude * t
    elif progress < t3:
        t = (progress - t2) / max(t3 - t2, 1e-6)
        start = config.onset_amplitude
        end = config.max_amplitude * 0.40
        return start + (end - start) * (t ** 2)
    else:
        t = (progress - t3) / max(1.0 - t3, 1e-6)
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
    phase_thresholds: tuple[float, float, float] | None = None,
    secondary_fault: dict | None = None,
) -> np.ndarray:
    """하나의 타임스텝, 하나의 채널에 해당하는 1차원 진동 신호를 생성한다.

    phase_thresholds: 각 run마다 다른 4단계 비율 (stochastic degradation phases)
    secondary_fault: multi-fault evolution용 {'progress_start': 0.7, 'type': 'BSF', 'amp_ratio': 0.3}
    """
    signal_length = int(config.fs * config.signal_duration_sec)
    noise_std = _degradation_noise_std(progress, config)

    base_amplitude = _degradation_amplitude(progress, config, phase_thresholds) * channel_amp_scale

    # Progress-dependent burst probability — early: 5%, late: 35%
    if base_amplitude > 0 and progress > 0.40:
        burst_prob = 0.05 + 0.30 * np.clip((progress - 0.40) / 0.60, 0.0, 1.0)
        if rng.uniform() < burst_prob:
            burst_factor = float(rng.uniform(2.0, 6.0))
            amplitude = base_amplitude * burst_factor
        else:
            amplitude = base_amplitude
    else:
        amplitude = base_amplitude

    # Weak-fault dampening: 50% 확률로 "near-healthy ambiguous" 신호 — 실측의 weak signature majority 반영
    # 단, 매우 late stage (progress > 0.9)는 dampen 하지 않음 (failure clear)
    if rng.uniform() < 0.40 and progress < 0.90:
        amplitude *= float(rng.uniform(0.10, 0.50))

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

        # Multi-fault evolution: run에 secondary_fault가 지정됐고 progress가 시작 시점 이후면 합산
        if secondary_fault is not None and progress >= secondary_fault["progress_start"]:
            sec_type = secondary_fault["type"]
            sec_amp = amplitude * float(secondary_fault["amp_ratio"]) * rng.uniform(0.5, 1.3)
            secondary_freq = get_fault_freq_for_rpm(sec_type, current_rpm)
            secondary = build_multi_resonance_impulse_train(
                signal_length=signal_length,
                fault_freq=secondary_freq,
                modes=resonance_modes,
                amplitude=sec_amp,
                slip_range=config.slip_range * 1.5,
                fs=config.fs,
                rng=rng,
            )
            secondary = add_sideband_modulation(secondary, shaft_freq, config.fs, rng)
            signal += secondary
        elif config.multi_fault and progress > 0.70:
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
    fn, zeta_L, zeta_R, _ = max(resonance_modes, key=lambda m: m[3])

    rpm_seq = _generate_rpm_trajectory(config, rng)

    channel_amp_scales = np.clip(
        rng.normal(1.0, config.channel_amp_jitter, size=config.n_channels),
        0.6, 1.4,
    ).astype(np.float32)

    # ── P1: Partial lifetime — 40% run은 부분 lifetime만, 60%는 full ──
    # 실측 validation/test는 partial이고 일부 train도 cliff까지 안 감
    partial_mode = rng.uniform() < 0.40
    if partial_mode:
        # 최종 progress가 0.5~0.95 사이 — run이 일찍 끝나는 시나리오
        progress_end = float(rng.uniform(0.50, 0.95))
    else:
        progress_end = 1.0
    progress_arr = np.linspace(0.0, progress_end, config.seq_len, dtype=np.float64)
    # RUL: 끝 시점이 (1-progress_end)*total_life 만큼 남도록 설계 (failure 도달 X 시나리오)
    rul_remaining_at_end = float((1.0 - progress_end) * config.total_life_seconds)
    rul_arr = np.linspace(config.total_life_seconds, rul_remaining_at_end, config.seq_len, dtype=np.float32)

    # ── P2: Stochastic degradation phases — run마다 4단계 비율 다름 ──
    # default: (0.30, 0.60, 0.85). jitter로 각 경계 ±0.07
    t1 = float(np.clip(rng.normal(0.30, 0.07), 0.10, 0.45))
    t2 = float(np.clip(rng.normal(0.60, 0.07), t1 + 0.10, 0.75))
    t3 = float(np.clip(rng.normal(0.85, 0.05), t2 + 0.05, 0.95))
    phase_thresholds = (t1, t2, t3)

    # ── P3: Resonance drift — fn이 run 중 slow random walk ──
    # 각 step에서 fn을 ±drift_pct% 변동시킴. 누적 분산이 너무 커지지 않게 walk_std 조절
    drift_pct = 0.05  # ±5% 최대 drift
    drift_steps = rng.normal(0.0, 0.005, size=config.seq_len).cumsum()  # cumulative walk
    drift_steps = np.clip(drift_steps, -drift_pct, drift_pct)

    # ── P5: Multi-fault evolution — 30% run은 secondary fault가 late stage에 발현 ──
    secondary_fault = None
    if rng.uniform() < 0.30:
        candidates = [f for f in ("BPFO", "BPFI", "BSF") if f != config.fault_type]
        sec_type = str(rng.choice(candidates))
        secondary_fault = {
            "progress_start": float(rng.uniform(0.60, 0.85)),
            "type": sec_type,
            "amp_ratio": float(rng.uniform(0.20, 0.50)),
        }

    for step_idx, progress in enumerate(progress_arr):
        current_rpm = float(rpm_seq[step_idx])
        current_fault_freq = get_fault_freq_for_rpm(config.fault_type, current_rpm)

        # Resonance drift 적용 — modes의 fn을 step별로 약간 이동
        drift_factor = 1.0 + drift_steps[step_idx]
        step_modes = [(fn_m * drift_factor, zL, zR, w) for (fn_m, zL, zR, w) in resonance_modes]

        for ch in range(config.n_channels):
            vibration[step_idx, ch, :] = _generate_timestep_signal(
                progress=float(progress),
                config=config,
                resonance_modes=step_modes,
                current_rpm=current_rpm,
                current_fault_freq=current_fault_freq,
                channel_amp_scale=float(channel_amp_scales[ch]),
                rng=rng,
                phase_thresholds=phase_thresholds,
                secondary_fault=secondary_fault,
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
