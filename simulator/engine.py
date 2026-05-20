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
    make_background_vibration,
    sample_fn_zeta,
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

    # 운전 조건
    rpm: float = 1000.0                # 회전 속도 (RPM)
    rpm_jitter: float = 5.0           # ±RPM 변동 (운전 불안정성)

    # 결함 설정
    fault_type: Literal["BPFI", "BPFO", "BSF"] = "BPFO"
    multi_fault: bool = False          # True이면 여러 결함 주파수를 혼합

    # 슬립(Jitter) 설정
    slip_range: float = 0.10           # ±10% 타이밍 지터

    # 노이즈 설정
    healthy_noise_std: float = 0.02    # Healthy 구간 노이즈 크기
    noise_growth_factor: float = 2.5   # Failure 시점 노이즈 배율

    # 진폭 설정 (임펄스)
    onset_amplitude: float = 0.05     # Onset 시점 초기 임펄스 진폭
    max_amplitude: float = 3.0        # Rapid Failure 최대 진폭

    # 채널 다양성
    channel_phase_jitter: float = 0.05  # 채널 간 위상차 (라디안 단위 노이즈로 표현)
    channel_amp_jitter: float = 0.1     # 채널 간 진폭 배율 변동 (±10%)

    # 임펄스 창 길이
    impulse_duration_ms: float = 3.0

    # 재현성
    seed: int | None = None


# ============================================================
# 열화 진행 계산 함수
# ============================================================

def _degradation_amplitude(progress: float, config: SyntheticConfig) -> float:
    """
    열화 진행률(0.0 ~ 1.0)에 따라 임펄스 진폭을 계산한다.

    구간:
      0.00 ~ 0.30 : Healthy  → amplitude = 0
      0.30 ~ 0.60 : Onset    → linear ramp from 0 to onset_amplitude
      0.60 ~ 0.85 : Growth   → quadratic ramp to max * 0.4
      0.85 ~ 1.00 : Rapid    → exponential surge to max_amplitude
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
# 단일 채널 타임스텝 신호 생성
# ============================================================

def _generate_timestep_signal(
    progress: float,
    config: SyntheticConfig,
    fn: float,
    zeta_L: float,
    zeta_R: float,
    fault_freq: float,
    channel_amp_scale: float,
    rng: Generator,
) -> np.ndarray:
    """하나의 타임스텝, 하나의 채널에 해당하는 1차원 진동 신호를 생성한다."""
    signal_length = int(config.fs * config.signal_duration_sec)
    amplitude = _degradation_amplitude(progress, config) * channel_amp_scale
    noise_std = _degradation_noise_std(progress, config)

    # 1. 배경 진동 생성
    rpm_actual = config.rpm + rng.uniform(-config.rpm_jitter, config.rpm_jitter)
    signal = make_background_vibration(signal_length, rpm_actual, noise_std, config.fs, rng)

    # 2. 임펄스 트레인 합산 (Onset 이후에만)
    if amplitude > 0:
        impulse = build_impulse_train(
            signal_length=signal_length,
            fault_freq=fault_freq,
            fn=fn,
            zeta_L=zeta_L,
            zeta_R=zeta_R,
            amplitude=amplitude,
            slip_range=config.slip_range,
            fs=config.fs,
            rng=rng,
        )
        signal += impulse

        # Multi-fault: BSF 또는 Cage 혼합 (열화 후반부에 소량 추가)
        if config.multi_fault and progress > 0.70:
            secondary_type = "BSF" if config.fault_type != "BSF" else "Cage"
            secondary_freq = get_fault_freq_for_rpm(secondary_type, rpm_actual)
            secondary_amp = amplitude * rng.uniform(0.15, 0.35)
            signal += build_impulse_train(
                signal_length=signal_length,
                fault_freq=secondary_freq,
                fn=fn,
                zeta_L=zeta_L,
                zeta_R=zeta_R,
                amplitude=secondary_amp,
                slip_range=config.slip_range * 1.5,
                fs=config.fs,
                rng=rng,
            )

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

    Returns:
        {
          "vibration": np.ndarray shape (seq_len, n_channels, signal_length),
          "rul":       np.ndarray shape (seq_len,),  -- 남은 수명 (타임스텝 단위)
          "progress":  np.ndarray shape (seq_len,),  -- 0.0~1.0 열화 진행률
          "fault_type": str,
          "rpm": float,
          "fault_freq": float,
          "fn": float,
          "zeta_L": float,
          "zeta_R": float,
          "config": SyntheticConfig,
        }
    """
    if rng is None:
        rng = np.random.default_rng(config.seed)

    signal_length = int(config.fs * config.signal_duration_sec)
    vibration = np.zeros(
        (config.seq_len, config.n_channels, signal_length), dtype=np.float32
    )

    # 이 Run 전체에 걸친 물리 파라미터 샘플링 (베어링마다 다른 특성)
    fn, zeta_L, zeta_R = sample_fn_zeta(rng, BEARING_SPECS)
    rpm_base = config.rpm + rng.uniform(-config.rpm_jitter * 2, config.rpm_jitter * 2)
    fault_freq = get_fault_freq_for_rpm(config.fault_type, rpm_base)

    # 채널별 진폭 스케일 (센서 위치 차이 모사)
    channel_amp_scales = np.clip(
        rng.normal(1.0, config.channel_amp_jitter, size=config.n_channels),
        0.6, 1.4,
    ).astype(np.float32)

    progress_arr = np.linspace(0.0, 1.0, config.seq_len, dtype=np.float64)
    rul_arr = np.array([config.seq_len - 1 - i for i in range(config.seq_len)], dtype=np.float32)

    for step_idx, progress in enumerate(progress_arr):
        for ch in range(config.n_channels):
            vibration[step_idx, ch, :] = _generate_timestep_signal(
                progress=float(progress),
                config=config,
                fn=fn,
                zeta_L=zeta_L,
                zeta_R=zeta_R,
                fault_freq=fault_freq,
                channel_amp_scale=float(channel_amp_scales[ch]),
                rng=rng,
            )

    return {
        "vibration": vibration,
        "rul": rul_arr,
        "progress": progress_arr.astype(np.float32),
        "fault_type": config.fault_type,
        "rpm": rpm_base,
        "fault_freq": fault_freq,
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

    Args:
        n_runs: 생성할 Run 수
        config: SyntheticConfig (None이면 기본값 사용)
        fault_types: 결함 타입 목록 (예: ["BPFO", "BPFI", "BSF"])
        verbose: 진행 상황 출력 여부

    Returns:
        runs: 각 Run의 dict를 담은 리스트
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
        run_config = SyntheticConfig(
            **{
                **config.__dict__,
                "fault_type": fault_type,
                "seed": run_seed,
            }
        )
        run_rng = np.random.default_rng(run_seed)
        run_data = generate_run(run_config, rng=run_rng)
        runs.append(run_data)

        if verbose and (i + 1) % 10 == 0:
            print(f"  [Engine] Generated {i + 1}/{n_runs} runs | "
                  f"fault={fault_type} | fn={run_data['fn']:.1f}Hz | "
                  f"zeta_L={run_data['zeta_L']:.4f}")

    return runs
