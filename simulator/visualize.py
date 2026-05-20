"""
simulator/visualize.py

생성된 합성 베어링 신호의 물리적 현실성을 검증하는 시각화 모듈.

지원하는 시각화:
  1. Raw Waveform     - 시간에 따른 진동 파형 (열화 진행 비교)
  2. FFT Spectrum     - 주파수 영역 분석 (결함 주파수 피크 확인)
  3. Envelope Spectrum - 포락선 스펙트럼 (BPFO/BPFI 하모닉 가시화)
  4. Spectrogram      - 공진 대역 및 하모닉 성장 확인
  5. RUL Trajectory   - 열화도(RMS, Kurtosis)가 시간에 따라 변하는 궤적
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")  # GUI 없이 파일로 저장
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.signal import hilbert


# ============================================================
# 내부 헬퍼
# ============================================================

def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)))


def _kurtosis(x: np.ndarray) -> float:
    x = x - x.mean()
    std = x.std()
    if std < 1e-10:
        return 0.0
    return float(np.mean((x / std) ** 4))


def _envelope_spectrum(signal: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
    """Hilbert Transform으로 포락선 스펙트럼 계산."""
    analytic = hilbert(signal)
    envelope = np.abs(analytic)
    envelope -= envelope.mean()
    freqs = np.fft.rfftfreq(len(envelope), 1.0 / fs)
    magnitudes = np.abs(np.fft.rfft(envelope)) * 2.0 / len(envelope)
    return freqs, magnitudes


# ============================================================
# 1. 단일 임펄스 커널 검증
# ============================================================

def plot_single_impulse(h: np.ndarray, fs: int, save_path: Path | None = None) -> None:
    """단일 임펄스 응답 h(t) 파형과 주파수 스펙트럼을 시각화한다."""
    t = np.arange(len(h)) / fs * 1000 - len(h) / (2 * fs) * 1000  # ms 단위
    freqs = np.fft.rfftfreq(len(h), 1.0 / fs)
    spec = np.abs(np.fft.rfft(h))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Single Fault Impulse Response h(t)", fontsize=14, fontweight="bold")

    ax1.plot(t, h, color="royalblue", linewidth=1.0)
    ax1.axvline(0, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Time Domain")
    ax1.grid(True, alpha=0.3)

    ax2.plot(freqs, spec, color="darkorange", linewidth=1.0)
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Magnitude")
    ax2.set_title("Frequency Domain")
    ax2.set_xlim(0, 5000)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [Viz] Saved: {save_path}")
    plt.close(fig)


# ============================================================
# 2. 열화 단계별 신호 비교 대시보드
# ============================================================

def plot_degradation_dashboard(
    run_data: dict,
    channel: int = 0,
    stages: list[int] | None = None,
    save_path: Path | None = None,
) -> None:
    """
    열화 진행의 4개 대표 타임스텝(Healthy/Onset/Growth/Rapid)에 대해
    Raw Waveform / Envelope Spectrum / Spectrogram을 나란히 시각화한다.

    Args:
        run_data: generate_run()의 반환 dict
        channel: 시각화할 채널 번호
        stages: 시각화할 타임스텝 인덱스 4개 (None이면 자동 선택)
        save_path: 저장 경로
    """
    vibration = run_data["vibration"]  # (seq_len, n_channels, signal_length)
    seq_len = vibration.shape[0]
    fs = run_data["config"].fs
    fault_freq = run_data["fault_freq"]
    fault_type = run_data["fault_type"]

    if stages is None:
        stages = [
            int(seq_len * 0.10),  # Healthy
            int(seq_len * 0.45),  # Onset
            int(seq_len * 0.72),  # Growth
            int(seq_len * 0.95),  # Rapid Failure
        ]
    stage_labels = ["Healthy (10%)", "Onset (45%)", "Growth (72%)", "Rapid (95%)"]

    n_stages = len(stages)
    fig = plt.figure(figsize=(6 * n_stages, 10))
    fig.suptitle(
        f"Physics-based Bearing Degradation Dashboard\n"
        f"Fault: {fault_type} | fault_freq={fault_freq:.1f}Hz | "
        f"fn={run_data['fn']:.0f}Hz | ζ_R={run_data['zeta_R']:.4f}",
        fontsize=12, fontweight="bold"
    )
    gs = gridspec.GridSpec(3, n_stages, figure=fig, hspace=0.4, wspace=0.35)

    for col, (step_idx, label) in enumerate(zip(stages, stage_labels)):
        sig = vibration[step_idx, channel, :]
        t = np.arange(len(sig)) / fs * 1000  # ms

        # ---------- Row 0: Raw Waveform ----------
        ax0 = fig.add_subplot(gs[0, col])
        ax0.plot(t[:2560], sig[:2560], linewidth=0.5, color="steelblue")  # 처음 100ms
        ax0.set_title(f"{label}\nRMS={_rms(sig):.4f} | Kurt={_kurtosis(sig):.2f}", fontsize=9)
        ax0.set_xlabel("Time (ms)" if col == 0 else "")
        ax0.set_ylabel("Amplitude" if col == 0 else "")
        ax0.grid(True, alpha=0.3)

        # ---------- Row 1: Envelope Spectrum ----------
        ax1 = fig.add_subplot(gs[1, col])
        env_freqs, env_mag = _envelope_spectrum(sig, fs)
        ax1.plot(env_freqs, env_mag, linewidth=0.6, color="darkorange")
        ax1.set_xlim(0, min(fault_freq * 8, 1000))
        # 결함 주파수 하모닉 표시
        for h_idx in range(1, 8):
            hf = fault_freq * h_idx
            if hf <= 1000:
                ax1.axvline(hf, color="red", linestyle="--", linewidth=0.7, alpha=0.6)
        ax1.set_xlabel("Frequency (Hz)" if col == 0 else "")
        ax1.set_ylabel("Envelope Mag." if col == 0 else "")
        ax1.set_title("Envelope Spectrum", fontsize=8)
        ax1.grid(True, alpha=0.3)

        # ---------- Row 2: Spectrogram ----------
        ax2 = fig.add_subplot(gs[2, col])
        # 단순 Short-time FFT로 스펙트로그램 생성 (scipy 없이)
        nperseg = 512
        n_frames = len(sig) // nperseg
        spec_matrix = []
        for fi in range(n_frames):
            frame = sig[fi * nperseg:(fi + 1) * nperseg] * np.hanning(nperseg)
            spec_matrix.append(np.abs(np.fft.rfft(frame)))
        spec_matrix = np.array(spec_matrix).T  # (freq, time)
        spec_db = 20 * np.log10(spec_matrix + 1e-10)
        ax2.imshow(
            spec_db,
            aspect="auto",
            origin="lower",
            extent=[0, len(sig) / fs * 1000, 0, fs // 2],
            cmap="inferno",
            vmin=spec_db.max() - 50,
        )
        ax2.set_ylim(0, 5000)
        ax2.axhline(run_data["fn"], color="cyan", linestyle="--", linewidth=0.8, alpha=0.7)
        ax2.set_xlabel("Time (ms)" if col == 0 else "")
        ax2.set_ylabel("Freq (Hz)" if col == 0 else "")
        ax2.set_title("Spectrogram", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [Viz] Saved: {save_path}")
    plt.close(fig)


# ============================================================
# 3. RUL 궤적 (물리 지표 시계열)
# ============================================================

def plot_rul_trajectory(
    run_data: dict,
    channel: int = 0,
    save_path: Path | None = None,
) -> None:
    """
    RUL 시퀀스 전체에 걸쳐 RMS와 Kurtosis의 변화를 시각화한다.
    실제 고장 직전에 이 지표들이 급격히 올라가야 물리적으로 올바른 시뮬레이션이다.
    """
    vibration = run_data["vibration"]  # (seq_len, n_channels, signal_length)
    rul = run_data["rul"]
    seq_len = vibration.shape[0]

    rms_arr = np.array([_rms(vibration[i, channel, :]) for i in range(seq_len)])
    kurt_arr = np.array([_kurtosis(vibration[i, channel, :]) for i in range(seq_len)])

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(
        f"RUL Trajectory | Fault: {run_data['fault_type']} | "
        f"fn={run_data['fn']:.0f}Hz",
        fontsize=12, fontweight="bold"
    )
    x = np.arange(seq_len)

    # 배경 구간 색상
    for ax in axes:
        ax.axvspan(0, seq_len * 0.30, alpha=0.07, color="green", label="Healthy")
        ax.axvspan(seq_len * 0.30, seq_len * 0.60, alpha=0.07, color="yellow", label="Onset")
        ax.axvspan(seq_len * 0.60, seq_len * 0.85, alpha=0.07, color="orange", label="Growth")
        ax.axvspan(seq_len * 0.85, seq_len, alpha=0.12, color="red", label="Rapid")

    axes[0].plot(x, rul, color="royalblue", linewidth=1.5)
    axes[0].set_ylabel("RUL (timesteps)")
    axes[0].set_title("Remaining Useful Life", fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, rms_arr, color="darkorange", linewidth=1.2)
    axes[1].set_ylabel("RMS")
    axes[1].set_title("Vibration RMS (Health Indicator)", fontsize=10)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(x, kurt_arr, color="crimson", linewidth=1.2)
    axes[2].set_ylabel("Kurtosis")
    axes[2].set_xlabel("Timestep (0=Healthy → 99=Failure)")
    axes[2].set_title("Kurtosis (Impulsiveness Indicator)", fontsize=10)
    axes[2].grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [Viz] Saved: {save_path}")
    plt.close(fig)
