"""
visualization/comparison.py

Synthetic vs Real TDMS 비교 대시보드.

사용법:
    python -m visualization.comparison
    python -m visualization.comparison --n-runs 3 --save-dir outputs/simulation_comparison
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import stft as scipy_stft, hilbert


# ============================================================
# 신호 분석 유틸
# ============================================================

def _envelope_spectrum(signal: np.ndarray, fs: int, n_fft: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Hilbert 포락선 → FFT → 포락선 스펙트럼."""
    analytic = hilbert(signal - signal.mean())
    envelope = np.abs(analytic)
    envelope -= envelope.mean()
    n = min(len(envelope), n_fft)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    spec = np.abs(np.fft.rfft(envelope[:n])) / n
    return freqs, spec


def _fft_spectrum(signal: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    spec = np.abs(np.fft.rfft(signal)) / n
    return freqs, spec


def _rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(signal ** 2)))


def _kurtosis(signal: np.ndarray) -> float:
    from scipy.stats import kurtosis
    return float(kurtosis(signal, fisher=False))


# ============================================================
# 실제 TDMS 로드 (첫 번째 케이스)
# ============================================================

def _load_real_signals(tdms_dir: Path, n_files: int = 10) -> list[np.ndarray]:
    """TDMS 디렉토리에서 CH1 신호를 최대 n_files개 읽는다."""
    try:
        from data_loader import load_tdms_channels
    except ImportError:
        return []

    files = sorted(tdms_dir.glob("*.tdms"))[:n_files]
    signals = []
    for f in files:
        try:
            ch = load_tdms_channels(f)
            normed = {k.upper(): v for k, v in ch.items()}
            sig = np.array(normed.get("CH1", []), dtype=np.float32)
            if len(sig) > 0:
                signals.append(sig)
        except Exception:
            continue
    return signals


# ============================================================
# 비교 플롯 생성
# ============================================================

def _find_real_dir() -> Path | None:
    """프로젝트 루트의 Train 데이터에서 첫 번째 진동 디렉토리를 찾는다."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    try:
        from config import TRAIN_DIR
        from data_loader import discover_cases
        cases = discover_cases(TRAIN_DIR)
        if cases:
            return cases[0][2]  # (case_name, csv_path, vibration_dir)
    except Exception:
        pass
    return None


def plot_comparison(
    synthetic_run: dict,
    real_signals: list[np.ndarray],
    fs: int = 25_600,
    save_dir: Path = Path("outputs/simulation_comparison"),
    run_idx: int = 0,
) -> None:
    """
    Synthetic run과 Real TDMS 신호를 비교하는 4-panel 대시보드를 저장한다.

    패널:
      1. FFT 스펙트럼 비교
      2. 포락선 스펙트럼 비교
      3. RMS 추세 비교
      4. Kurtosis 추세 비교
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    vib = synthetic_run["vibration"]      # (seq_len, n_ch, sig_len)
    rul = synthetic_run["rul"]             # (seq_len,)
    progress = synthetic_run["progress"]  # (seq_len,)
    seq_len, n_ch, sig_len = vib.shape

    # ── Synthetic 통계 계산 ────────────────────────────────────
    syn_rms  = [_rms(vib[i, 0, :])       for i in range(seq_len)]
    syn_kurt = [_kurtosis(vib[i, 0, :])  for i in range(seq_len)]
    syn_mid  = vib[seq_len // 2, 0, :]   # 중간 열화 단계 신호

    # ── Real 통계 계산 ─────────────────────────────────────────
    real_rms  = [_rms(s)      for s in real_signals]
    real_kurt = [_kurtosis(s) for s in real_signals]
    real_mid  = real_signals[len(real_signals) // 2] if real_signals else None

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Synthetic vs Real — Run {run_idx:03d} | fault={synthetic_run['fault_type']}",
                 fontsize=13, fontweight="bold")

    # 1. FFT 스펙트럼
    ax = axes[0, 0]
    f_s, sp_s = _fft_spectrum(syn_mid, fs)
    ax.semilogy(f_s / 1000, sp_s + 1e-12, color="steelblue", alpha=0.85, label="Synthetic (mid-life)")
    if real_mid is not None:
        f_r, sp_r = _fft_spectrum(real_mid, fs)
        ax.semilogy(f_r / 1000, sp_r + 1e-12, color="tomato", alpha=0.75, label="Real (mid-life)")
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Amplitude (log)")
    ax.set_title("FFT Spectrum")
    ax.set_xlim(0, fs / 2 / 1000)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. 포락선 스펙트럼
    ax = axes[0, 1]
    f_s, ep_s = _envelope_spectrum(syn_mid, fs)
    ax.plot(f_s, ep_s, color="steelblue", alpha=0.85, label="Synthetic")
    if real_mid is not None:
        f_r, ep_r = _envelope_spectrum(real_mid, fs)
        ax.plot(f_r, ep_r, color="tomato", alpha=0.75, label="Real")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Envelope Spectrum")
    ax.set_xlim(0, min(500, fs / 2))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. RMS 추세
    ax = axes[1, 0]
    ax.plot(progress, syn_rms, color="steelblue", linewidth=1.5, label="Synthetic")
    if real_rms:
        real_x = np.linspace(0, 1, len(real_rms))
        ax.plot(real_x, real_rms, color="tomato", linewidth=1.5, label="Real")
    ax.set_xlabel("Degradation Progress")
    ax.set_ylabel("RMS")
    ax.set_title("RMS Trend")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4. Kurtosis 추세
    ax = axes[1, 1]
    ax.plot(progress, syn_kurt, color="steelblue", linewidth=1.5, label="Synthetic")
    if real_kurt:
        real_x = np.linspace(0, 1, len(real_kurt))
        ax.plot(real_x, real_kurt, color="tomato", linewidth=1.5, label="Real")
    ax.set_xlabel("Degradation Progress")
    ax.set_ylabel("Kurtosis")
    ax.set_title("Kurtosis Trend")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = save_dir / f"comparison_run{run_idx:03d}.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Comparison] Saved -> {out_path}")


# ============================================================
# CLI 진입점
# ============================================================

def main() -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    parser = argparse.ArgumentParser(description="Synthetic vs Real comparison dashboard")
    parser.add_argument("--n-runs",   type=int, default=3,    help="비교할 synthetic run 수")
    parser.add_argument("--save-dir", type=str, default="outputs/simulation_comparison")
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()

    from simulator.engine import SyntheticConfig, generate_run
    from simulator.physics import BEARING_SPECS

    save_dir = Path(args.save_dir)
    real_dir = _find_real_dir()
    if real_dir:
        print(f"[Real] Loading signals from {real_dir}")
        real_signals = _load_real_signals(real_dir, n_files=20)
        print(f"  Loaded {len(real_signals)} real signals")
    else:
        print("[Real] No real TDMS found — synthetic-only plots")
        real_signals = []

    cfg = SyntheticConfig(seed=args.seed)
    fault_types = ["BPFO", "BPFI", "BSF"]
    master_rng = np.random.default_rng(args.seed)

    for i in range(args.n_runs):
        run_seed = int(master_rng.integers(0, 2 ** 31))
        run_rng  = np.random.default_rng(run_seed)
        life_sec = float(run_rng.uniform(cfg.total_life_seconds_min, cfg.total_life_seconds_max))
        run_cfg  = SyntheticConfig(
            fault_type=fault_types[i % len(fault_types)],
            seed=run_seed,
            total_life_seconds=life_sec,
        )
        run = generate_run(run_cfg, rng=run_rng)
        plot_comparison(run, real_signals, save_dir=save_dir, run_idx=i)

    print(f"\n[Done] Comparison plots saved to {save_dir}")


if __name__ == "__main__":
    main()