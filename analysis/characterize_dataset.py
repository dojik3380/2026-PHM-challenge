"""
analysis/characterize_dataset.py

실제 KSPHM TDMS 훈련 데이터(Train1~Train4)를 분석하여
Synthetic Bearing Degradation Simulator에 사용될 물리 파라미터 분포를 추출한다.

Usage:
    python analysis/characterize_dataset.py --data-dir data/Train
    python analysis/characterize_dataset.py --data-dir data/Train --test-mode
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import butter, filtfilt, hilbert, stft as scipy_stft, welch
from scipy.stats import kurtosis as scipy_kurtosis

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_loader import discover_cases, load_tdms_channels
from features.rpm_estimator import estimate_rpm_from_signal
from simulator.physics import BEARING_SPECS

# ── 상수 ──────────────────────────────────────────────────────────────────
FS = 25_600
CHANNELS = ["CH1", "CH2", "CH3", "CH4"]
OUTPUT_DIR = Path("outputs/real_characterization")
JSON_OUT = Path("outputs/physics_parameters.json")

BASE_RPM = 1000.0


# ═══════════════════════════════════════════════════════════════════════════
# 유틸리티
# ═══════════════════════════════════════════════════════════════════════════

def _save_fig(fig: plt.Figure, filename: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=100, bbox_inches="tight")
    plt.close(fig)


def _safe_mean(lst: list) -> float:
    return float(np.mean(lst)) if lst else 0.0


def _safe_std(lst: list) -> float:
    return float(np.std(lst)) if lst else 0.0


def _load_first_channel(tdms_path: Path) -> np.ndarray | None:
    """TDMS 파일에서 첫 번째 유효 채널을 float32로 반환한다."""
    try:
        data = load_tdms_channels(tdms_path)
    except Exception:
        return None
    for ch in CHANNELS:
        # numpy array는 `or` 연산이 불가하므로 명시적 None 체크
        sig = data.get(ch)
        if sig is None:
            sig = data.get(ch.lower())
        if sig is not None and len(sig) > 64:
            return np.asarray(sig, dtype=np.float32)
    # 채널명이 표준과 다를 경우 첫 번째 채널 반환
    for sig in data.values():
        arr = np.asarray(sig, dtype=np.float32)
        if arr.ndim == 1 and len(arr) > 64:
            return arr
    return None


def _get_fault_freqs(rpm: float) -> dict[str, float]:
    """실제 RPM에 맞게 스케일링된 결함 주파수 딕셔너리를 반환한다."""
    return {
        k: v * (rpm / BASE_RPM)
        for k, v in BEARING_SPECS["fault_frequencies"].items()
    }


# ═══════════════════════════════════════════════════════════════════════════
# 신호 분석 함수
# ═══════════════════════════════════════════════════════════════════════════

def _compute_psd(signal: np.ndarray, fs: int = FS, nperseg: int = 4096):
    """Welch PSD 계산. (f, Pxx) 반환."""
    nperseg = min(nperseg, len(signal))
    f, Pxx = welch(signal, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    return f, Pxx


def estimate_resonance_fn(
    signal: np.ndarray, fs: int = FS, search_band: tuple = (3_000, 15_000)
) -> tuple[float, np.ndarray, np.ndarray]:
    """Welch PSD 피크를 통해 공진 주파수 fn 을 추정한다."""
    f, Pxx = _compute_psd(signal, fs)
    lo, hi = search_band
    mask = (f >= lo) & (f <= hi)
    if mask.any():
        fn = float(f[mask][np.argmax(Pxx[mask])])
    else:
        fn = 4_000.0
    return fn, f, Pxx


def estimate_damping_zeta(
    signal: np.ndarray, fn: float, fs: int = FS
) -> float:
    """
    Bandpass 필터링 + Hilbert 변환으로 감쇠비 ζ를 근사 추정한다.

    fn 근처를 bandpass 필터링 → envelope = |Hilbert(filtered)| 계산 →
    a·exp(-b·t) 피팅 → ζ = b / (2π·fn).

    Note: 연속 회전 진동 신호에서 개별 임펄스 decay를 정확히 분리할 수 없으므로
    통계적 근사치를 반환한다.
    """
    bw = max(200.0, fn * 0.15)
    lo_hz = max(10.0, fn - bw)
    hi_hz = min(fs / 2.0 - 1.0, fn + bw)

    try:
        b_coef, a_coef = butter(4, [lo_hz / (fs / 2), hi_hz / (fs / 2)], btype="band")
        filtered = filtfilt(b_coef, a_coef, signal.astype(np.float64))
    except Exception:
        return 0.01

    env = np.abs(hilbert(filtered))

    # 신호 초반 20%에서 decay 피팅
    fit_len = max(128, len(env) // 5)
    t = np.arange(fit_len) / fs
    env_fit = env[:fit_len]

    peak = env_fit.max()
    if peak < 1e-10:
        return 0.01

    env_norm = env_fit / peak

    try:
        popt, _ = curve_fit(
            lambda t, a, b: a * np.exp(-b * t),
            t,
            env_norm,
            p0=[1.0, 2 * np.pi * fn * 0.01],
            bounds=([0.0, 0.0], [2.0, 2 * np.pi * fn]),
            maxfev=2000,
        )
        b_fit = float(popt[1])
        zeta = b_fit / (2 * np.pi * fn)
    except Exception:
        return 0.01

    return float(np.clip(zeta, 1e-4, 0.5))


def compute_envelope_spectrum(
    signal: np.ndarray, fs: int = FS, fn: float = 4_000.0
) -> tuple[np.ndarray, np.ndarray]:
    """Bandpass + Hilbert 변환으로 Envelope Spectrum(FFT of |hilbert|)을 계산한다."""
    bw = max(500.0, fn * 0.2)
    lo_hz = max(10.0, fn - bw)
    hi_hz = min(fs / 2.0 - 1.0, fn + bw)

    try:
        b_coef, a_coef = butter(4, [lo_hz / (fs / 2), hi_hz / (fs / 2)], btype="band")
        filtered = filtfilt(b_coef, a_coef, signal.astype(np.float64))
    except Exception:
        filtered = signal.astype(np.float64)

    env = np.abs(hilbert(filtered))
    env -= env.mean()
    env_fft = np.abs(np.fft.rfft(env))
    f_env = np.fft.rfftfreq(len(env), d=1 / fs)
    return env_fft, f_env


def compute_harmonic_metrics(
    env_fft: np.ndarray, f_env: np.ndarray, bpfo: float, n_harmonics: int = 5
) -> tuple[float, float]:
    """BPFO 배수에서의 가시성(visibility)과 harmonic decay(2x/1x)를 측정한다."""
    bg_noise = float(np.median(env_fft)) + 1e-8

    mags = []
    for h in range(1, n_harmonics + 1):
        freq = bpfo * h
        if freq >= f_env[-1]:
            break
        idx = int(np.argmin(np.abs(f_env - freq)))
        mags.append(float(env_fft[idx]))

    if not mags:
        return 0.0, 0.0

    visibility = mags[0] / bg_noise
    decay = mags[1] / (mags[0] + 1e-8) if len(mags) > 1 else 0.0
    return visibility, decay


def estimate_noise_snr(
    signal: np.ndarray, fs: int = FS, fn: float = 4_000.0
) -> tuple[float, float]:
    """공진 대역 피크 vs 전체 배경 노이즈로 SNR을 추정한다."""
    f, Pxx = _compute_psd(signal, fs)
    bw = max(500.0, fn * 0.2)
    res_mask = (f >= fn - bw) & (f <= fn + bw)
    noise_mask = f > 100

    if not (res_mask.any() and noise_mask.any()):
        return 0.0, 0.0

    noise_floor = float(np.mean(Pxx[noise_mask]))
    peak_pwr = float(np.max(Pxx[res_mask]))
    snr = 10 * np.log10(peak_pwr / (noise_floor + 1e-30))
    return snr, noise_floor


def estimate_rpm_stats(tdms_files: list, sample_indices: list) -> dict:
    """샘플링된 파일들에서 RPM 분포 통계를 추출한다."""
    rpms = []
    for idx in sample_indices:
        sig = _load_first_channel(tdms_files[idx])
        if sig is not None:
            rpm = estimate_rpm_from_signal(sig.astype(np.float64), fs=FS)
            if rpm > 100:
                rpms.append(rpm)

    if not rpms:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0,
                "n_high": 0, "n_low": 0, "switching_rate": 0.0}

    arr = np.array(rpms)
    median = float(np.median(arr))
    high_mask = arr > median
    # switching_rate: 연속된 샘플 간 regime 전환 비율
    switching_rate = float(np.mean(np.abs(np.diff(high_mask.astype(int))))) if len(rpms) > 1 else 0.0

    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n_high": int(np.sum(high_mask)),
        "n_low": int(np.sum(~high_mask)),
        "switching_rate": switching_rate,
    }


def analyze_rms_kurtosis_trend(
    tdms_files: list, n_files: int
) -> tuple[list, list, list]:
    """전체 생애주기에 걸쳐 다운샘플링된 RMS/Kurtosis 추세를 계산한다."""
    step = max(1, n_files // 60)
    trend_indices = list(range(0, n_files, step))
    rms_vals, kurt_vals = [], []

    for i in trend_indices:
        sig = _load_first_channel(tdms_files[i])
        if sig is not None:
            rms_vals.append(float(np.sqrt(np.mean(sig ** 2))))
            sig_c = sig - sig.mean()
            kurt_vals.append(float(scipy_kurtosis(sig_c)))
        else:
            rms_vals.append(float("nan"))
            kurt_vals.append(float("nan"))

    return trend_indices, rms_vals, kurt_vals


def fit_degradation_trend(rms_vals: list) -> dict:
    """RMS 추세에서 exponential burst 시점을 추정한다."""
    arr = np.array([v for v in rms_vals if np.isfinite(v)])
    n = len(arr)
    if n < 6:
        return {"type": "unknown", "early_to_late_ratio": 1.0, "burst_point_normalized": -1.0}

    norm = arr / (arr.mean() + 1e-10)
    early_mean = float(norm[: n // 2].mean())
    late_mean = float(norm[-max(1, n // 5) :].mean())
    ratio = late_mean / (early_mean + 1e-8)

    burst_norm = -1.0
    if ratio > 3.0:
        diff = np.diff(norm)
        thresh = diff.mean() + 2 * diff.std()
        cands = np.where(diff > thresh)[0]
        burst_idx = int(cands[0]) if len(cands) > 0 else n - max(1, n // 5)
        burst_norm = float(burst_idx / n)

    return {
        "type": "exponential" if ratio > 3.0 else "linear",
        "early_to_late_ratio": float(ratio),
        "burst_point_normalized": burst_norm,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 시각화 함수
# ═══════════════════════════════════════════════════════════════════════════

def _plot_rms_trend(case_name: str, indices: list, vals: list) -> None:
    valid = [(i, v) for i, v in zip(indices, vals) if np.isfinite(v)]
    if not valid:
        return
    xi, yi = zip(*valid)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(xi, yi, "b-o", markersize=3, linewidth=0.8)
    ax.set_xlabel("File Index")
    ax.set_ylabel("RMS (g)")
    ax.set_title(f"{case_name} — RMS Degradation Trend")
    ax.grid(True, alpha=0.3)
    _save_fig(fig, f"rms_trend_{case_name}.png")


def _plot_kurtosis_trend(case_name: str, indices: list, vals: list) -> None:
    valid = [(i, v) for i, v in zip(indices, vals) if np.isfinite(v)]
    if not valid:
        return
    xi, yi = zip(*valid)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(xi, yi, "r-o", markersize=3, linewidth=0.8)
    ax.axhline(3.0, color="gray", linestyle="--", alpha=0.5, label="Gaussian baseline (K=3)")
    ax.set_xlabel("File Index")
    ax.set_ylabel("Kurtosis")
    ax.set_title(f"{case_name} — Kurtosis Trend")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _save_fig(fig, f"kurtosis_trend_{case_name}.png")


def _plot_fft(case_name: str, f: np.ndarray, Pxx: np.ndarray, fn: float) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.semilogy(f, Pxx, "b-", linewidth=0.6)
    ax.axvline(fn, color="r", linestyle="--", linewidth=1.5, label=f"fn = {fn:.0f} Hz")
    bw = max(200.0, fn * 0.15)
    ax.axvspan(fn - bw, fn + bw, alpha=0.12, color="red", label="Bandpass region")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (g²/Hz)")
    ax.set_title(f"{case_name} — Failure Stage PSD")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _save_fig(fig, f"fft_{case_name}_fail.png")


def _plot_envelope(
    case_name: str, f_env: np.ndarray, env_fft: np.ndarray, fault_freqs: dict
) -> None:
    # 저주파 대역만 표시 (최대 1500 Hz 또는 전체의 10%)
    cutoff_hz = 1500.0
    cutoff_idx = int(np.searchsorted(f_env, cutoff_hz))
    cutoff_idx = max(cutoff_idx, len(f_env) // 20)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(f_env[:cutoff_idx], env_fft[:cutoff_idx], "g-", linewidth=0.7)

    colors = {"BPFO": "r", "BPFI": "b", "BSF": "orange", "Cage": "purple"}
    for fname, freq in fault_freqs.items():
        c = colors.get(fname, "gray")
        for h in range(1, 5):
            line_freq = freq * h
            if line_freq >= f_env[cutoff_idx]:
                break
            label = f"{fname}" if h == 1 else None
            ax.axvline(line_freq, color=c, linestyle="--", alpha=0.6,
                       linewidth=0.9, label=label)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"{case_name} — Failure Envelope Spectrum")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    _save_fig(fig, f"envelope_{case_name}_fail.png")


def _plot_spectrogram(case_name: str, signal: np.ndarray, fs: int = FS) -> None:
    nperseg = 512
    f_spec, t_spec, Zxx = scipy_stft(
        signal.astype(np.float64), fs=fs, nperseg=nperseg, noverlap=nperseg * 3 // 4
    )
    mag_db = 20 * np.log10(np.abs(Zxx) + 1e-10)

    # 0 ~ 8000 Hz 대역 표시
    freq_limit_hz = 8000.0
    n_freq = int(np.searchsorted(f_spec, freq_limit_hz)) + 1

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.pcolormesh(
        t_spec, f_spec[:n_freq], mag_db[:n_freq], shading="gouraud", cmap="viridis"
    )
    plt.colorbar(im, ax=ax, label="Magnitude (dB)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"{case_name} — STFT Spectrogram (Failure Stage, 0–8 kHz)")
    _save_fig(fig, f"spectrogram_{case_name}.png")


def _plot_resonance_analysis(
    case_name: str,
    f: np.ndarray,
    Pxx: np.ndarray,
    fn: float,
    fn_history: list,
    zeta_history: list,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 왼쪽: PSD + 공진 대역 마킹
    ax = axes[0]
    ax.semilogy(f, Pxx, "b-", linewidth=0.7)
    ax.axvline(fn, color="r", linewidth=1.8, label=f"fn = {fn:.0f} Hz")
    bw = max(200.0, fn * 0.15)
    ax.axvspan(fn - bw, fn + bw, alpha=0.15, color="red", label="±15% bandwidth")
    ax.set_xlim(0, min(20_000, float(f[-1])))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (g²/Hz)")
    ax.set_title("PSD with Resonance Band")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 오른쪽: 생애주기별 fn / ζ 변화
    ax2 = axes[1]
    x = list(range(len(fn_history)))
    if fn_history:
        ax2.plot(x, fn_history, "bo-", markersize=5, label="fn (Hz)")
        ax2.set_ylabel("fn (Hz)", color="b")
        ax2.tick_params(axis="y", labelcolor="b")

    if zeta_history:
        ax2r = ax2.twinx()
        ax2r.plot(x, zeta_history, "rs--", markersize=5, label="ζ")
        ax2r.set_ylabel("Damping Ratio ζ", color="r")
        ax2r.tick_params(axis="y", labelcolor="r")

    ax2.set_xlabel("Sample Index (Healthy → Failure)")
    ax2.set_title("fn & ζ Across Lifecycle")
    ax2.grid(True, alpha=0.3)

    _save_fig(fig, f"resonance_analysis_{case_name}.png")


# ═══════════════════════════════════════════════════════════════════════════
# 케이스 분석 메인 함수
# ═══════════════════════════════════════════════════════════════════════════

def analyze_case(case_name: str, tdms_dir: Path, test_mode: bool = False) -> dict | None:
    """하나의 Train 케이스에 대해 초기/중기/말기 분석을 수행한다."""
    tdms_files = sorted(tdms_dir.glob("*.tdms"))
    if not tdms_files:
        print(f"  [{case_name}] TDMS 파일 없음.")
        return None

    n_files = len(tdms_files)
    print(f"  [{case_name}] {n_files}개 TDMS 파일 발견.")

    # ── 1. 샘플링 인덱스 결정 ───────────────────────────────────────
    if test_mode:
        sample_indices = sorted({0, n_files // 2, n_files - 1})
    else:
        healthy = list(range(0, min(5, n_files)))
        mid_s = max(0, n_files // 2 - 2)
        mid = list(range(mid_s, min(mid_s + 5, n_files)))
        fail_s = max(0, n_files - 5)
        fail = list(range(fail_s, n_files))
        sample_indices = sorted(set(healthy + mid + fail))

    # ── 2. RPM 통계 ─────────────────────────────────────────────────
    rpm_stats = estimate_rpm_stats(tdms_files, sample_indices)
    ref_rpm = max(rpm_stats["mean"], 100.0)

    # ── 3. 전체 RMS/Kurtosis 추세 ───────────────────────────────────
    trend_idx, rms_trend, kurt_trend = analyze_rms_kurtosis_trend(tdms_files, n_files)
    rms_fit = fit_degradation_trend(rms_trend)
    valid_kurts = [k for k in kurt_trend if np.isfinite(k)]
    kurtosis_early_rise = (
        bool(
            len(valid_kurts) >= 4
            and float(np.mean(valid_kurts[: max(1, len(valid_kurts) // 4)])) > 3.5
        )
    )

    # ── 4. 샘플 파일별 상세 분석 ────────────────────────────────────
    fn_values, zeta_values, snr_values = [], [], []
    bpfo_vis_values, harmonic_decay_values = [], []
    healthy_noise_floor = None

    healthy_cut = sample_indices[len(sample_indices) // 3] if len(sample_indices) >= 3 else sample_indices[-1]

    for idx in sample_indices:
        sig = _load_first_channel(tdms_files[idx])
        if sig is None:
            continue

        # RPM (개별 추정, fallback to mean)
        rpm = estimate_rpm_from_signal(sig.astype(np.float64), fs=FS)
        if rpm < 100:
            rpm = ref_rpm

        # 공진 주파수
        fn, f_psd, Pxx = estimate_resonance_fn(sig, FS)
        fn_values.append(fn)

        # 감쇠비 (Bandpass + Hilbert 근사)
        zeta = estimate_damping_zeta(sig, fn, FS)
        zeta_values.append(zeta)

        # SNR & Noise floor
        snr, noise_floor = estimate_noise_snr(sig, FS, fn)
        snr_values.append(snr)
        if idx <= healthy_cut and healthy_noise_floor is None:
            healthy_noise_floor = noise_floor

        # Envelope Harmonics
        fault_freqs = _get_fault_freqs(rpm)
        env_fft, f_env = compute_envelope_spectrum(sig, FS, fn)
        vis, h_decay = compute_harmonic_metrics(env_fft, f_env, fault_freqs["BPFO"])
        bpfo_vis_values.append(vis)
        harmonic_decay_values.append(h_decay)

    # ── 5. 시각화 ───────────────────────────────────────────────────
    _plot_rms_trend(case_name, trend_idx, rms_trend)
    _plot_kurtosis_trend(case_name, trend_idx, kurt_trend)

    # Failure 시점 (마지막 샘플) 시각화
    fail_sig = _load_first_channel(tdms_files[sample_indices[-1]])
    if fail_sig is not None:
        fail_rpm = estimate_rpm_from_signal(fail_sig.astype(np.float64), fs=FS)
        if fail_rpm < 100:
            fail_rpm = ref_rpm
        fail_fn = fn_values[-1] if fn_values else 4_000.0

        f_psd, Pxx_fail = _compute_psd(fail_sig)
        _plot_fft(case_name, f_psd, Pxx_fail, fail_fn)

        env_fft, f_env = compute_envelope_spectrum(fail_sig, FS, fail_fn)
        _plot_envelope(case_name, f_env, env_fft, _get_fault_freqs(fail_rpm))

        _plot_spectrogram(case_name, fail_sig)
        _plot_resonance_analysis(case_name, f_psd, Pxx_fail, fail_fn, fn_values, zeta_values)

    # ── 6. 결과 반환 ────────────────────────────────────────────────
    return {
        "case_name": case_name,
        "rpm": rpm_stats,
        "fn_mean": _safe_mean(fn_values),
        "fn_std": _safe_std(fn_values),
        "damping_mean": _safe_mean(zeta_values),
        "damping_std": _safe_std(zeta_values),
        "bpfo_visibility_mean": _safe_mean(bpfo_vis_values),
        "harmonic_decay_mean": _safe_mean(harmonic_decay_values),
        "snr_mean": _safe_mean(snr_values),
        "healthy_noise_floor": float(healthy_noise_floor) if healthy_noise_floor else 0.0,
        "rms_trend": rms_fit,
        "kurtosis_early_rise": kurtosis_early_rise,
        "n_files": n_files,
        "n_samples_analyzed": len(sample_indices),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 집계
# ═══════════════════════════════════════════════════════════════════════════

def aggregate_results(all_results: list[dict]) -> dict:
    """케이스별 결과를 집계하여 Simulator용 파라미터 분포를 산출한다."""

    def _collect(key): return [r[key] for r in all_results if r.get(key, 0) > 0]

    rpm_mins = [r["rpm"]["min"] for r in all_results if r["rpm"]["min"] > 0]
    rpm_maxs = [r["rpm"]["max"] for r in all_results if r["rpm"]["max"] > 0]
    fns = _collect("fn_mean")
    fn_stds = [r["fn_std"] for r in all_results]
    dampings = _collect("damping_mean")
    damp_stds = [r["damping_std"] for r in all_results]
    snrs = [r["snr_mean"] for r in all_results]
    bpfo_vis = [r["bpfo_visibility_mean"] for r in all_results]
    h_decay = [r["harmonic_decay_mean"] for r in all_results]

    return {
        "rpm_range": [_safe_mean(rpm_mins), _safe_mean(rpm_maxs)],
        "rpm_mean": _safe_mean([r["rpm"]["mean"] for r in all_results]),
        "rpm_std": _safe_mean([r["rpm"]["std"] for r in all_results]),
        "fn_primary_mean": _safe_mean(fns),
        "fn_primary_std": _safe_mean(fn_stds),
        "damping_mean": _safe_mean(dampings),
        "damping_std": _safe_mean(damp_stds),
        "bpfo_visibility": _safe_mean(bpfo_vis),
        "harmonic_decay": _safe_mean(h_decay),
        "snr_mean": _safe_mean(snrs),
        "rms_growth_type": "exponential",
        "kurtosis_early_rise": bool(any(r.get("kurtosis_early_rise", False) for r in all_results)),
        "per_case": all_results,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 진입점
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="TDMS 데이터 물리 파라미터 특성화")
    parser.add_argument("--data-dir", type=str, default="data/Train",
                        help="훈련 데이터 루트 디렉토리 (기본: data/Train)")
    parser.add_argument("--test-mode", action="store_true",
                        help="빠른 검증 모드: Train1에서 3 파일만 샘플링")
    args = parser.parse_args()

    root_dir = Path(args.data_dir)
    if not root_dir.exists():
        print(f"[ERROR] 데이터 디렉토리를 찾을 수 없음: {root_dir}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)

    cases = discover_cases(root_dir)
    if not cases:
        print(f"[ERROR] {root_dir} 에서 케이스(Operation CSV + TDMS 폴더)를 찾을 수 없음.")
        return

    print(f"분석 대상 케이스: {len(cases)}개")

    all_results = []
    for case_name, _op_csv, vib_dir in cases:
        print(f"\n[{case_name}] 분석 시작...")
        try:
            res = analyze_case(case_name, vib_dir, test_mode=args.test_mode)
            if res:
                all_results.append(res)
                print(
                    f"  fn={res['fn_mean']:.0f}Hz  ζ={res['damping_mean']:.4f}"
                    f"  RPM={res['rpm']['mean']:.0f}  SNR={res['snr_mean']:.1f}dB"
                )
        except Exception as exc:
            import traceback
            print(f"  [{case_name}] 오류: {exc}")
            traceback.print_exc()

        if args.test_mode and len(all_results) >= 1:
            print("\n[test-mode] Train1 분석 완료, 종료.")
            break

    if not all_results:
        print("추출된 결과 없음.")
        return

    agg = aggregate_results(all_results)

    with open(JSON_OUT, "w", encoding="utf-8") as fp:
        json.dump(agg, fp, indent=4, ensure_ascii=False)

    print(f"\n분석 완료.")
    print(f"  파라미터 저장: {JSON_OUT}")
    print(f"  그래프 저장:   {OUTPUT_DIR}/")

    summary = {k: v for k, v in agg.items() if k != "per_case"}
    print("\n집계 파라미터:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
