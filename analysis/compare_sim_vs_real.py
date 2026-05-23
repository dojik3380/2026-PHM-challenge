"""합성(Synthetic) vs 실측(Real TDMS) STFT 특징 분포 비교.

목적:
    동일한 STFT 파이프라인(vibration_stft_timestep)으로
    실측 TDMS와 pretrain_data.npz를 처리하고 분포 차이를 정량화한다.

출력:
    outputs/sim_vs_real/
    ├── comparison_report.json
    ├── spectrum_overlay.png       (평균 스펙트럼 곡선 overlay)
    ├── magnitude_histogram.png    (전체 magnitude 히스토그램)
    └── per_band_energy.png        (주파수 대역별 에너지 비교)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import SAMPLING_RATE, STFT_FREQ_BINS, STFT_NPERSEG
from data_loader import discover_cases
from features.vibration import vibration_stft_timestep
from features.rpm_estimator import extract_auxiliary_vector
from data_loader import load_tdms_channels


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "sim_vs_real"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_real_features(
    n_files_per_case: int = 30,
    train_dir: Path = PROJECT_ROOT / "data" / "Train",
) -> tuple[np.ndarray, np.ndarray, dict]:
    """모든 Train 케이스에서 균등 샘플링한 TDMS의 STFT 특징을 추출한다.

    Returns:
        vib_features: (N_samples, 4, 1026)
        aux_features: (N_samples, 2)  [RPM, RMS]
        info: dict
    """
    cases = discover_cases(train_dir)
    print(f"[Real] Found {len(cases)} cases in {train_dir}")

    vib_list = []
    aux_list = []
    case_info = []

    for case_name, _csv, vib_dir in cases:
        tdms_files = sorted(Path(vib_dir).glob("*.tdms"))
        n_total = len(tdms_files)
        if n_total == 0:
            continue
        # 균등 샘플링: healthy ~ failure 골고루
        idxs = np.linspace(0, n_total - 1, num=min(n_files_per_case, n_total), dtype=int)
        print(f"  [{case_name}] {n_total} files, sampling {len(idxs)} timesteps")

        for i in idxs:
            tdms_path = tdms_files[i]
            try:
                vib_feat = vibration_stft_timestep(tdms_path)  # (4, 1026)
                ch_data = load_tdms_channels(tdms_path)
                aux_feat = extract_auxiliary_vector(ch_data)  # (2,) [RPM, RMS]
                vib_list.append(vib_feat)
                aux_list.append(aux_feat)
                case_info.append({"case": case_name, "file_idx": int(i), "progress": float(i / max(1, n_total - 1))})
            except Exception as exc:
                print(f"    [WARN] Failed {tdms_path.name}: {exc}")

    vib_arr = np.stack(vib_list, axis=0).astype(np.float32)
    aux_arr = np.stack(aux_list, axis=0).astype(np.float32)
    info = {"n_samples": len(vib_arr), "cases": [c["case"] for c in case_info]}
    return vib_arr, aux_arr, info


def load_synthetic_features(
    npz_path: Path = PROJECT_ROOT / "outputs" / "synthetic" / "pretrain_data.npz",
    n_subsample: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    """pretrain_data.npz에서 window 차원 평탄화 후 부분 샘플링.

    Returns:
        vib_features: (n_subsample, 4, 1026)
        aux_features: (n_subsample, 2)
    """
    data = np.load(npz_path)
    xvib = data["X_vib"]  # (N, W, 4, 1026)
    xaux = data["X_aux"]  # (N, W, 2)

    # window 별 timestep을 하나씩 뽑기: window 중간 시점 1개만
    mid_idx = xvib.shape[1] // 2
    vib_flat = xvib[:, mid_idx, :, :]  # (N, 4, 1026)
    aux_flat = xaux[:, mid_idx, :]     # (N, 2)

    # 부분 샘플링 (메모리 절약)
    rng = np.random.default_rng(42)
    if len(vib_flat) > n_subsample:
        sel = rng.choice(len(vib_flat), size=n_subsample, replace=False)
        vib_flat = vib_flat[sel]
        aux_flat = aux_flat[sel]

    return vib_flat.astype(np.float32), aux_flat.astype(np.float32)


def summary_stats(arr: np.ndarray, name: str) -> dict:
    """기본 분포 통계."""
    flat = arr.flatten()
    return {
        "name": name,
        "shape": list(arr.shape),
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "q05": float(np.quantile(flat, 0.05)),
        "q50": float(np.quantile(flat, 0.50)),
        "q95": float(np.quantile(flat, 0.95)),
    }


def per_band_energy(vib: np.ndarray, fs: int = SAMPLING_RATE) -> dict[str, float]:
    """주파수 대역별 평균 에너지. vib shape: (N, 4, 1026) — 앞 513은 mean, 뒤 513은 std."""
    mean_half = vib[..., :STFT_FREQ_BINS]  # (N, 4, 513)
    freqs = np.fft.rfftfreq(STFT_NPERSEG, d=1 / fs)  # 513 bins, 0~12800 Hz

    bands = {
        "0-500Hz":     (0, 500),
        "500-2kHz":    (500, 2000),
        "2-5kHz":      (2000, 5000),
        "5-8kHz":      (5000, 8000),
        "8-12.8kHz":   (8000, 12800),
    }
    result = {}
    for name, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs < hi)
        result[name] = float(mean_half[..., mask].mean())
    return result


def plot_spectrum_overlay(real_vib: np.ndarray, sim_vib: np.ndarray, save_path: Path) -> None:
    """채널 평균 STFT 평균 스펙트럼 (앞 513 bin) overlay."""
    fs = SAMPLING_RATE
    freqs = np.fft.rfftfreq(STFT_NPERSEG, d=1 / fs)  # (513,)
    real_mean_spec = real_vib[..., :STFT_FREQ_BINS].mean(axis=(0, 1))  # (513,)
    sim_mean_spec = sim_vib[..., :STFT_FREQ_BINS].mean(axis=(0, 1))    # (513,)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, scale in zip(axes, ["linear", "log"]):
        ax.plot(freqs, real_mean_spec, label=f"Real (n={len(real_vib)})", color="blue", lw=1.2)
        ax.plot(freqs, sim_mean_spec, label=f"Synthetic (n={len(sim_vib)})", color="red", lw=1.2, alpha=0.7)
        ax.set_yscale(scale)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel(f"STFT mean magnitude ({scale})")
        ax.set_title(f"Mean Spectrum Overlay ({scale})")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_magnitude_histogram(real_vib: np.ndarray, sim_vib: np.ndarray, save_path: Path) -> None:
    """전체 magnitude 분포 히스토그램 (log scale)."""
    real_flat = real_vib.flatten()
    sim_flat = sim_vib.flatten()
    # 0 이하 제거 (log 위해)
    real_flat = real_flat[real_flat > 1e-12]
    sim_flat = sim_flat[sim_flat > 1e-12]

    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.logspace(np.log10(min(real_flat.min(), sim_flat.min())),
                       np.log10(max(real_flat.max(), sim_flat.max())), 80)
    ax.hist(real_flat, bins=bins, alpha=0.5, label=f"Real (n={len(real_flat):,})", color="blue", density=True)
    ax.hist(sim_flat, bins=bins, alpha=0.5, label=f"Synthetic (n={len(sim_flat):,})", color="red", density=True)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("STFT magnitude (log)")
    ax.set_ylabel("density")
    ax.set_title("STFT Magnitude Distribution Overlay")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_per_band_energy(real_bands: dict, sim_bands: dict, save_path: Path) -> None:
    """주파수 대역별 에너지 막대 그래프."""
    bands = list(real_bands.keys())
    real_vals = [real_bands[b] for b in bands]
    sim_vals = [sim_bands[b] for b in bands]

    x = np.arange(len(bands))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, real_vals, width, label="Real", color="blue", alpha=0.8)
    ax.bar(x + width / 2, sim_vals, width, label="Synthetic", color="red", alpha=0.8)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(bands, rotation=20)
    ax.set_ylabel("Mean STFT magnitude (mean-half) [log]")
    ax.set_title("Per-band Energy Comparison")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    for xi, (rv, sv) in enumerate(zip(real_vals, sim_vals)):
        ratio = (sv / rv) if rv > 0 else float("inf")
        ax.text(xi, max(rv, sv) * 1.3, f"sim/real={ratio:.2f}x", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print("=" * 60)
    print("  Sim-vs-Real STFT Distribution Comparison")
    print("=" * 60)

    print("\n[Step 1] Real TDMS feature extraction...")
    real_vib, real_aux, real_info = extract_real_features(n_files_per_case=30)
    print(f"  real_vib: {real_vib.shape} | real_aux: {real_aux.shape}")

    print("\n[Step 2] Synthetic feature loading...")
    sim_vib, sim_aux = load_synthetic_features(n_subsample=2000)
    print(f"  sim_vib : {sim_vib.shape} | sim_aux : {sim_aux.shape}")

    print("\n[Step 3] Summary statistics...")
    stats = {
        "real_vib": summary_stats(real_vib, "real_vib"),
        "sim_vib":  summary_stats(sim_vib, "sim_vib"),
        "real_aux_RPM": summary_stats(real_aux[:, 0], "real_RPM"),
        "sim_aux_RPM":  summary_stats(sim_aux[:, 0], "sim_RPM"),
        "real_aux_RMS": summary_stats(real_aux[:, 1], "real_RMS"),
        "sim_aux_RMS":  summary_stats(sim_aux[:, 1], "sim_RMS"),
    }

    print("\n[Step 4] Per-band energy...")
    real_bands = per_band_energy(real_vib)
    sim_bands = per_band_energy(sim_vib)
    print("  Real bands:", real_bands)
    print("  Sim  bands:", sim_bands)

    print("\n[Step 5] Plots...")
    plot_spectrum_overlay(real_vib, sim_vib, OUTPUT_DIR / "spectrum_overlay.png")
    plot_magnitude_histogram(real_vib, sim_vib, OUTPUT_DIR / "magnitude_histogram.png")
    plot_per_band_energy(real_bands, sim_bands, OUTPUT_DIR / "per_band_energy.png")
    print(f"  [OK] saved to {OUTPUT_DIR}")

    # 보고서 저장
    report = {
        "n_real_samples": int(real_vib.shape[0]),
        "n_sim_samples": int(sim_vib.shape[0]),
        "stats": stats,
        "per_band_energy": {
            "real": real_bands,
            "sim": sim_bands,
            "ratio_sim_over_real": {k: (sim_bands[k] / real_bands[k]) if real_bands[k] > 0 else None
                                    for k in real_bands},
        },
        "real_cases": real_info["cases"],
    }
    with open(OUTPUT_DIR / "comparison_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  [OK] report -> {OUTPUT_DIR / 'comparison_report.json'}")

    print("\n[Done]")


if __name__ == "__main__":
    main()