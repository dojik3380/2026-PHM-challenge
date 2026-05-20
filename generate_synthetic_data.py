"""
generate_synthetic_data.py

Physics-based Synthetic Bearing Degradation Data 생성 메인 스크립트.

사용법:
    # 기본 실행 (50개 Run 생성)
    python generate_synthetic_data.py

    # 커스텀 설정
    python generate_synthetic_data.py --n-runs 100 --seq-len 120 --seed 42

출력:
    outputs/synthetic/
    ├── pretrain_data.npz        ← 전이학습용 배열 (X_vib, y_rul, metadata)
    ├── pretrain_config.json     ← 생성에 사용한 설정 저장
    └── visualization/
        ├── run_000_dashboard.png
        ├── run_000_rul_trajectory.png
        ├── single_impulse_h.png
        └── ...
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.signal import stft as scipy_stft

from simulator.engine import SyntheticConfig, generate_dataset
from simulator.physics import single_impulse_response, sample_fn_zeta, BEARING_SPECS
from simulator.visualize import (
    plot_single_impulse,
    plot_degradation_dashboard,
    plot_rul_trajectory,
)

# ============================================================
# 경로 설정
# ============================================================
PROJECT_ROOT = Path(__file__).parent
OUTPUT_DIR   = PROJECT_ROOT / "outputs" / "synthetic"
VIZ_DIR      = OUTPUT_DIR / "visualization"


# ============================================================
# STFT 특징 추출 (기존 model.py 파이프라인과 동일한 규격)
# ============================================================

def extract_stft_features(
    signal: np.ndarray,
    fs: int = 25_600,
    nperseg: int = 1024,
    noverlap: int = 512,
) -> np.ndarray:
    """
    1D 진동 신호에서 STFT magnitude를 추출한다.
    출력 shape: (freq_bins,) = (513,)
    기존 train.py의 _compute_stft_features()와 동일한 규격.
    """
    _, _, Zxx = scipy_stft(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
    # 평균 magnitude (시간 축 평균)
    mag = np.mean(np.abs(Zxx), axis=1).astype(np.float32)
    return mag


def extract_handcrafted(signal: np.ndarray) -> np.ndarray:
    """RMS, Kurtosis, Crest Factor, Envelope Energy, Band Energy 5개 특징."""
    rms = np.sqrt(np.mean(signal ** 2))
    signal_c = signal - signal.mean()
    std = signal_c.std() + 1e-10
    kurtosis = np.mean((signal_c / std) ** 4)
    peak = np.max(np.abs(signal))
    crest = peak / (rms + 1e-10)
    from scipy.signal import hilbert
    env = np.abs(hilbert(signal))
    env_energy = float(np.mean(env ** 2))
    # 2~5kHz 대역 에너지
    freqs = np.fft.rfftfreq(len(signal), 1.0 / 25_600)
    fft_mag = np.abs(np.fft.rfft(signal))
    band_mask = (freqs >= 2000) & (freqs <= 5000)
    band_energy = float(np.mean(fft_mag[band_mask] ** 2)) if band_mask.any() else 0.0
    return np.array([rms, kurtosis, crest, env_energy, band_energy], dtype=np.float32)


# ============================================================
# 데이터 파이프라인
# ============================================================

def build_pretrain_arrays(
    runs: list[dict],
    fs: int = 25_600,
    window_size: int = 32,
    stride: int = 4,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """
    생성된 Run 리스트를 슬라이딩 윈도우로 자르고
    STFT + Handcrafted 특징을 추출하여 학습 배열로 변환한다.

    Returns:
        X_vib: shape (N, window_size, n_channels, vibration_features)
        y_rul: shape (N,)
        meta:  각 샘플의 메타데이터 리스트
    """
    X_vib_list = []
    y_rul_list = []
    meta_list = []

    for run_idx, run in enumerate(runs):
        vib = run["vibration"]  # (seq_len, n_channels, signal_length)
        rul = run["rul"]        # (seq_len,)
        seq_len, n_channels, _ = vib.shape

        # 채널별 STFT + handcrafted 특징 추출
        features = np.zeros(
            (seq_len, n_channels, 513 + 5), dtype=np.float32
        )
        for step in range(seq_len):
            for ch in range(n_channels):
                sig = vib[step, ch, :]
                stft_feat = extract_stft_features(sig, fs=fs)
                hc_feat    = extract_handcrafted(sig)
                features[step, ch, :513] = stft_feat
                features[step, ch, 513:] = hc_feat

        # 슬라이딩 윈도우
        for start in range(0, seq_len - window_size + 1, stride):
            end = start + window_size
            window_feat = features[start:end]  # (window_size, n_channels, vib_features)
            target_rul  = rul[end - 1]

            X_vib_list.append(window_feat)
            y_rul_list.append(target_rul)
            meta_list.append({
                "run_idx":    run_idx,
                "fault_type": run["fault_type"],
                "rpm":        run["rpm"],
                "fault_freq": run["fault_freq"],
                "fn":         run["fn"],
                "zeta_L":     run["zeta_L"],
                "zeta_R":     run["zeta_R"],
                "step":       end - 1,
            })

    X_vib = np.stack(X_vib_list, axis=0)  # (N, window_size, n_channels, vib_features)
    y_rul = np.array(y_rul_list, dtype=np.float32)

    return X_vib, y_rul, meta_list


# ============================================================
# 메인
# ============================================================

def main(args: argparse.Namespace) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Physics-based Synthetic Bearing Data Generator")
    print("=" * 60)

    # 설정 구성
    config = SyntheticConfig(
        fs=args.fs,
        signal_duration_sec=args.signal_sec,
        seq_len=args.seq_len,
        n_channels=args.n_channels,
        rpm=args.rpm,
        multi_fault=args.multi_fault,
        slip_range=args.slip,
        seed=args.seed,
    )

    # 설정 저장 (재현성)
    config_dict = {k: v for k, v in config.__dict__.items() if not isinstance(v, np.ndarray)}
    config_dict["n_runs"] = args.n_runs
    with open(OUTPUT_DIR / "pretrain_config.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    print(f"  Config saved -> {OUTPUT_DIR / 'pretrain_config.json'}")

    # ── 단일 임펄스 시각화 (검증용) ─────────────────
    print("\n[Step 1] Generating single impulse response visualization...")
    rng_check = np.random.default_rng(args.seed)
    fn, zeta_L, zeta_R = sample_fn_zeta(rng_check, BEARING_SPECS)
    h = single_impulse_response(fn, zeta_L, zeta_R, fs=config.fs)
    plot_single_impulse(h, config.fs, save_path=VIZ_DIR / "single_impulse_h.png")
    print(f"  fn={fn:.1f}Hz | zeta_L={zeta_L:.4f} | zeta_R={zeta_R:.4f}")

    # ── 데이터 대량 생성 ────────────────────────────
    print(f"\n[Step 2] Generating {args.n_runs} synthetic runs...")
    t0 = time.time()
    runs = generate_dataset(
        n_runs=args.n_runs,
        config=config,
        fault_types=["BPFO", "BPFI", "BSF"],
        verbose=True,
    )
    elapsed = time.time() - t0
    print(f"  [OK] Done! {args.n_runs} runs in {elapsed:.1f}s ({elapsed/args.n_runs:.2f}s/run)")

    # ── 시각화 (처음 3개 Run만) ──────────────────────
    print(f"\n[Step 3] Visualizing first {min(3, len(runs))} runs...")
    for i, run in enumerate(runs[:3]):
        plot_degradation_dashboard(
            run,
            save_path=VIZ_DIR / f"run_{i:03d}_dashboard.png",
        )
        plot_rul_trajectory(
            run,
            save_path=VIZ_DIR / f"run_{i:03d}_rul_trajectory.png",
        )
    print(f"  [OK] Visualizations saved -> {VIZ_DIR}")

    # ── 특징 추출 및 배열 빌드 ───────────────────────
    print(f"\n[Step 4] Extracting STFT features (window={args.window}, stride={args.stride})...")
    t0 = time.time()
    X_vib, y_rul, meta = build_pretrain_arrays(
        runs,
        fs=config.fs,
        window_size=args.window,
        stride=args.stride,
    )
    elapsed = time.time() - t0
    print(f"  [OK] X_vib: {X_vib.shape}  y_rul: {y_rul.shape}  ({elapsed:.1f}s)")

    # ── 저장 ─────────────────────────────────────────
    print(f"\n[Step 5] Saving pretrain_data.npz...")
    save_path = OUTPUT_DIR / "pretrain_data.npz"
    np.savez_compressed(
        save_path,
        X_vib=X_vib,
        y_rul=y_rul,
        fault_types=np.array([m["fault_type"] for m in meta]),
        rpms=np.array([m["rpm"] for m in meta], dtype=np.float32),
    )
    size_mb = save_path.stat().st_size / 1024 / 1024
    print(f"  [OK] Saved -> {save_path} ({size_mb:.1f} MB)")
    print(f"\n  RUL range: {y_rul.min():.0f} ~ {y_rul.max():.0f}")
    print(f"  X_vib shape: {X_vib.shape}")
    print("\n[Done] Synthetic data generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Physics-based Synthetic Bearing Data Generator")
    parser.add_argument("--n-runs",     type=int,   default=50,    help="생성할 Run 수")
    parser.add_argument("--seq-len",    type=int,   default=100,   help="Run당 타임스텝 수")
    parser.add_argument("--n-channels", type=int,   default=4,     help="진동 채널 수")
    parser.add_argument("--fs",         type=int,   default=25600, help="샘플링 레이트 (Hz)")
    parser.add_argument("--signal-sec", type=float, default=1.0,   help="타임스텝당 신호 길이 (초)")
    parser.add_argument("--rpm",        type=float, default=1000.0,help="회전 속도 (RPM)")
    parser.add_argument("--slip",       type=float, default=0.10,  help="타이밍 지터 비율 (0.10=±10%)")
    parser.add_argument("--window",     type=int,   default=32,    help="슬라이딩 윈도우 크기")
    parser.add_argument("--stride",     type=int,   default=4,     help="슬라이딩 윈도우 스트라이드")
    parser.add_argument("--multi-fault",action="store_true",       help="다중 결함 혼합 활성화")
    parser.add_argument("--seed",       type=int,   default=42,    help="랜덤 시드")
    args = parser.parse_args()
    main(args)
