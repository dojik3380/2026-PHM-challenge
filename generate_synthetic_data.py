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

from config import (
    SAMPLING_RATE, STFT_NPERSEG, STFT_NOVERLAP, STFT_FREQ_BINS,
    VIBRATION_FEATURES_PER_CHANNEL, WINDOW_SIZE, STRIDE,
)
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
PROJECT_ROOT    = Path(__file__).parent
OUTPUT_DIR      = PROJECT_ROOT / "outputs" / "synthetic"
VIZ_DIR         = OUTPUT_DIR / "visualization"
PHYSICS_JSON    = PROJECT_ROOT / "outputs" / "physics_parameters.json"


# ============================================================
# physics_parameters.json 연동
# ============================================================

def load_physics_params(json_path: Path = PHYSICS_JSON) -> dict | None:
    """physics_parameters.json을 로드한다. 없으면 None 반환."""
    if not json_path.exists():
        return None
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def build_bearing_specs_from_json(params: dict) -> dict:
    """
    physics_parameters.json → BEARING_SPECS 형식 딕셔너리 변환.
    fn_primary_mean이 8000Hz 초과이거나 std=0이면 test-mode artifact로 판단하여
    기본값(simulator/physics.py의 BEARING_SPECS)을 유지한다.
    """
    from simulator.physics import BEARING_SPECS
    specs = dict(BEARING_SPECS)

    fn_mean = float(params.get("fn_primary_mean", specs["fn_mean"]))
    fn_std  = float(params.get("fn_primary_std",  specs["fn_std"]))

    if fn_mean > 8_000 or fn_std == 0.0:
        print(f"  [fn 경고] JSON fn={fn_mean:.0f}Hz / std={fn_std:.1f} → test-mode 추정값. "
              f"기본값 유지 ({specs['fn_mean']:.0f}±{specs['fn_std']:.1f}Hz)")
    else:
        specs["fn_mean"] = fn_mean
        specs["fn_std"]  = fn_std if fn_std > 0 else fn_mean * 0.05
        specs["fn_min"]  = fn_mean * 0.70
        specs["fn_max"]  = fn_mean * 1.25

    z_mean = float(params.get("damping_mean", specs["zeta_mean"]))
    z_std  = float(params.get("damping_std",  specs["zeta_std"]))
    specs["zeta_mean"] = z_mean
    specs["zeta_std"]  = z_std if z_std > 0 else z_mean * 0.30
    specs["zeta_min"]  = z_mean * 0.10
    specs["zeta_max"]  = min(z_mean * 3.0, 0.30)

    return specs


def build_rpm_ranges_from_json(params: dict) -> tuple[tuple, tuple]:
    """
    JSON의 RPM 통계 → (rpm_low_range, rpm_high_range) 변환.
    전체 범위를 low/high 두 regime으로 분할:
      low  = (rpm_min, rpm_mean * 0.85)
      high = (rpm_mean * 0.95, rpm_max)
    """
    rpm_range = params.get("rpm_range", [700.0, 1500.0])
    rpm_min  = float(rpm_range[0])
    rpm_max  = float(rpm_range[1])
    rpm_mean = float(params.get("rpm_mean", (rpm_min + rpm_max) / 2))

    low_range  = (rpm_min,             max(rpm_min * 1.05, rpm_mean * 0.85))
    high_range = (rpm_mean * 0.95,     rpm_max)
    return low_range, high_range


def print_physics_summary(bearing_specs: dict, rpm_low: tuple, rpm_high: tuple) -> None:
    """적용된 물리 파라미터를 출력한다."""
    print("\n  [Physics Parameters from JSON]")
    print(f"    fn    : {bearing_specs['fn_mean']:.0f} ± {bearing_specs['fn_std']:.1f} Hz"
          f"  (range {bearing_specs['fn_min']:.0f}~{bearing_specs['fn_max']:.0f})")
    print(f"    zeta  : {bearing_specs['zeta_mean']:.4f} ± {bearing_specs['zeta_std']:.4f}"
          f"  (range {bearing_specs['zeta_min']:.4f}~{bearing_specs['zeta_max']:.4f})")
    print(f"    RPM low  : {rpm_low[0]:.0f}~{rpm_low[1]:.0f}")
    print(f"    RPM high : {rpm_high[0]:.0f}~{rpm_high[1]:.0f}")


# ============================================================
# STFT 특징 추출 (기존 model.py 파이프라인과 동일한 규격)
# ============================================================

def extract_stft_features(
    signal: np.ndarray,
    fs: int = SAMPLING_RATE,
    nperseg: int = STFT_NPERSEG,
    noverlap: int = STFT_NOVERLAP,
) -> np.ndarray:
    """
    1D 진동 신호에서 STFT 특징 벡터를 추출한다.
    출력 shape: (VIBRATION_FEATURES_PER_CHANNEL,) = (1026,)
    concat(mean(|Zxx|), std(|Zxx|)) — features/vibration.py와 동일한 규격.
    boundary=None, padded=False로 실제 파이프라인과 일치.
    """
    def _pad_or_trim(v: np.ndarray) -> np.ndarray:
        if v.size < STFT_FREQ_BINS:
            return np.pad(v, (0, STFT_FREQ_BINS - v.size))
        return v[:STFT_FREQ_BINS]

    _, _, Zxx = scipy_stft(signal, fs=fs, nperseg=nperseg, noverlap=noverlap,
                           boundary=None, padded=False)
    mag = np.abs(Zxx)
    freq_mean = _pad_or_trim(np.mean(mag, axis=1)).astype(np.float32)
    freq_std  = _pad_or_trim(np.std(mag,  axis=1)).astype(np.float32)
    return np.concatenate([freq_mean, freq_std])  # (VIBRATION_FEATURES_PER_CHANNEL,)


def extract_rms(signal: np.ndarray) -> float:
    """단일 채널 RMS 계산"""
    return float(np.sqrt(np.mean(signal ** 2)))


# ============================================================
# 데이터 파이프라인
# ============================================================

def build_pretrain_arrays(
    runs: list[dict],
    fs: int = 25_600,
    window_size: int = 32,
    stride: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """
    생성된 Run 리스트를 슬라이딩 윈도우로 자르고
    STFT와 auxiliary 특징을 추출하여 학습 배열로 변환한다.

    Returns:
        X_vib: shape (N, window_size, n_channels, 513)
        X_aux: shape (N, window_size, 2) - [RPM, RMS]
        y_rul: shape (N,)
        meta:  각 샘플의 메타데이터 리스트
    """
    X_vib_list = []
    X_aux_list = []
    y_rul_list = []
    meta_list = []

    for run_idx, run in enumerate(runs):
        vib = run["vibration"]  # (seq_len, n_channels, signal_length)
        rul = run["rul"]        # (seq_len,)
        rpm_traj = run["rpm_trajectory"] # (seq_len,)
        seq_len, n_channels, _ = vib.shape

        # 채널별 STFT + RMS 특징 추출
        stft_features = np.zeros((seq_len, n_channels, VIBRATION_FEATURES_PER_CHANNEL), dtype=np.float32)
        rms_features = np.zeros((seq_len, n_channels), dtype=np.float32)
        
        for step in range(seq_len):
            for ch in range(n_channels):
                sig = vib[step, ch, :]
                stft_features[step, ch, :] = extract_stft_features(sig, fs=fs)
                rms_features[step, ch] = extract_rms(sig)

        # 각 스텝별 평균 RMS (4채널 평균)
        mean_rms = np.mean(rms_features, axis=1)

        # Auxiliary 생성: (seq_len, 2)
        aux_features = np.stack([rpm_traj, mean_rms], axis=1).astype(np.float32)

        # 슬라이딩 윈도우
        for start in range(0, seq_len - window_size + 1, stride):
            end = start + window_size
            window_vib = stft_features[start:end]  # (window_size, n_channels, 513)
            window_aux = aux_features[start:end]   # (window_size, 2)
            target_rul = rul[end - 1]

            X_vib_list.append(window_vib)
            X_aux_list.append(window_aux)
            y_rul_list.append(target_rul)
            meta_list.append({
                "run_idx":    run_idx,
                "fault_type": run["fault_type"],
                "rpm_mean":   np.mean(window_aux[:, 0]),
                "fn":         run["fn"],
                "zeta_L":     run["zeta_L"],
                "zeta_R":     run["zeta_R"],
                "step":       end - 1,
            })

    X_vib = np.stack(X_vib_list, axis=0)  # (N, window_size, n_channels, 513)
    X_aux = np.stack(X_aux_list, axis=0)  # (N, window_size, 2)
    y_rul = np.array(y_rul_list, dtype=np.float32)

    return X_vib, X_aux, y_rul, meta_list


# ============================================================
# 메인
# ============================================================

def main(args: argparse.Namespace) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Physics-based Synthetic Bearing Data Generator")
    print("=" * 60)

    # ── physics_parameters.json 로드 → 시뮬레이터 파라미터 교체 ──────
    physics_params = load_physics_params(PHYSICS_JSON)
    if physics_params is not None and not args.no_json:
        print(f"  [JSON] {PHYSICS_JSON.name} 로드 성공 → 실제 데이터 기반 파라미터 적용")
        bearing_specs  = build_bearing_specs_from_json(physics_params)
        rpm_low_range, rpm_high_range = build_rpm_ranges_from_json(physics_params)
        print_physics_summary(bearing_specs, rpm_low_range, rpm_high_range)
    else:
        if args.no_json:
            print("  [JSON] --no-json 플래그 → 기본 BEARING_SPECS 사용")
        else:
            print(f"  [JSON] {PHYSICS_JSON} 없음 → 기본 BEARING_SPECS 사용")
        bearing_specs = None
        rpm_center = args.rpm
        rpm_low_range  = (rpm_center * 0.72, rpm_center * 0.78)
        rpm_high_range = (rpm_center * 0.94, rpm_center * 0.98)

    config = SyntheticConfig(
        fs=args.fs,
        signal_duration_sec=args.signal_sec,
        seq_len=args.seq_len,
        n_channels=args.n_channels,
        rpm_low_range=rpm_low_range,
        rpm_high_range=rpm_high_range,
        multi_fault=args.multi_fault,
        slip_range=args.slip,
        seed=args.seed,
        bearing_specs=bearing_specs,
        total_life_seconds=args.total_life_sec,
    )

    # 설정 저장 (재현성)
    config_dict = {
        k: v for k, v in config.__dict__.items()
        if not isinstance(v, np.ndarray) and k != "bearing_specs"
    }
    if config.bearing_specs is not None:
        config_dict["bearing_specs_fn_mean"]  = config.bearing_specs.get("fn_mean")
        config_dict["bearing_specs_zeta_mean"] = config.bearing_specs.get("zeta_mean")
    config_dict["n_runs"] = args.n_runs
    with open(OUTPUT_DIR / "pretrain_config.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    print(f"  Config saved -> {OUTPUT_DIR / 'pretrain_config.json'}")

    # ── 단일 임펄스 시각화 (검증용) ─────────────────
    print("\n[Step 1] Generating single impulse response visualization...")
    rng_check = np.random.default_rng(args.seed)
    fn, zeta_L, zeta_R = sample_fn_zeta(rng_check, bearing_specs or BEARING_SPECS)
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
    X_vib, X_aux, y_rul, meta = build_pretrain_arrays(
        runs,
        fs=config.fs,
        window_size=args.window,
        stride=args.stride,
    )
    elapsed = time.time() - t0
    print(f"  [OK] X_vib: {X_vib.shape}  X_aux: {X_aux.shape}  y_rul: {y_rul.shape}  ({elapsed:.1f}s)")

    # ── 저장 ─────────────────────────────────────────
    print(f"\n[Step 5] Saving pretrain_data.npz...")
    save_path = OUTPUT_DIR / "pretrain_data.npz"
    np.savez_compressed(
        save_path,
        X_vib=X_vib,
        X_aux=X_aux,
        y_rul=y_rul,
        fault_types=np.array([m["fault_type"] for m in meta]),
        rpms=np.array([m["rpm_mean"] for m in meta], dtype=np.float32),
    )
    size_mb = save_path.stat().st_size / 1024 / 1024
    print(f"  [OK] Saved -> {save_path} ({size_mb:.1f} MB)")
    print(f"\n  RUL range: {y_rul.min():.0f} ~ {y_rul.max():.0f}")
    print(f"  X_vib shape: {X_vib.shape}")
    print("\n[Done] Synthetic data generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Physics-based Synthetic Bearing Data Generator")
    parser.add_argument("--n-runs",     type=int,   default=50,           help="생성할 Run 수")
    parser.add_argument("--seq-len",    type=int,   default=100,          help="Run당 타임스텝 수")
    parser.add_argument("--n-channels", type=int,   default=4,            help="진동 채널 수")
    parser.add_argument("--fs",         type=int,   default=SAMPLING_RATE,help="샘플링 레이트 (Hz)")
    parser.add_argument("--signal-sec", type=float, default=1.0,          help="타임스텝당 신호 길이 (초)")
    parser.add_argument("--rpm",        type=float, default=1000.0,       help="회전 속도 (RPM)")
    parser.add_argument("--slip",       type=float, default=0.10,         help="타이밍 지터 비율 (0.10=±10%)")
    parser.add_argument("--window",     type=int,   default=WINDOW_SIZE,  help="슬라이딩 윈도우 크기")
    parser.add_argument("--stride",     type=int,   default=STRIDE,       help="슬라이딩 윈도우 스트라이드")
    parser.add_argument("--total-life-sec", type=float, default=70000.0,  help="합성 RUL 총 수명(초) — 실제 데이터 평균에 맞춤")
    parser.add_argument("--multi-fault",action="store_true",              help="다중 결함 혼합 활성화")
    parser.add_argument("--no-json",    action="store_true",              help="physics_parameters.json 무시하고 기본값 사용")
    parser.add_argument("--seed",       type=int,   default=42,           help="랜덤 시드")
    args = parser.parse_args()
    main(args)
