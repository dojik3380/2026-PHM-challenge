"""v3 합성 데이터의 추가 갭 정밀 진단.

분석:
    1. RPM/RMS auxiliary 분포 비교
    2. RUL 분포 (lifetime evolution) 비교
    3. Channel correlation 비교 (4채널 간 상관)
    4. Temporal evolution (각 timestep의 spectrum 변화)
"""
from __future__ import annotations

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import SAMPLING_RATE
from data_loader import discover_cases, load_tdms_channels
from features.rpm_estimator import extract_auxiliary_vector
from features.vibration import vibration_stft_timestep


OUT = PROJECT_ROOT / "outputs" / "v3_gap_diag"
OUT.mkdir(parents=True, exist_ok=True)


def extract_real_per_case(n_per_case: int = 25):
    """case별 RPM/RMS trajectory + 채널별 STFT 통계."""
    cases = discover_cases(PROJECT_ROOT / "data" / "Train")
    out = {}
    for case_name, _csv, vib_dir in cases:
        tdms_files = sorted(Path(vib_dir).glob("*.tdms"))
        idxs = np.linspace(0, len(tdms_files) - 1, num=min(n_per_case, len(tdms_files)), dtype=int)
        rpms, rmss, vib_list, channels_raw = [], [], [], []
        progresses = []
        for i in idxs:
            tdms_path = tdms_files[i]
            try:
                ch_data = load_tdms_channels(tdms_path)
                aux = extract_auxiliary_vector(ch_data)
                rpms.append(aux[0])
                rmss.append(aux[1])
                vib = vibration_stft_timestep(tdms_path)  # (4, 1026)
                vib_list.append(vib)
                progresses.append(float(i / max(1, len(tdms_files) - 1)))
                # channel correlation 계산용: 첫 1 sec raw signal 1024 sample만
                # 메모리 절약을 위해 첫 4096 samples만 보관
                raw = []
                for ch_name in ("CH1", "CH2", "CH3", "CH4"):
                    s = ch_data.get(ch_name) if ch_data.get(ch_name) is not None else ch_data.get(ch_name.lower())
                    if s is not None:
                        raw.append(np.asarray(s[:4096], dtype=np.float32))
                if len(raw) == 4:
                    channels_raw.append(np.stack(raw, axis=0))
            except Exception as exc:
                print(f"  [WARN] {tdms_path.name}: {exc}")

        out[case_name] = {
            "rpms": np.array(rpms),
            "rmss": np.array(rmss),
            "vib": np.stack(vib_list, axis=0) if vib_list else None,
            "channels_raw": np.stack(channels_raw, axis=0) if channels_raw else None,  # (N, 4, 4096)
            "progresses": np.array(progresses),
        }
        print(f"  [{case_name}] RPM={rpms[0]:.0f}-{rpms[-1]:.0f}, RMS={rmss[0]:.3f}-{rmss[-1]:.3f}, n={len(rpms)}")
    return out


def synthetic_per_run():
    """합성 데이터 npz에서 window별 RPM/RMS/vib 추출."""
    data = np.load(PROJECT_ROOT / "outputs" / "synthetic" / "pretrain_data.npz")
    xvib = data["X_vib"]   # (N_win, 32, 4, 1026)
    xaux = data["X_aux"]   # (N_win, 32, 2)
    yrul = data["y_rul"]   # (N_win,)
    # window 중간 timestep을 sample로 사용
    mid = xvib.shape[1] // 2
    return {
        "rpms": xaux[:, mid, 0],   # (N_win,)
        "rmss": xaux[:, mid, 1],
        "vib":  xvib[:, mid],      # (N_win, 4, 1026)
        "rul":  yrul,
    }


def channel_correlation_matrix(raw_samples: np.ndarray) -> np.ndarray:
    """raw_samples shape: (N, 4, len) → 평균 cross-correlation matrix (4, 4)."""
    if raw_samples is None or len(raw_samples) == 0:
        return None
    cc = np.zeros((4, 4))
    for s in raw_samples:
        for i in range(4):
            for j in range(4):
                vi, vj = s[i], s[j]
                vi = vi - vi.mean()
                vj = vj - vj.mean()
                denom = (vi.std() * vj.std() + 1e-10) * len(vi)
                cc[i, j] += float(np.dot(vi, vj) / denom)
    return cc / len(raw_samples)


def main():
    print("=" * 60)
    print("  v3 simulator gap diagnosis")
    print("=" * 60)

    print("\n[Real per case]")
    real_cases = extract_real_per_case(n_per_case=25)
    real_all_rpm = np.concatenate([d["rpms"] for d in real_cases.values()])
    real_all_rms = np.concatenate([d["rmss"] for d in real_cases.values()])

    print("\n[Synthetic]")
    sim = synthetic_per_run()
    print(f"  N windows: {len(sim['rpms'])}")
    print(f"  RPM: {sim['rpms'].min():.0f} - {sim['rpms'].max():.0f}, mean={sim['rpms'].mean():.0f}")
    print(f"  RMS: {sim['rmss'].min():.4f} - {sim['rmss'].max():.4f}, mean={sim['rmss'].mean():.4f}")

    print("\n[Comparison]")
    print(f"  Real RPM: {real_all_rpm.min():.0f} - {real_all_rpm.max():.0f}, mean={real_all_rpm.mean():.0f}")
    print(f"  Sim  RPM: {sim['rpms'].min():.0f} - {sim['rpms'].max():.0f}, mean={sim['rpms'].mean():.0f}")
    print(f"  Real RMS: {real_all_rms.min():.4f} - {real_all_rms.max():.4f}, mean={real_all_rms.mean():.4f}")
    print(f"  Sim  RMS: {sim['rmss'].min():.4f} - {sim['rmss'].max():.4f}, mean={sim['rmss'].mean():.4f}")
    print(f"  RPM mean ratio (sim/real): {sim['rpms'].mean() / real_all_rpm.mean():.2f}")
    print(f"  RMS mean ratio (sim/real): {sim['rmss'].mean() / real_all_rms.mean():.2f}")

    # ── Channel correlation 비교 ─────────────────────
    print("\n[Channel correlation (raw signal cross-correlation)]")
    for cname, d in real_cases.items():
        cc = channel_correlation_matrix(d["channels_raw"])
        if cc is not None:
            off_diag = cc[np.triu_indices(4, k=1)]
            print(f"  Real [{cname}]: off-diag mean={off_diag.mean():.3f} (range {off_diag.min():.3f}..{off_diag.max():.3f})")

    # 합성 데이터는 raw signal 저장 안 되어 있음 — channel별 STFT 상관 대신
    sim_vib = sim["vib"]  # (N_win, 4, 1026)
    # 채널 간 STFT vector 상관 (mean spectrum 영역)
    n_win = sim_vib.shape[0]
    sample = sim_vib[:200, :, :513]  # mean-half만, 200 sample
    cc_stft = np.zeros((4, 4))
    for s in sample:
        for i in range(4):
            for j in range(4):
                vi, vj = s[i] - s[i].mean(), s[j] - s[j].mean()
                cc_stft[i, j] += float(np.dot(vi, vj) / (vi.std() * vj.std() * len(vi) + 1e-10))
    cc_stft /= len(sample)
    off_diag_sim = cc_stft[np.triu_indices(4, k=1)]
    print(f"  Sim  STFT off-diag mean={off_diag_sim.mean():.3f} (range {off_diag_sim.min():.3f}..{off_diag_sim.max():.3f})")

    # 마찬가지로 실측의 STFT off-diag 확인
    for cname, d in real_cases.items():
        if d["vib"] is None:
            continue
        sample = d["vib"][:, :, :513]
        cc_stft = np.zeros((4, 4))
        for s in sample:
            for i in range(4):
                for j in range(4):
                    vi, vj = s[i] - s[i].mean(), s[j] - s[j].mean()
                    cc_stft[i, j] += float(np.dot(vi, vj) / (vi.std() * vj.std() * len(vi) + 1e-10))
        cc_stft /= len(sample)
        off_diag_real = cc_stft[np.triu_indices(4, k=1)]
        print(f"  Real [{cname}] STFT off-diag mean={off_diag_real.mean():.3f}")

    # ── 시각화: RPM/RMS 히스토그램 + RUL distribution ──────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.hist(real_all_rpm, bins=30, alpha=0.5, label="Real", color="blue", density=True)
    ax.hist(sim["rpms"], bins=30, alpha=0.5, label="Sim", color="red", density=True)
    ax.set_xlabel("RPM"); ax.set_ylabel("density"); ax.set_title("RPM distribution"); ax.legend()

    ax = axes[0, 1]
    bins = np.linspace(0, max(real_all_rms.max(), sim["rmss"].max()), 40)
    ax.hist(real_all_rms, bins=bins, alpha=0.5, label="Real", color="blue", density=True)
    ax.hist(sim["rmss"], bins=bins, alpha=0.5, label="Sim", color="red", density=True)
    ax.set_xlabel("RMS"); ax.set_ylabel("density"); ax.set_title("RMS distribution"); ax.legend()
    ax.set_xlim(0, real_all_rms.quantile(0.95) if hasattr(real_all_rms, "quantile") else np.quantile(real_all_rms, 0.95) * 2)

    # RMS trajectory per case (real) vs RUL (sim aggregate)
    ax = axes[1, 0]
    for cname, d in real_cases.items():
        ax.plot(d["progresses"], d["rmss"], "o-", markersize=3, label=cname, alpha=0.7)
    ax.set_xlabel("Progress (0=start, 1=end)"); ax.set_ylabel("RMS"); ax.set_title("Real RMS trajectory per case"); ax.legend(fontsize=8)

    ax = axes[1, 1]
    # sim rul → progress (1 - rul/max)
    sim_progress = 1.0 - sim["rul"] / sim["rul"].max()
    # bin progress, plot mean RMS per bin
    bin_edges = np.linspace(0, 1, 21)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    bin_means = []
    for i in range(20):
        m = (sim_progress >= bin_edges[i]) & (sim_progress < bin_edges[i + 1])
        bin_means.append(sim["rmss"][m].mean() if m.any() else np.nan)
    ax.plot(bin_centers, bin_means, "ro-", label="Sim (binned)")
    ax.set_xlabel("Progress"); ax.set_ylabel("RMS"); ax.set_title("Sim RMS vs progress"); ax.legend()

    fig.tight_layout()
    fig.savefig(OUT / "rpm_rms_distribution.png", dpi=110)
    plt.close(fig)
    print(f"\n  Plot saved → {OUT / 'rpm_rms_distribution.png'}")

    # 보고서
    summary = {
        "real_rpm": {"min": float(real_all_rpm.min()), "max": float(real_all_rpm.max()), "mean": float(real_all_rpm.mean()), "std": float(real_all_rpm.std())},
        "sim_rpm":  {"min": float(sim["rpms"].min()),  "max": float(sim["rpms"].max()),  "mean": float(sim["rpms"].mean()),  "std": float(sim["rpms"].std())},
        "real_rms": {"min": float(real_all_rms.min()), "max": float(real_all_rms.max()), "mean": float(real_all_rms.mean()), "std": float(real_all_rms.std())},
        "sim_rms":  {"min": float(sim["rmss"].min()),  "max": float(sim["rmss"].max()),  "mean": float(sim["rmss"].mean()),  "std": float(sim["rmss"].std())},
        "rpm_ratio_sim_over_real": float(sim["rpms"].mean() / real_all_rpm.mean()),
        "rms_ratio_sim_over_real": float(sim["rmss"].mean() / real_all_rms.mean()),
        "real_per_case_rpm": {k: {"min": float(d["rpms"].min()), "max": float(d["rpms"].max()), "mean": float(d["rpms"].mean())} for k, d in real_cases.items()},
        "real_per_case_rms": {k: {"min": float(d["rmss"].min()), "max": float(d["rmss"].max()), "mean": float(d["rmss"].mean())} for k, d in real_cases.items()},
    }
    with open(OUT / "diagnosis.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  Report saved → {OUT / 'diagnosis.json'}")


if __name__ == "__main__":
    main()