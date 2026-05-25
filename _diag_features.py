"""Check raw feature trajectories for Train4 vs other cases.
No torch needed — just numpy + parquet + the cached feature files.
"""
import numpy as np
from pathlib import Path

# Load cached npz files for each case
cache_dir = Path("data2_features")
npz_files = sorted(cache_dir.glob("original_Train*.npz"))
print(f"Found {len(npz_files)} original case cache files")

for npz_path in npz_files:
    data = np.load(npz_path, allow_pickle=False)
    vib = data["vib"]    # (T, 4, 1026)
    feat = data["feat"]  # (T, 4, 10)
    times = data["times"]
    case_max = float(data["case_max"])

    case_name = npz_path.name.split("_")[1]  # "Train1", "Train2", etc.
    T = vib.shape[0]

    # RMS = feature index 0 (from FEATURE_INDEX)
    # Looking at HANDCRAFTED_FEATURES = ("RMS","PEAK_TO_PEAK","ABS_MEAN","SKEW","KURT","CREST","IMPULSE","SHAPE","ABS_MAX","RMS_HIGH")
    rms_idx = 0
    rms_per_step = feat[:, :, rms_idx].mean(axis=1)  # mean across 4 channels

    baseline_n = 10
    baseline_rms = rms_per_step[:baseline_n].mean()
    rms_growth = (rms_per_step - baseline_rms) / max(baseline_rms, 1e-9)

    # STFT RMS (mean power across all frequency bins): proxy for amplitude change
    vib_mean_power = vib.mean(axis=(1, 2))  # (T,) mean across channels and freq bins

    # KURT = feature index 4
    kurt_per_step = feat[:, :, 4].mean(axis=1)

    print(f"\n=== {case_name} (T={T}, case_max={case_max:.0f}s) ===")
    print(f"  RMS baseline: {baseline_rms:.4f}")
    print(f"  RMS at end:   {rms_per_step[-1]:.4f}  growth={rms_growth[-1]*100:.1f}%")
    print(f"  RMS max:      {rms_per_step.max():.4f}  growth={rms_growth.max()*100:.1f}%")
    print(f"  VIBE power baseline: {vib_mean_power[:baseline_n].mean():.4f}")
    print(f"  VIBE power at end:   {vib_mean_power[-1]:.4f}")
    print(f"  KURT range:   [{kurt_per_step.min():.2f}, {kurt_per_step.max():.2f}]")

    # Show every 10th timestep: time%, rms_growth%, kurt
    print("  [t%, rms_growth%, kurt_avg]")
    step = max(1, T // 8)
    for i in list(range(0, T, step)) + [T - 1]:
        pct = 100 * times[i] / case_max
        print(f"    t={pct:5.1f}%  rms_growth={rms_growth[i]*100:+6.1f}%  kurt={kurt_per_step[i]:.2f}")
