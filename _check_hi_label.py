"""Verify new cumulative HI label on actual cached features."""
import numpy as np
from pathlib import Path

# Simulate the new compute_hi_labels for hybrid mode
def compute_hi_hybrid_new(times, case_max, rms_per_step, baseline):
    progress = np.clip(times / max(case_max, 1.0), 0.0, 1.0)
    if baseline < 1e-9:
        return 0.5 * progress
    positive_dev = np.maximum(rms_per_step - baseline, 0.0)
    cum_damage = np.cumsum(positive_dev)
    final_cum = float(cum_damage[-1]) if len(cum_damage) > 0 else 0.0
    if final_cum > 1e-9:
        cum_damage_norm = cum_damage / final_cum
    else:
        cum_damage_norm = np.zeros_like(rms_per_step)
    return 0.5 * progress + 0.5 * cum_damage_norm

FEATURE_INDEX = {"RMS": 0}
DEGRADATION_BASELINE_TIMESTEPS = 10

cache_dir = Path("data2_features")
for npz_path in sorted(cache_dir.glob("original_Train*.npz")):
    data = np.load(npz_path, allow_pickle=False)
    feat = data["feat"]      # (T, 4, 10)
    times = data["times"]
    case_max = float(data["case_max"])
    case_name = npz_path.name.split("_")[1]

    rms_per_step = feat[:, :, 0].mean(axis=1).astype(np.float64)
    n_base = min(DEGRADATION_BASELINE_TIMESTEPS, len(rms_per_step))
    baseline = float(np.mean(rms_per_step[:n_base]))

    hi = compute_hi_hybrid_new(times, case_max, rms_per_step, baseline)

    print(f"{case_name}: HI[0]={hi[0]:.3f}  HI[25%]={hi[len(hi)//4]:.3f}  "
          f"HI[50%]={hi[len(hi)//2]:.3f}  HI[75%]={hi[3*len(hi)//4]:.3f}  "
          f"HI[-1]={hi[-1]:.3f}  min={hi.min():.3f}  max={hi.max():.3f}  "
          f"monotonic={bool(np.all(np.diff(hi) >= -1e-6))}")
