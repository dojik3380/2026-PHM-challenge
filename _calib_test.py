"""Test if per-case min-max calibration of HI predictions improves A_RUL.

Hypothesis: the model's HI ranking is good (spearman 0.81) but its absolute
levels are compressed/biased. If we rescale predictions per case to span
[0, 1] using the case's own min/max, then run Stage 2, A_RUL should improve.

No retraining; just post-hoc transform on the existing OOF parquet.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from inference import hi_sequence_to_rul
from model import asymmetric_rul_score_np

OOF = Path("models/HI_seed42_fold1_oof.parquet")
df = pd.read_parquet(OOF).sort_values("time_sec").reset_index(drop=True)

t = df["time_sec"].to_numpy(dtype=np.float64)
hi_p = df["hi_pred"].to_numpy(dtype=np.float64)
hi_t = df["hi_true"].to_numpy(dtype=np.float64)
rul_t = df["rul_true"].to_numpy(dtype=np.float64)
case_max = float(df["case_max"].iloc[0])

# Baseline: existing rul_pred saved by train.py
rul_p_orig = df["rul_pred"].to_numpy(dtype=np.float64)
arul_orig = float(np.mean(asymmetric_rul_score_np(rul_p_orig, rul_t)))
mae_orig = float(np.mean(np.abs(rul_p_orig - rul_t)))

# Calibration variant A: pure min-max per case to [0, 1]
hi_min = hi_p.min()
hi_max = hi_p.max()
hi_cal_A = (hi_p - hi_min) / max(hi_max - hi_min, 1e-9)

# Calibration variant B: stretch but assume start should be 0, end NOT necessarily 1
#   subtract min, divide by (max - min), but multiply by 0.95 so we don't force end=1
hi_cal_B = (hi_p - hi_min) / max(hi_max - hi_min, 1e-9) * 0.95

# Calibration variant C: anchor to typical training-case calibration. From earlier
# train pass (eyeballed): first-window pred ~ 0.45, last-window pred ~ 0.70.
# So pred 0.45 -> HI 0, pred 0.70 -> HI ~0.85 (typical end-of-life). Linear map.
PRED_HEALTHY = 0.45
PRED_FAILURE = 0.70
TRUE_FAILURE_HI = 0.85
hi_cal_C = (hi_p - PRED_HEALTHY) / max(PRED_FAILURE - PRED_HEALTHY, 1e-9) * TRUE_FAILURE_HI
hi_cal_C = np.clip(hi_cal_C, 0.0, 1.0)


def rul_from_hi(times, hi):
    out = np.zeros(len(times))
    order = np.argsort(times)
    t_s = times[order]
    h_s = hi[order]
    for i in range(len(t_s)):
        t_fail = hi_sequence_to_rul(t_s[: i + 1], h_s[: i + 1],
                                    t_now=float(t_s[i]), case_max=case_max)
        out[order[i]] = max(0.0, t_fail - t_s[i])
    return out


results = {
    "original (no calibration)": (hi_p, rul_p_orig),
    "A: per-case min-max -> [0,1]": (hi_cal_A, rul_from_hi(t, hi_cal_A)),
    "B: per-case min-max -> [0,0.95]": (hi_cal_B, rul_from_hi(t, hi_cal_B)),
    f"C: linear anchor ({PRED_HEALTHY}->0, {PRED_FAILURE}->{TRUE_FAILURE_HI})": (hi_cal_C, rul_from_hi(t, hi_cal_C)),
}

print(f"=== Train4 (n={len(df)}, case_max={case_max:.0f}s) ===\n")
print(f"{'variant':<55} {'A_RUL':>8} {'RUL_MAE':>9} {'HI_MAE':>8} {'pred_min':>9} {'pred_max':>9}")
for name, (hi_cal, rul_p) in results.items():
    arul = float(np.mean(asymmetric_rul_score_np(rul_p, rul_t)))
    mae = float(np.mean(np.abs(rul_p - rul_t)))
    hi_mae = float(np.mean(np.abs(hi_cal - hi_t)))
    print(f"{name:<55} {arul:>8.4f} {mae:>9.0f} {hi_mae:>8.3f} {hi_cal.min():>9.3f} {hi_cal.max():>9.3f}")
