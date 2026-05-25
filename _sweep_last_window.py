"""Optimal DENORM_SCALE for LAST-WINDOW prediction only (matches competition metric)."""
import numpy as np
import pandas as pd

OVER_EST_PENALTY_SCALE = 20.0
UNDER_EST_PENALTY_SCALE = 50.0


def a_rul_score(pred, target):
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    denominator = np.maximum(np.abs(target), 1e-6)
    er = 100.0 * (target - pred) / denominator
    ln_half = np.log(0.5)
    exponent = np.where(
        er <= 0,
        -ln_half * er / OVER_EST_PENALTY_SCALE,
        ln_half * er / UNDER_EST_PENALTY_SCALE,
    )
    return float(np.exp(exponent).mean())


fold_results = {}
for fold_idx in range(1, 5):
    fold_results[fold_idx] = pd.read_parquet(f"models/RUL_seed42_fold{fold_idx}_oof.parquet")

raw_col = "rul_pred_raw"
true_col = "rul_true"

scales = np.concatenate([
    np.arange(0.30, 1.01, 0.10),
    np.arange(1.00, 4.01, 0.25),
])

print("=== Per-window (all windows) ===")
print(f"{'scale':>6}  " + "  ".join(f"f{i:1d}" for i in range(1, 5)) + "  AVERAGE")
print("-" * 55)
best_all, best_all_avg = 1.0, 0.0
for scale in scales:
    per_fold = []
    for fi in range(1, 5):
        df = fold_results[fi]
        pred = np.maximum(df[raw_col].to_numpy() * scale, 0.0)
        per_fold.append(a_rul_score(pred, df[true_col].to_numpy()))
    avg = float(np.mean(per_fold))
    if avg > best_all_avg:
        best_all_avg = avg; best_all = scale
    print(f"{scale:6.2f}  " + "  ".join(f"{s:.4f}" for s in per_fold) + f"  {avg:.4f}")
print(f"Best all-window: scale={best_all:.2f}  avg={best_all_avg:.4f}")

print("\n=== LAST WINDOW ONLY (competition metric) ===")
print(f"{'scale':>6}  " + "  ".join(f"f{i:1d}" for i in range(1, 5)) + "  AVERAGE")
print("-" * 55)
best_lw, best_lw_avg = 1.0, 0.0
lw_rows = []
for scale in scales:
    per_fold = []
    for fi in range(1, 5):
        df = fold_results[fi]
        # Last window = highest end_timestep per case (only 1 case per fold here)
        last_idx = df["end_timestep"].idxmax()
        pred_lw = max(df.loc[last_idx, raw_col] * scale, 0.0)
        true_lw = df.loc[last_idx, true_col]
        score_lw = a_rul_score(np.array([pred_lw]), np.array([true_lw]))
        per_fold.append(score_lw)
    avg = float(np.mean(per_fold))
    lw_rows.append((scale, per_fold, avg))
    if avg > best_lw_avg:
        best_lw_avg = avg; best_lw = scale

for scale, per_fold, avg in lw_rows:
    mark = " <-- BEST" if abs(scale - best_lw) < 1e-9 else ""
    print(f"{scale:6.2f}  " + "  ".join(f"{s:.4f}" for s in per_fold) + f"  {avg:.4f}{mark}")
print(f"Best last-window: scale={best_lw:.2f}  avg={best_lw_avg:.4f}")

print("\n=== Last-window predictions at best scales ===")
for fi in range(1, 5):
    df = fold_results[fi]
    last_idx = df["end_timestep"].idxmax()
    raw = df.loc[last_idx, raw_col]
    true = df.loc[last_idx, true_col]
    held = df["held_out_case"].iloc[0]
    print(f"  fold{fi} ({held}): raw={raw:.0f}s  true={true:.0f}s  "
          f"@scale=1.80→pred={raw*1.80:.0f}  @scale={best_lw:.2f}→pred={raw*best_lw:.0f}")
