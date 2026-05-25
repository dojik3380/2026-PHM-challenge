"""Sweep DENORM_SCALE on existing OOF parquets — no retraining needed."""
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
    np.arange(0.60, 1.11, 0.05),
    np.arange(1.20, 3.01, 0.20),
])

print(f"{'scale':>6}  " + "  ".join(f"f{i:1d}" for i in range(1, 5)) + "  AVERAGE")
print("-" * 55)

best_scale, best_avg = 1.0, 0.0
rows = []
for scale in scales:
    per_fold = []
    for fold_idx in range(1, 5):
        df = fold_results[fold_idx]
        pred = np.maximum(df[raw_col].to_numpy() * scale, 0.0)
        true_v = df[true_col].to_numpy()
        per_fold.append(a_rul_score(pred, true_v))
    avg = float(np.mean(per_fold))
    rows.append((scale, per_fold, avg))
    if avg > best_avg:
        best_avg = avg
        best_scale = scale

for scale, per_fold, avg in rows:
    marker = " <-- BEST" if abs(scale - best_scale) < 1e-9 else ""
    print(f"{scale:6.2f}  " + "  ".join(f"{s:.4f}" for s in per_fold) + f"  {avg:.4f}{marker}")

print(f"\nBest DENORM_SCALE = {best_scale:.2f}  (avg A_RUL = {best_avg:.4f})")

# Also show: what is the per-case last-window prediction breakdown at best scale?
print("\n--- Per-fold breakdown at best scale ---")
for fold_idx in range(1, 5):
    df = fold_results[fold_idx]
    pred_raw = df[raw_col].to_numpy()
    true_v = df[true_col].to_numpy()
    pred = np.maximum(pred_raw * best_scale, 0.0)
    # Over vs under counts
    over = int((pred > true_v).sum())
    under = int((pred <= true_v).sum())
    print(f"  fold{fold_idx}: over={over}({100*over/len(pred):.1f}%)  under={under}({100*under/len(pred):.1f}%)  "
          f"pred_mean={pred.mean():.0f}  true_mean={true_v.mean():.0f}")
