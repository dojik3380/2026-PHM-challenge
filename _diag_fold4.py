"""Diagnose fold4 (Train4 holdout) OOF predictions."""
import numpy as np
import pandas as pd

df = pd.read_parquet("models/RUL_seed42_fold4_oof.parquet")
print(f"Rows: {len(df)}")
print(f"Columns: {list(df.columns)}")
print()

# Split by case
cases = df["case_name"].unique()
print(f"Cases in fold4 OOF: {list(cases)}")
print()

for case in sorted(cases):
    sub = df[df["case_name"] == case].sort_values("end_timestep").reset_index(drop=True)
    raw = sub["rul_pred_raw"].to_numpy()
    true = sub["rul_true"].to_numpy()
    time = sub["time_sec"].to_numpy()
    hi_pred = sub["hi_pred"].to_numpy()
    hi_true = sub["hi_true"].to_numpy()
    print(f"=== {case} ({len(sub)} windows, case_max={sub['case_max'].iloc[0]:.0f}s) ===")
    print(f"  rul_pred_raw: min={raw.min():.0f}  max={raw.max():.0f}  mean={raw.mean():.0f}")
    print(f"  rul_true:     min={true.min():.0f}  max={true.max():.0f}  mean={true.mean():.0f}")
    print(f"  ratio(pred/true): {(raw/np.maximum(true,1)).mean():.3f}")
    print(f"  hi_pred: min={hi_pred.min():.3f}  max={hi_pred.max():.3f}  mean={hi_pred.mean():.3f}")
    print(f"  hi_true: min={hi_true.min():.3f}  max={hi_true.max():.3f}  mean={hi_true.mean():.3f}")
    # Show first and last 5 windows
    print("  [first 5 windows]")
    for r in sub.head(5).itertuples():
        print(f"    t={r.time_sec:7.0f}s  rul_true={r.rul_true:7.0f}  rul_pred_raw={r.rul_pred_raw:7.0f}  "
              f"hi_true={r.hi_true:.3f}  hi_pred={r.hi_pred:.3f}")
    print("  [last 5 windows]")
    for r in sub.tail(5).itertuples():
        print(f"    t={r.time_sec:7.0f}s  rul_true={r.rul_true:7.0f}  rul_pred_raw={r.rul_pred_raw:7.0f}  "
              f"hi_true={r.hi_true:.3f}  hi_pred={r.hi_pred:.3f}")
    print()

# Compare with fold1 (also weak) and fold2 (strong)
print("\n=== Cross-fold comparison ===")
for fi in range(1, 5):
    df_f = pd.read_parquet(f"models/RUL_seed42_fold{fi}_oof.parquet")
    raw = df_f["rul_pred_raw"].to_numpy()
    true = df_f["rul_true"].to_numpy()
    case = df_f["held_out_case"].iloc[0]
    pct_over = 100 * (raw > true).mean()
    print(f"  fold{fi} ({case}): pred_mean={raw.mean():.0f}  true_mean={true.mean():.0f}  "
          f"ratio={raw.mean()/true.mean():.3f}  pct_over={pct_over:.1f}%")
