"""Direct look at Train4 OOF predictions: what is the model actually doing?"""

from pathlib import Path
import numpy as np
import pandas as pd

OOF_PATH = Path("models/HI_seed42_fold1_oof.parquet")

if not OOF_PATH.exists():
    raise SystemExit(f"missing {OOF_PATH}")

df = pd.read_parquet(OOF_PATH).sort_values("time_sec").reset_index(drop=True)
print(f"loaded {len(df)} OOF rows from {OOF_PATH}")
print(f"columns: {list(df.columns)}")
print()

t = df["time_sec"].to_numpy()
hi_t = df["hi_true"].to_numpy()
hi_p = df["hi_pred"].to_numpy()
ru_t = df["rul_true"].to_numpy()
ru_p = df["rul_pred"].to_numpy()
case_max = float(df["case_max"].iloc[0])
print(f"case_max = {case_max:.0f}s ({case_max/3600:.2f}h)")
print()

print("HI prediction vs truth at 5 lifecycle positions (early/early-mid/mid/mid-late/late):")
positions = [0.0, 0.25, 0.50, 0.75, 1.0]
for p in positions:
    target_t = p * case_max
    idx = int(np.argmin(np.abs(t - target_t)))
    print(f"  t={t[idx]:7.0f}s ({p*100:3.0f}%) | hi_true={hi_t[idx]:.3f}  hi_pred={hi_p[idx]:.3f}  "
          f"rul_true={ru_t[idx]:7.0f}  rul_pred={ru_p[idx]:7.0f}")
print()

print("Histogram of HI predictions (range, percentiles):")
print(f"  hi_pred: min={hi_p.min():.3f} max={hi_p.max():.3f} mean={hi_p.mean():.3f}")
print(f"           p10={np.percentile(hi_p,10):.3f} p50={np.percentile(hi_p,50):.3f} p90={np.percentile(hi_p,90):.3f}")
print(f"  hi_true: min={hi_t.min():.3f} max={hi_t.max():.3f} mean={hi_t.mean():.3f}")
print(f"           p10={np.percentile(hi_t,10):.3f} p50={np.percentile(hi_t,50):.3f} p90={np.percentile(hi_t,90):.3f}")
print()

print("Spearman ranking of predicted HI vs time:")
rt = pd.Series(t).rank().to_numpy()
rp = pd.Series(hi_p).rank().to_numpy()
spear = float(np.corrcoef(rt, rp)[0, 1])
print(f"  spearman(time, hi_pred) = {spear:+.3f}  (target: > +0.7, ideal: +1.0 = HI grows monotonically)")
rt_h = pd.Series(hi_t).rank().to_numpy()
spear_hi = float(np.corrcoef(rt_h, rp)[0, 1])
print(f"  spearman(hi_true, hi_pred) = {spear_hi:+.3f}  (target: > +0.7)")
print()

print("Where is the prediction WORST? (top-10 |hi_pred - hi_true| windows)")
err = np.abs(hi_p - hi_t)
order = np.argsort(err)[::-1][:10]
for j in order:
    print(f"  t={t[j]:7.0f}s ({t[j]/case_max*100:5.1f}%) | hi_true={hi_t[j]:.3f} hi_pred={hi_p[j]:.3f} |err|={err[j]:.3f}")
print()

print("RUL prediction vs truth at the same 5 positions:")
for p in positions:
    target_t = p * case_max
    idx = int(np.argmin(np.abs(t - target_t)))
    err_rul = ru_p[idx] - ru_t[idx]
    print(f"  t={t[idx]:7.0f}s ({p*100:3.0f}%) | rul_true={ru_t[idx]:7.0f}  rul_pred={ru_p[idx]:7.0f}  err={err_rul:+8.0f}")
