"""One-off lifetime audit. Reports the exact case_max that load_dataset uses."""

from pathlib import Path
import pyarrow.parquet as pq

from data_loader import (
    discover_cases,
    load_data2_case_timesteps,
    load_operation_csv,
    load_original_case_timesteps,
)
from config import DATA2_DIR, TRAIN_DIR


print("=" * 80)
print("TDMS Train1-4 (data/Train/) - lifetime from operation CSV")
print("=" * 80)
tdms_results = []
for case_name, csv_path, vib_dir in discover_cases(TRAIN_DIR):
    df = load_operation_csv(csv_path)
    t_min = float(df["time_sec"].min())
    t_max = float(df["time_sec"].max())
    duration = t_max - t_min
    n_rows = len(df)
    vib, feat, times, case_max_loaded, _ = load_original_case_timesteps(
        case_name, csv_path, vib_dir, use_cache=True,
    )
    print(f"  {case_name:>8}: csv t_min={t_min:.1f}s t_max={t_max:.1f}s duration={duration:.1f}s")
    print(f"          csv n_rows={n_rows}  cache n_timesteps={len(times)}  cache case_max={case_max_loaded:.1f}s")
    print(f"          lifetime = {case_max_loaded:.0f}s = {case_max_loaded/3600:.2f}h = {case_max_loaded/60:.0f}min")
    tdms_results.append((case_name, case_max_loaded))

print()
print("=" * 80)
print("data2 Train_No_1-3 (data2/) - lifetime via datetime delta in vibration.parquet")
print("=" * 80)
data2_results = []
for case_dir in sorted(DATA2_DIR.glob("Train_No_*")):
    if not (case_dir / "vibration.parquet").exists():
        continue
    case_name = case_dir.name

    op_df = pq.read_table(case_dir / "operation.parquet").to_pandas()
    n_op_rows = len(op_df)
    op_ts_max = int(op_df["timestep_idx"].max())

    vib, feat, times, case_max_loaded, _ = load_data2_case_timesteps(case_dir, use_cache=True)
    n_steps = len(times)

    print(f"  {case_name:>14}: operation rows={n_op_rows} max_timestep_idx={op_ts_max}")
    print(f"                   load_data2 n_steps={n_steps} case_max={case_max_loaded:.1f}s times[-1]={times[-1]:.1f}s")
    print(f"                   lifetime = {case_max_loaded:.0f}s = {case_max_loaded/3600:.2f}h = {case_max_loaded/60:.0f}min")
    print(f"                   (10s/step nominal: {n_steps*10}s; actual datetime-based: {case_max_loaded:.0f}s)")
    data2_results.append((case_name, case_max_loaded))

print()
print("=" * 80)
print("SUMMARY (case_max that load_dataset uses for HI labels & RUL targets)")
print("=" * 80)
all_cases = [("TDMS", n, lm) for n, lm in tdms_results] + [("data2", n, lm) for n, lm in data2_results]
print(f"  {'source':>6} {'case':>16} {'lifetime_s':>12} {'hours':>8}")
for src, name, lm in all_cases:
    print(f"  {src:>6} {name:>16} {lm:>12.0f} {lm/3600:>8.2f}")
print()
lifetimes = [lm for _, _, lm in all_cases]
print(f"  min  = {min(lifetimes):.0f}s ({min(lifetimes)/3600:.2f}h)")
print(f"  max  = {max(lifetimes):.0f}s ({max(lifetimes)/3600:.2f}h)")
print(f"  mean = {sum(lifetimes)/len(lifetimes):.0f}s")
print(f"  ratio max/min = {max(lifetimes)/min(lifetimes):.1f}x")
