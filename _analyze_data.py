"""데이터셋 심층 분석 — 시간 구조 검증.

사용자 정보:
  - data/Train (TDMS): 1분 측정 + 9분 휴식 (10분 cycle)
  - data2:             10초 간격 연속 측정

검증 항목:
  1. TDMS: 파일 개수, op_df.time_sec 범위·간격, 파일명 timestamp pattern
  2. data2: timestep 간격 (datetime 차)
  3. case_max 가 wall-clock 인지 측정 누적인지 결정
  4. 현재 data_loader 가 가정하는 등간격 timestep 의 정확성
"""

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from config import TRAIN_DIR, DATA2_DIR, TDMS_CHUNKS_PER_FILE
from data_loader import discover_cases, load_operation_csv, _parse_datetime


def analyze_tdms_case(case_name: str, csv_path: Path, vib_dir: Path) -> None:
    print(f"\n[TDMS] {case_name}")
    print(f"  CSV : {csv_path.name}")
    print(f"  Dir : {vib_dir}")

    # TDMS 파일 패턴
    tdms_files = sorted(vib_dir.glob("*.tdms"))
    n_files = len(tdms_files)
    print(f"  TDMS files: {n_files}  →  n_steps={n_files * TDMS_CHUNKS_PER_FILE}")
    if tdms_files:
        print(f"    first: {tdms_files[0].name}")
        print(f"    last : {tdms_files[-1].name}")

    # Operation CSV
    op_df = load_operation_csv(csv_path)
    print(f"  op_df: {len(op_df)} rows, columns={list(op_df.columns)}")
    print(f"    time_sec: min={op_df['time_sec'].min():.1f}  max={op_df['time_sec'].max():.1f}")
    print(f"    range total: {op_df['time_sec'].max() - op_df['time_sec'].min():.1f} s "
          f"({(op_df['time_sec'].max() - op_df['time_sec'].min()) / 3600:.2f} h)")
    # time_sec 간격 분포
    if len(op_df) > 1:
        dt = np.diff(op_df["time_sec"].to_numpy())
        print(f"    time_sec diff: min={dt.min():.2f}, max={dt.max():.2f}, mean={dt.mean():.2f}, median={np.median(dt):.2f}")
        # 분포 quantile
        print(f"    diff quantiles: 10%={np.percentile(dt, 10):.2f}, 50%={np.percentile(dt, 50):.2f}, "
              f"90%={np.percentile(dt, 90):.2f}, 99%={np.percentile(dt, 99):.2f}")

    # RPM (있다면)
    for rpm_col in ("Rotor Speed [RPM]", "motor_speed_rpm", "RPM"):
        if rpm_col in op_df.columns:
            rpm = op_df[rpm_col].dropna().to_numpy()
            print(f"    RPM ({rpm_col}): min={rpm.min():.0f} max={rpm.max():.0f} mean={rpm.mean():.0f}")
            break

    # 측정 wall-clock 추정: n_files * 60s = 측정 시간만; case_max 와 비교
    pure_measure_time = n_files * 60.0  # 1 file = 60s 측정
    case_max = float(op_df["time_sec"].max())
    print(f"  pure measurement time: {pure_measure_time:.0f} s  ({pure_measure_time/3600:.2f}h)")
    print(f"  case_max (op csv):     {case_max:.0f} s  ({case_max/3600:.2f}h)")
    ratio = case_max / max(pure_measure_time, 1.0)
    print(f"  ratio (case_max / pure): {ratio:.2f}  "
          f"→ {'wall-clock' if ratio > 5 else 'measurement-only'}")


def analyze_data2_case(case_dir: Path) -> None:
    import pyarrow.parquet as pq
    print(f"\n[data2] {case_dir.name}")
    op_path = case_dir / "operation.parquet"
    if not op_path.exists():
        print("  no operation.parquet")
        return
    op_df = pq.read_table(op_path).to_pandas()
    print(f"  rows: {len(op_df)}  columns: {list(op_df.columns)}")

    # vibration parquet 의 datetime 분포로 wall-clock 추정
    vib_path = case_dir / "vibration.parquet"
    pf = pq.ParquetFile(vib_path)
    # 첫 row group 의 datetime 만 sample
    table = pf.read_row_group(0, columns=["timestep_idx", "datetime"])
    rows = table.to_pylist()
    dt_by_step: dict[int, datetime] = {}
    for r in rows:
        ts = int(r["timestep_idx"])
        dt = _parse_datetime(r["datetime"])
        if dt is not None and ts not in dt_by_step:
            dt_by_step[ts] = dt
    # 추가 group 도 sample
    for g in range(1, min(pf.num_row_groups, 5)):
        table = pf.read_row_group(g, columns=["timestep_idx", "datetime"])
        for r in table.to_pylist():
            ts = int(r["timestep_idx"])
            dt = _parse_datetime(r["datetime"])
            if dt is not None and ts not in dt_by_step:
                dt_by_step[ts] = dt

    if len(dt_by_step) < 2:
        print("  not enough datetime entries")
        return
    sorted_steps = sorted(dt_by_step.keys())
    dts = [dt_by_step[s] for s in sorted_steps]
    diffs = [(dts[i+1] - dts[i]).total_seconds() for i in range(len(dts)-1)]
    diffs = np.asarray(diffs)
    print(f"  datetime samples: {len(diffs)+1}")
    print(f"  step-to-step delta: min={diffs.min():.1f}, max={diffs.max():.1f}, "
          f"mean={diffs.mean():.2f}, median={np.median(diffs):.2f}")
    # case_max 추정
    first_dt = min(dts)
    last_dt = max(dts)
    pure_steps = len(set(t for r in rows for t in [int(r['timestep_idx'])]))
    wall_clock_range = (last_dt - first_dt).total_seconds()
    print(f"  first dt: {first_dt}")
    print(f"  last  dt: {last_dt}  (in sampled groups)")
    print(f"  wall-clock range (sampled): {wall_clock_range:.0f}s ({wall_clock_range/3600:.2f}h)")


def main() -> None:
    print("=" * 78)
    print("TDMS cases (data/Train)")
    print("=" * 78)
    for case_name, csv_path, vib_dir in discover_cases(TRAIN_DIR):
        analyze_tdms_case(case_name, csv_path, vib_dir)

    print("\n" + "=" * 78)
    print("data2 cases")
    print("=" * 78)
    for case_dir in sorted(Path(DATA2_DIR).glob("Train_No_*")):
        if (case_dir / "vibration.parquet").exists() and case_dir.is_dir():
            analyze_data2_case(case_dir)


if __name__ == "__main__":
    main()
