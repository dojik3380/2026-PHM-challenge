"""Convert raw JSON cases (Train_No_4/5/6/8) → parquet format like Train_No_1/2/3.

Raw layout (per case):
  Rawdata/CH##_YYMMDD_HHMMSS.json   ← {Channel, DateTime, Samples: [256000 floats]}
  Operation/Operation_YYMMDD_HHMMSS.json
  Feature/CH##_*.json               ← unused by current loader

Output:
  vibration.parquet   columns: timestep_idx, datetime, channel, samples
  operation.parquet   columns: timestep_idx, datetime, ...op fields
  feature.parquet     skipped (loader doesn't use it)

Streaming write via ParquetWriter — JSON 한 파일씩 읽고 row append → memory safe
even for 35GB cases.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


CHANNELS = ("CH03", "CH04", "CH05", "CH06")
TIMESTAMP_RE = re.compile(r"(\d{6})_(\d{6})")
VIB_CHUNK_ROWS = 32   # 한번에 write 하는 row 수 (4 channels × N timesteps)


def parse_timestamp(name: str) -> datetime | None:
    m = TIMESTAMP_RE.search(name)
    if not m:
        return None
    return datetime.strptime(f"{m.group(1)}_{m.group(2)}", "%y%m%d_%H%M%S")


def parse_channel(name: str) -> str | None:
    m = re.match(r"(CH\d{2})", name)
    return m.group(1) if m else None


def collect_timesteps(raw_dir: Path) -> tuple[list[datetime], dict[datetime, dict[str, Path]]]:
    """Return sorted list of unique timestamps and {dt: {channel: path}} mapping."""
    ts_to_files: dict[datetime, dict[str, Path]] = defaultdict(dict)
    for f in raw_dir.glob("*.json"):
        ch = parse_channel(f.name)
        dt = parse_timestamp(f.name)
        if ch is None or dt is None or ch not in CHANNELS:
            continue
        ts_to_files[dt][ch] = f
    timesteps = sorted(ts_to_files.keys())
    return timesteps, ts_to_files


def write_vibration(case_dir: Path, timesteps: list[datetime],
                    ts_to_files: dict[datetime, dict[str, Path]]) -> None:
    out = case_dir / "vibration.parquet"
    schema = pa.schema([
        ("timestep_idx", pa.int32()),
        ("datetime",     pa.string()),
        ("channel",      pa.string()),
        ("samples",      pa.list_(pa.float32())),
    ])
    writer = pq.ParquetWriter(out, schema, compression="snappy")
    buf: list[dict] = []
    total = len(timesteps) * len(CHANNELS)
    done = 0
    for ts_idx, dt in enumerate(timesteps):
        dt_str = dt.strftime("%Y-%m-%d %H:%M:%S")
        for ch in CHANNELS:
            fpath = ts_to_files[dt].get(ch)
            if fpath is None:
                continue
            with open(fpath, "rb") as fp:
                d = json.load(fp)
            samples = np.asarray(d.get("Samples", []), dtype=np.float32)
            buf.append({
                "timestep_idx": ts_idx,
                "datetime":     dt_str,
                "channel":      ch,
                "samples":      samples.tolist(),
            })
            done += 1
            if len(buf) >= VIB_CHUNK_ROWS:
                writer.write_table(pa.Table.from_pylist(buf, schema=schema))
                buf.clear()
                gc.collect()
                pct = 100.0 * done / total
                sys.stdout.write(f"\r  vib: {done}/{total} ({pct:5.1f}%)")
                sys.stdout.flush()
    if buf:
        writer.write_table(pa.Table.from_pylist(buf, schema=schema))
    writer.close()
    print(f"\n  → {out.name}  ({out.stat().st_size / 1e9:.2f} GB)")


def write_operation(case_dir: Path, timesteps: list[datetime]) -> None:
    op_dir = case_dir / "Operation"
    dt_to_idx = {dt: i for i, dt in enumerate(timesteps)}
    rows: list[dict] = []
    for f in sorted(op_dir.glob("*.json")):
        dt = parse_timestamp(f.name)
        if dt is None or dt not in dt_to_idx:
            continue
        with open(f, "rb") as fp:
            d = json.load(fp)
        d = {str(k): v for k, v in d.items()}
        d["timestep_idx"] = dt_to_idx[dt]
        d["datetime"]     = dt.strftime("%Y-%m-%d %H:%M:%S")
        rows.append(d)
    if not rows:
        print("  ! no operation rows")
        return
    # union of keys
    keys = sorted({k for r in rows for k in r})
    table = pa.table({k: [r.get(k) for r in rows] for k in keys})
    out = case_dir / "operation.parquet"
    pq.write_table(table, out, compression="snappy")
    print(f"  → {out.name}  ({out.stat().st_size / 1e6:.1f} MB, {len(rows)} rows)")


def convert_case(case_dir: Path, skip_if_exists: bool = True) -> None:
    vib_out = case_dir / "vibration.parquet"
    op_out  = case_dir / "operation.parquet"
    if skip_if_exists and vib_out.exists() and op_out.exists():
        print(f"[skip] {case_dir.name}: parquets already exist")
        return
    raw_dir = case_dir / "Rawdata"
    op_dir  = case_dir / "Operation"
    if not raw_dir.is_dir() or not op_dir.is_dir():
        print(f"[skip] {case_dir.name}: missing Rawdata/Operation folder")
        return
    print(f"\n[{case_dir.name}]")
    timesteps, ts_to_files = collect_timesteps(raw_dir)
    print(f"  timesteps: {len(timesteps)}  (× {len(CHANNELS)} channels)")
    write_vibration(case_dir, timesteps, ts_to_files)
    write_operation(case_dir, timesteps)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data2-dir", type=Path, default=Path(__file__).parent / "data2")
    p.add_argument("--cases", nargs="+", default=["Train_No_4", "Train_No_5", "Train_No_6", "Train_No_8"])
    p.add_argument("--force", action="store_true", help="overwrite existing parquets")
    args = p.parse_args()

    for name in args.cases:
        cdir = args.data2_dir / name
        if not cdir.is_dir():
            print(f"[skip] {name}: directory not found")
            continue
        convert_case(cdir, skip_if_exists=not args.force)


if __name__ == "__main__":
    main()
