"""Unified data loader for HI prediction (TDMS data/ + parquet data2/).

Outputs per window:
    X_vib  : (N, window, 4, VIBRATION_FEATURES_PER_CHANNEL=1026)
    X_feat : (N, window, 4, HANDCRAFTED_DIM=10) - RPM-INDEPENDENT features only
    hi     : (N,)  Health Indicator label in [0, 1]  (training target)
    rul    : (N,)  True RUL in seconds              (for evaluation only)
    metadata: case_name, source, start_timestep, end_timestep, time_sec, case_max

RPM has been removed from the pipeline (RPM ablation showed it was not a usable
signal). Bearing fault-frequency features (BPFO/BPFI/BSF/FTF, F_1X..F_3456X)
have been dropped along with it; the 10 retained handcrafted features
(RMS/kurtosis/crest/etc.) are all RPM-independent.

Operation CSV is still read at training time to obtain `case_max` (the
ground-truth lifetime needed for HI labels). At inference time no CSV is
required - the model predicts HI from vibration alone.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from nptdms import TdmsFile

from config import (
    DATA2_DIR,
    DATA2_FEATURE_CACHE_DIR,
    DEGRADATION_BASELINE_TIMESTEPS,
    HANDCRAFTED_DIM,
    HANDCRAFTED_FEATURES,
    HI_DAMAGE_SCALE,
    HI_FEATURES_ENABLED,
    HI_LABEL_MODE,
    HI_LABEL_POWER,
    SAMPLING_RATE,
    TDMS_CHUNK_SAMPLES,
    TDMS_CHUNK_SECONDS,
    TDMS_CHUNKS_PER_FILE,
    TRAIN_DIR,
    VIBRATION_FEATURES_PER_CHANNEL,
)
from features.degradation import augment_with_degradation, compute_global_baseline
from features.vibration import bearing_fault_amplitudes, stft_magnitude_vector


CSV_ENCODINGS = ("utf-8-sig", "utf-8", "cp949")
DATA2_CHANNELS = ("CH03", "CH04", "CH05", "CH06")
TDMS_CHANNELS = ("CH1", "CH2", "CH3", "CH4")
DATA2_TIMESTEP_SECONDS = 10.0


# ============================================================================
# TDMS / Operation CSV helpers
# ============================================================================


def load_tdms_channels(file_path) -> Dict[str, np.ndarray]:
    """Read TDMS file -> {channel_name: samples_array}."""
    tdms_file = TdmsFile.read(file_path)
    out: Dict[str, np.ndarray] = {}
    for group in tdms_file.groups():
        for channel in group.channels():
            out[channel.name] = channel[:]
    return out


def _read_csv(path: Path) -> pd.DataFrame:
    last_error: Optional[Exception] = None
    for encoding in CSV_ENCODINGS:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise last_error or ValueError(f"Could not read CSV: {path}")


def _find_column(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    normalized = {str(col).strip().lower(): col for col in columns}
    for cand in candidates:
        if cand.lower() in normalized:
            return normalized[cand.lower()]
    for col in columns:
        col_lower = str(col).strip().lower()
        if any(cand.lower() in col_lower for cand in candidates):
            return col
    return None


def load_operation_csv(csv_path: Path) -> pd.DataFrame:
    """Read TDMS operation CSV; only time_sec is needed for HI labels."""
    df = _read_csv(csv_path)
    time_col = _find_column(df.columns, ("time_sec", "time", "sec"))
    if time_col is None:
        raise ValueError(f"{csv_path} must contain a time/sec column.")
    df = df.rename(columns={time_col: "time_sec"})
    df["time_sec"] = pd.to_numeric(df["time_sec"], errors="coerce").fillna(0.0)
    return df.sort_values("time_sec").reset_index(drop=True)


def _case_name_from_operation(csv_path: Path) -> str:
    return csv_path.stem.replace("_Operation", "")


def _find_vibration_dir(case_name: str, root_dir: Path) -> Optional[Path]:
    candidates = [
        root_dir / f"{case_name}_Vibration" / f"{case_name}_Vibration",
        root_dir / f"{case_name}_Vibration",
        root_dir / case_name / case_name,
        root_dir / case_name,
    ]
    for cand in candidates:
        if cand.exists() and any(cand.glob("*.tdms")):
            return cand
    for cand in sorted(root_dir.rglob(f"{case_name}*")):
        if cand.is_dir() and any(cand.glob("*.tdms")):
            return cand
    return None


def discover_cases(root_dir: Path) -> List[Tuple[str, Path, Path]]:
    """Discover TDMS train cases (case_name, csv_path, vibration_dir)."""
    root_dir = Path(root_dir)
    cases = []
    for csv_path in sorted(root_dir.glob("*_Operation.csv")):
        case_name = _case_name_from_operation(csv_path)
        vib_dir = _find_vibration_dir(case_name, root_dir)
        if vib_dir is not None:
            cases.append((case_name, csv_path, vib_dir))
    return cases


# ============================================================================
# Handcrafted PHM feature extraction (RPM-independent only)
# ============================================================================


_FEATURE_ALIASES = {
    "PEAKTOPEAK": "PEAK_TO_PEAK",
    "ABS.MEAN": "ABS_MEAN",
    "ABSMEAN": "ABS_MEAN",
    "ABS.MAX": "ABS_MAX",
    "ABSMAX": "ABS_MAX",
}


def _clean_feature_name(name: str) -> str:
    text = str(name).strip().upper().replace(" ", "_").replace("-", "_")
    compact = re.sub(r"[^A-Z0-9.]+", "", text)
    return _FEATURE_ALIASES.get(compact, text.replace(".", "_"))


FEATURE_INDEX = {_clean_feature_name(name): i for i, name in enumerate(HANDCRAFTED_FEATURES)}


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return v if np.isfinite(v) else default


def _parse_datetime(value: object) -> Optional[datetime]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%y%m%d_%H%M%S", "%Y%m%d_%H%M%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    try:
        return pd.to_datetime(text).to_pydatetime()
    except Exception:
        return None


def _amplitude_band(signal: np.ndarray, low: float, high: float) -> float:
    if signal.size < 4:
        return 0.0
    arr = np.asarray(signal, dtype=np.float32)
    spectrum = np.abs(np.fft.rfft(arr))
    freqs = np.fft.rfftfreq(arr.size, d=1.0 / SAMPLING_RATE)
    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        return 0.0
    return float(np.mean(spectrum[mask]) / max(arr.size, 1))


def handcrafted_features_from_signal(signal: Iterable[float]) -> np.ndarray:
    """Compute 10 RPM-independent statistics + 6 bearing fault frequency amplitudes."""
    arr = np.asarray(signal, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    out = np.zeros(HANDCRAFTED_DIM, dtype=np.float32)
    if arr.size == 0:
        return out

    mean = float(np.mean(arr))
    centered = arr - mean
    std = float(np.std(arr))
    rms = float(np.sqrt(np.mean(arr ** 2)))
    abs_mean = float(np.mean(np.abs(arr)))
    abs_max = float(np.max(np.abs(arr)))
    p2p = float(np.ptp(arr))
    skew = float(np.mean(centered ** 3) / (std ** 3 + 1e-12)) if std > 0 else 0.0
    kurt = float(np.mean(centered ** 4) / (std ** 4 + 1e-12)) if std > 0 else 0.0

    values = {
        "RMS": rms,
        "PEAK_TO_PEAK": p2p,
        "ABS_MEAN": abs_mean,
        "SKEW": skew,
        "KURT": kurt,
        "CREST": abs_max / (rms + 1e-12),
        "IMPULSE": abs_max / (abs_mean + 1e-12),
        "SHAPE": rms / (abs_mean + 1e-12),
        "ABS_MAX": abs_max,
        "RMS_HIGH": _amplitude_band(arr, 5_000.0, SAMPLING_RATE / 2.0),
    }
    for name, value in values.items():
        out[FEATURE_INDEX[name]] = _safe_float(value)

    # Bearing fault frequency amplitudes: RPM estimated from this chunk's own FFT
    fault_amps = bearing_fault_amplitudes(arr.astype(np.float64), SAMPLING_RATE)
    for i, name in enumerate(("BPFI_1X", "BPFI_2X", "BPFO_1X", "BPFO_2X", "BSF_1X", "FTF_1X")):
        out[FEATURE_INDEX[name]] = _safe_float(fault_amps[i])

    return out


# ============================================================================
# HI label generation
# ============================================================================


def _rms_baseline_and_trajectory(feat_raw: np.ndarray) -> tuple[float, np.ndarray]:
    """RMS healthy baseline (mean of first N) and per-timestep avg-channel RMS.

    feat_raw shape: (T, C, F) raw 10-dim features. F=HANDCRAFTED_DIM.
    Returns (baseline_scalar, per_timestep_rms).
    """
    rms_idx = FEATURE_INDEX["RMS"]
    rms_per_step = feat_raw[:, :, rms_idx].mean(axis=1).astype(np.float64)
    n_base = min(DEGRADATION_BASELINE_TIMESTEPS, len(rms_per_step))
    baseline = float(np.mean(rms_per_step[:n_base])) if n_base > 0 else 0.0
    return baseline, rms_per_step


def compute_hi_labels(
    times: np.ndarray,
    case_max: float,
    feat_raw: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Per-timestep HI label in [0, 1].

    Time-based modes need only (times, case_max). Damage/hybrid modes also
    need feat_raw (T, C, HANDCRAFTED_DIM) to read RMS.

    "hybrid" uses CUMULATIVE damage (not instantaneous) so the label is
    monotonically increasing and always reaches 1.0 at end-of-life.
    This correctly handles cases like Train4 where instantaneous RMS growth
    is erratic — cumulative damage grows smoothly throughout.
    """
    progress = np.clip(times / max(case_max, 1.0), 0.0, 1.0).astype(np.float32)

    if HI_LABEL_MODE == "linear":
        return progress
    if HI_LABEL_MODE == "power":
        return np.power(progress, HI_LABEL_POWER, dtype=np.float32)
    if HI_LABEL_MODE in ("damage", "hybrid"):
        if feat_raw is None:
            raise ValueError(f"HI_LABEL_MODE={HI_LABEL_MODE} requires feat_raw")
        baseline, rms_per_step = _rms_baseline_and_trajectory(feat_raw)
        if baseline < 1e-9:
            cum_damage_norm = np.zeros_like(rms_per_step, dtype=np.float32)
        else:
            # Positive RMS deviation from healthy baseline, accumulated over time.
            # Cumsum is monotonically increasing and always ends at its maximum.
            positive_dev = np.maximum(rms_per_step - baseline, 0.0)
            cum_damage = np.cumsum(positive_dev).astype(np.float64)
            final_cum = float(cum_damage[-1]) if len(cum_damage) > 0 else 0.0
            if final_cum > 1e-9:
                cum_damage_norm = (cum_damage / final_cum).astype(np.float32)
            else:
                cum_damage_norm = np.zeros_like(rms_per_step, dtype=np.float32)
        if HI_LABEL_MODE == "damage":
            return cum_damage_norm
        # hybrid: always spans [0, 1] by construction (both components end at 1.0)
        return (0.5 * progress + 0.5 * cum_damage_norm).astype(np.float32)

    raise ValueError(f"Unknown HI_LABEL_MODE: {HI_LABEL_MODE}")


# ============================================================================
# Per-case cache
# ============================================================================


CACHE_VERSION = "v8_faultfreq"


def _cache_key(paths: Iterable[Path], extra: str) -> str:
    digest = hashlib.md5(extra.encode("utf-8"))
    for path in sorted(Path(p) for p in paths if Path(p).exists()):
        st = path.stat()
        digest.update(str(path.resolve()).encode("utf-8"))
        digest.update(str(st.st_mtime_ns).encode("ascii"))
        digest.update(str(st.st_size).encode("ascii"))
    return digest.hexdigest()


def _save_case_cache(path: Path, vib: np.ndarray, feat: np.ndarray,
                     times: np.ndarray, case_max: float, meta: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        vib=vib.astype(np.float32),
        feat=feat.astype(np.float32),
        times=times.astype(np.float32),
        case_max=np.float32(case_max),
        metadata_json=meta.to_json(orient="records", force_ascii=True),
    )


def _load_case_cache(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, pd.DataFrame]:
    data = np.load(path, allow_pickle=False)
    meta = pd.read_json(StringIO(data["metadata_json"].item()))
    return (
        data["vib"], data["feat"], data["times"],
        float(data["case_max"]), meta,
    )


def _parquet():
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pyarrow is required for data2 parquet loading. "
            "Install with: python -m pip install pyarrow"
        ) from exc
    return pq


# ============================================================================
# Per-case extractors
# ============================================================================


def load_original_case_timesteps(
    case_name: str,
    operation_csv: Path,
    vibration_dir: Path,
    use_cache: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, pd.DataFrame]:
    """TDMS train case -> (vib, feat, times, case_max, meta) per chunk."""
    tdms_files = sorted(Path(vibration_dir).glob("*.tdms"))
    op_df = load_operation_csv(operation_csv)
    key = _cache_key([operation_csv, *tdms_files], f"orig:{case_name}:{CACHE_VERSION}")
    cache_path = DATA2_FEATURE_CACHE_DIR / f"original_{case_name}_{key}.npz"
    if use_cache and cache_path.exists():
        return _load_case_cache(cache_path)

    n_files = len(tdms_files)
    n_steps = n_files * TDMS_CHUNKS_PER_FILE
    if n_steps == 0:
        raise ValueError(f"No TDMS files in {vibration_dir}")

    case_max = float(op_df["time_sec"].max())
    if n_steps > 1:
        times = (case_max * np.arange(n_steps, dtype=np.float32) / float(n_steps - 1)).astype(np.float32)
    else:
        times = np.array([case_max], dtype=np.float32)

    vib_steps = np.zeros((n_steps, len(TDMS_CHANNELS), VIBRATION_FEATURES_PER_CHANNEL), dtype=np.float32)
    feat_steps = np.zeros((n_steps, len(TDMS_CHANNELS), HANDCRAFTED_DIM), dtype=np.float32)

    for file_idx, tdms_path in enumerate(tdms_files):
        chans = load_tdms_channels(tdms_path)
        normalized = {name.upper(): np.asarray(v, dtype=np.float32) for name, v in chans.items()}
        ch_arrays = [normalized.get(ch, np.array([], dtype=np.float32)) for ch in TDMS_CHANNELS]
        for chunk_idx in range(TDMS_CHUNKS_PER_FILE):
            gstep = file_idx * TDMS_CHUNKS_PER_FILE + chunk_idx
            start = chunk_idx * TDMS_CHUNK_SAMPLES
            end = start + TDMS_CHUNK_SAMPLES
            for ch_i, ch_arr in enumerate(ch_arrays):
                chunk = ch_arr[start:end] if ch_arr.size >= end else ch_arr[start:]
                vib_steps[gstep, ch_i] = stft_magnitude_vector(chunk)
                feat_steps[gstep, ch_i] = handcrafted_features_from_signal(chunk)

    meta = pd.DataFrame({
        "case_name": case_name,
        "source": "original",
        "timestep_idx": np.arange(n_steps),
        "time_sec": times,
    })
    result = (np.nan_to_num(vib_steps), np.nan_to_num(feat_steps), times, case_max, meta)
    if use_cache:
        _save_case_cache(cache_path, *result)
    return result


def load_data2_case_timesteps(
    case_dir: Path,
    use_cache: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, pd.DataFrame]:
    """data2 parquet case -> (vib, feat, times, case_max, meta) per timestep."""
    case_dir = Path(case_dir)
    case_name = case_dir.name
    op_path = case_dir / "operation.parquet"
    vib_path = case_dir / "vibration.parquet"
    key = _cache_key([op_path, vib_path], f"data2:{case_name}:{CACHE_VERSION}")
    cache_path = DATA2_FEATURE_CACHE_DIR / f"data2_{case_name}_{key}.npz"
    if use_cache and cache_path.exists():
        return _load_case_cache(cache_path)

    pq = _parquet()
    op_df = pq.read_table(op_path).to_pandas().sort_values("timestep_idx").reset_index(drop=True)
    n_steps = int(op_df["timestep_idx"].max()) + 1

    vib_steps = np.zeros(
        (n_steps, len(DATA2_CHANNELS), stft_magnitude_vector(np.zeros(1024, dtype=np.float32)).shape[0]),
        dtype=np.float32,
    )
    feat_steps = np.zeros((n_steps, len(DATA2_CHANNELS), HANDCRAFTED_DIM), dtype=np.float32)
    dt_by_step: Dict[int, datetime] = {}

    pf = pq.ParquetFile(vib_path)
    for group_idx in range(pf.num_row_groups):
        table = pf.read_row_group(group_idx, columns=["timestep_idx", "datetime", "channel", "samples"])
        for row in table.to_pylist():
            ts = int(row["timestep_idx"])
            channel = str(row["channel"])
            if ts < 0 or ts >= n_steps or channel not in DATA2_CHANNELS:
                continue
            ch_idx = DATA2_CHANNELS.index(channel)
            samples = np.asarray(row["samples"], dtype=np.float32)
            vib_steps[ts, ch_idx] = stft_magnitude_vector(samples)
            feat_steps[ts, ch_idx] = handcrafted_features_from_signal(samples)
            parsed = _parse_datetime(row.get("datetime"))
            if parsed is not None:
                dt_by_step.setdefault(ts, parsed)

    if len(dt_by_step) >= 2:
        first_dt = min(dt_by_step.values())
        last_dt = max(dt_by_step.values())
        times = np.zeros(n_steps, dtype=np.float32)
        for i in range(n_steps):
            cur = dt_by_step.get(i)
            times[i] = (i * DATA2_TIMESTEP_SECONDS) if cur is None \
                else float((cur - first_dt).total_seconds())
        case_max = float((last_dt - first_dt).total_seconds())
        if case_max > times[-1]:
            times[-1] = case_max
    else:
        times = np.arange(n_steps, dtype=np.float32) * DATA2_TIMESTEP_SECONDS
        case_max = float(times[-1])

    meta = pd.DataFrame({
        "case_name": case_name,
        "source": "data2",
        "timestep_idx": np.arange(n_steps),
        "time_sec": times,
    })
    result = (np.nan_to_num(vib_steps), np.nan_to_num(feat_steps), times, case_max, meta)
    if use_cache:
        _save_case_cache(cache_path, *result)
    return result


def discover_tdms_inference_cases(test_dir: Path) -> List[Tuple[str, Path]]:
    """Find TDMS-only inference cases (no operation CSV) under data/Test."""
    test_dir = Path(test_dir)
    cases: List[Tuple[str, Path]] = []
    if not test_dir.exists():
        return cases
    for child in sorted(test_dir.iterdir()):
        if not child.is_dir():
            continue
        if any(child.glob("*.tdms")):
            cases.append((child.name, child))
            continue
        for grand in sorted(child.iterdir()):
            if grand.is_dir() and any(grand.glob("*.tdms")):
                cases.append((child.name, grand))
                break
    return cases


def load_original_inference_case(
    case_name: str,
    vibration_dir: Path,
    use_cache: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """TDMS inference case (no CSV) -> (vib, feat, times, meta) per chunk."""
    tdms_files = sorted(Path(vibration_dir).glob("*.tdms"))
    if not tdms_files:
        raise ValueError(f"No TDMS files in {vibration_dir}")

    key = _cache_key(tdms_files, f"orig_inf:{case_name}:{CACHE_VERSION}")
    cache_path = DATA2_FEATURE_CACHE_DIR / f"original_inference_{case_name}_{key}.npz"
    if use_cache and cache_path.exists():
        vib, feat, times, _case_max, meta = _load_case_cache(cache_path)
        return vib, feat, times, meta

    n_files = len(tdms_files)
    n_steps = n_files * TDMS_CHUNKS_PER_FILE

    vib_steps = np.zeros((n_steps, len(TDMS_CHANNELS), VIBRATION_FEATURES_PER_CHANNEL), dtype=np.float32)
    feat_steps = np.zeros((n_steps, len(TDMS_CHANNELS), HANDCRAFTED_DIM), dtype=np.float32)

    for file_idx, tdms_path in enumerate(tdms_files):
        chans = load_tdms_channels(tdms_path)
        normalized = {name.upper(): np.asarray(v, dtype=np.float32) for name, v in chans.items()}
        ch_arrays = [normalized.get(ch, np.array([], dtype=np.float32)) for ch in TDMS_CHANNELS]
        for chunk_idx in range(TDMS_CHUNKS_PER_FILE):
            gstep = file_idx * TDMS_CHUNKS_PER_FILE + chunk_idx
            start = chunk_idx * TDMS_CHUNK_SAMPLES
            end = start + TDMS_CHUNK_SAMPLES
            for ch_i, ch_arr in enumerate(ch_arrays):
                chunk = ch_arr[start:end] if ch_arr.size >= end else ch_arr[start:]
                vib_steps[gstep, ch_i] = stft_magnitude_vector(chunk)
                feat_steps[gstep, ch_i] = handcrafted_features_from_signal(chunk)

    times = np.arange(n_steps, dtype=np.float32) * TDMS_CHUNK_SECONDS
    meta = pd.DataFrame({
        "case_name": case_name,
        "source": "original_inference",
        "timestep_idx": np.arange(n_steps),
        "time_sec": times,
    })
    case_max = float(times[-1])
    result = (np.nan_to_num(vib_steps), np.nan_to_num(feat_steps), times, case_max, meta)
    if use_cache:
        _save_case_cache(cache_path, *result)
    return result[0], result[1], result[2], result[4]


# ============================================================================
# Sliding window builder
# ============================================================================


def _build_windows(
    vib: np.ndarray,
    feat_aug: np.ndarray,
    feat_raw: np.ndarray,
    times: np.ndarray,
    case_max: float,
    case_name: str,
    source: str,
    window_size: int,
    stride: int,
    max_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Return (X_vib, X_feat, hi, rul, metadata) per window.

    HI label is read at the END timestep of each window from a per-timestep HI
    sequence computed once per case (cheap and consistent across windows).
    feat_aug = augmented features (HI-augmented, used as model input).
    feat_raw = raw 10-dim features (used to compute damage HI label).
    """
    hi_per_step = compute_hi_labels(times, case_max, feat_raw=feat_raw)

    X_vib, X_feat, hi, rul, rows = [], [], [], [], []
    for start in range(0, len(times) - window_size + 1, stride):
        if max_samples is not None and len(hi) >= max_samples:
            break
        end = start + window_size
        current_time = float(times[end - 1])
        X_vib.append(vib[start:end])
        X_feat.append(feat_aug[start:end])
        rul.append(max(0.0, case_max - current_time))
        hi.append(float(hi_per_step[end - 1]))
        rows.append({
            "case_name": case_name,
            "source": source,
            "start_timestep": start,
            "end_timestep": end - 1,
            "time_sec": current_time,
            "case_max": case_max,
        })

    if not hi:
        raise ValueError(f"window_size {window_size} > timesteps in case {case_name}")
    return (
        np.asarray(X_vib, dtype=np.float32),
        np.asarray(X_feat, dtype=np.float32),
        np.asarray(hi, dtype=np.float32),
        np.asarray(rul, dtype=np.float32),
        pd.DataFrame(rows),
    )


# ============================================================================
# Top-level dataset loaders
# ============================================================================


def load_dataset(
    original_dir: Path = TRAIN_DIR,
    data2_dir: Path = DATA2_DIR,
    include_original: bool = True,
    include_data2: bool = True,
    window_size: int = 32,
    stride: int = 1,
    max_samples: Optional[int] = None,
    use_cache: bool = True,
    degradation_baseline: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
    """Load all training cases.

    Returns (X_vib, X_feat, hi, rul, metadata, degradation_baseline).
      - hi : HI labels in [0, 1] - the training target
      - rul: true RUL in seconds - kept alongside hi for evaluation use only
    """
    case_records: list[dict] = []

    if include_original:
        for case_name, op_csv, vib_dir in discover_cases(original_dir):
            vib, feat, times, case_max, _ = load_original_case_timesteps(
                case_name, op_csv, vib_dir, use_cache=use_cache,
            )
            case_records.append(dict(
                vib=vib, feat=feat, times=times, case_max=case_max,
                case_name=case_name, source="original",
            ))

    if include_data2:
        for case_dir in sorted(Path(data2_dir).glob("Train_No_*")):
            if not (case_dir / "vibration.parquet").exists():
                continue
            vib, feat, times, case_max, _ = load_data2_case_timesteps(case_dir, use_cache=use_cache)
            case_records.append(dict(
                vib=vib, feat=feat, times=times, case_max=case_max,
                case_name=case_dir.name, source="data2",
            ))

    if not case_records:
        raise ValueError("No cases found in original_dir or data2_dir.")

    if degradation_baseline is None:
        degradation_baseline = compute_global_baseline(
            [r["feat"] for r in case_records],
            n_baseline=DEGRADATION_BASELINE_TIMESTEPS,
        )
    degradation_baseline = np.asarray(degradation_baseline, dtype=np.float32)

    vibs, feats, his, ruls, metas = [], [], [], [], []
    for r in case_records:
        aug_feat = (
            augment_with_degradation(r["feat"], degradation_baseline)
            if HI_FEATURES_ENABLED else r["feat"].astype(np.float32)
        )
        Xv, Xf, hi, rul, m = _build_windows(
            r["vib"], aug_feat, r["feat"], r["times"], r["case_max"],
            r["case_name"], r["source"], window_size, stride, max_samples,
        )
        vibs.append(Xv); feats.append(Xf); his.append(hi); ruls.append(rul); metas.append(m)

    return (
        np.concatenate(vibs, axis=0),
        np.concatenate(feats, axis=0),
        np.concatenate(his, axis=0),
        np.concatenate(ruls, axis=0),
        pd.concat(metas, ignore_index=True),
        degradation_baseline,
    )


def load_inference_dataset(
    test_dir: Path,
    window_size: int,
    stride: int = 1,
    use_cache: bool = True,
    degradation_baseline: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load TDMS-only Test cases. Returns ALL sliding windows per case
    (stride=1 by default) so stage-2 can fit the HI trajectory.

    Returns (X_vib, X_feat, metadata).  No labels available at inference.
    Each metadata row identifies the window: case_name, end_timestep, time_sec.
    """
    if degradation_baseline is None:
        raise ValueError(
            "degradation_baseline is required for inference. "
            "Load it from the fold checkpoint and pass it through."
        )
    degradation_baseline = np.asarray(degradation_baseline, dtype=np.float32)

    cases = discover_tdms_inference_cases(test_dir)
    if not cases:
        raise ValueError(f"No TDMS inference cases under {test_dir}")

    vibs, feats, metas = [], [], []
    for case_name, vib_dir in cases:
        vib, feat, times, _ = load_original_inference_case(case_name, vib_dir, use_cache=use_cache)
        aug_feat = (
            augment_with_degradation(feat, degradation_baseline)
            if HI_FEATURES_ENABLED else feat.astype(np.float32)
        )
        n = len(vib)
        if n < window_size:
            raise ValueError(f"case {case_name}: only {n} TDMS chunks, need >= window_size={window_size}")
        for start in range(0, n - window_size + 1, stride):
            end = start + window_size
            vibs.append(vib[start:end][None, ...])
            feats.append(aug_feat[start:end][None, ...])
            metas.append({
                "case_name": case_name,
                "source": "original_inference",
                "start_timestep": start,
                "end_timestep": end - 1,
                "time_sec": float(times[end - 1]),
                "num_timesteps_total": n,
            })

    return (
        np.concatenate(vibs, axis=0).astype(np.float32),
        np.concatenate(feats, axis=0).astype(np.float32),
        pd.DataFrame(metas),
    )
