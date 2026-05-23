from pathlib import Path

import numpy as np
import pandas as pd
from nptdms import TdmsFile
from scipy.signal import detrend, find_peaks, windows


PROJECT_ROOT = Path(__file__).resolve().parent
TRAIN_DIR = PROJECT_ROOT / "data" / "Train"
TRAIN1_OPERATION_CSV = TRAIN_DIR / "Train4_Operation.csv"
TRAIN1_VIBRATION_DIR = TRAIN_DIR / "Train4_Vibration" / "Train4_Vibration"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "rpm_estimation"
OUTPUT_CSV = OUTPUT_DIR / "Train4_estimated_rpm.csv"

SAMPLING_RATE = 25_600
TDMS_CHANNELS = ("CH1", "CH2", "CH3", "CH4")

# The challenge states that the machine runs around 700~950 RPM and changes
# speed at roughly one-hour intervals. Keep a small margin to avoid missing
# edge cases while reducing false low-frequency peak matches.
MIN_RPM = 600.0
MAX_RPM = 1_100.0


def read_operation_csv(csv_path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "cp949"):
        try:
            df = pd.read_csv(csv_path, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        df = pd.read_csv(csv_path)

    rename_map = {}
    for column in df.columns:
        clean = str(column).strip()
        if "Time" in clean:
            rename_map[column] = "time_sec"
        elif "Motor speed" in clean:
            rename_map[column] = "motor_speed_rpm"
        elif "Torque" in clean:
            rename_map[column] = "torque_nm"
        elif "Front" in clean:
            rename_map[column] = "temp_front_c"
        elif "Rear" in clean:
            rename_map[column] = "temp_rear_c"

    df = df.rename(columns=rename_map)
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.dropna(subset=["time_sec"]).sort_values("time_sec").reset_index(drop=True)


def read_tdms_channels(tdms_path: Path) -> dict[str, np.ndarray]:
    tdms = TdmsFile.read(tdms_path)
    channels = {}
    for group in tdms.groups():
        for channel in group.channels():
            if channel.name in TDMS_CHANNELS:
                channels[channel.name] = np.asarray(channel[:], dtype=np.float64)

    missing = sorted(set(TDMS_CHANNELS) - set(channels))
    if missing:
        raise ValueError(f"{tdms_path.name} is missing channels: {missing}")
    return channels


def operation_time_for_tdms_index(
    file_index: int,
    file_count: int,
    operation_start: float,
    operation_end: float,
) -> float:
    if file_count <= 1:
        return operation_end
    progress = file_index / (file_count - 1)
    return operation_start + progress * (operation_end - operation_start)


def nearest_operation_row(operation_df: pd.DataFrame, time_sec: float) -> pd.Series:
    nearest_index = (operation_df["time_sec"] - time_sec).abs().idxmin()
    return operation_df.loc[nearest_index]


def channel_spectrum(signal: np.ndarray, sampling_rate: int) -> tuple[np.ndarray, np.ndarray]:
    signal = np.asarray(signal, dtype=np.float64)
    signal = signal[np.isfinite(signal)]
    if signal.size < 4:
        raise ValueError("Signal is too short for FFT.")

    signal = detrend(signal, type="constant")
    window = windows.hann(signal.size, sym=False)
    spectrum = np.abs(np.fft.rfft(signal * window))
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / sampling_rate)
    return freqs, spectrum


def combined_low_frequency_spectrum(
    channels: dict[str, np.ndarray],
    sampling_rate: int = SAMPLING_RATE,
) -> tuple[np.ndarray, np.ndarray]:
    spectra = []
    freqs = None

    for channel_name in TDMS_CHANNELS:
        channel_freqs, spectrum = channel_spectrum(channels[channel_name], sampling_rate)
        freqs = channel_freqs if freqs is None else freqs
        spectra.append(spectrum)

    combined = np.mean(np.stack(spectra, axis=0), axis=0)
    return freqs, combined


def amplitude_at(freqs: np.ndarray, spectrum: np.ndarray, target_hz: float) -> float:
    if target_hz > freqs[-1]:
        return 0.0
    index = int(np.argmin(np.abs(freqs - target_hz)))
    return float(spectrum[index])


def estimate_rpm_from_spectrum(
    freqs: np.ndarray,
    spectrum: np.ndarray,
    min_rpm: float = MIN_RPM,
    max_rpm: float = MAX_RPM,
) -> tuple[float, float, float]:
    min_hz = min_rpm / 60.0
    max_hz = max_rpm / 60.0
    mask = (freqs >= min_hz) & (freqs <= max_hz)
    candidate_freqs = freqs[mask]

    if candidate_freqs.size == 0:
        raise ValueError("No FFT bins are available in the requested RPM range.")

    search_spectrum = spectrum[mask]
    peak_indices, _ = find_peaks(search_spectrum)
    if peak_indices.size == 0:
        peak_indices = np.arange(candidate_freqs.size)

    best_freq = float(candidate_freqs[0])
    best_score = -np.inf

    for peak_index in peak_indices:
        f = float(candidate_freqs[peak_index])
        score = (
            amplitude_at(freqs, spectrum, f)
            + 0.5 * amplitude_at(freqs, spectrum, 2.0 * f)
            + 0.25 * amplitude_at(freqs, spectrum, 3.0 * f)
        )
        if score > best_score:
            best_score = score
            best_freq = f

    local_noise_floor = float(np.median(search_spectrum)) + 1e-12
    confidence = float(best_score / local_noise_floor)
    estimated_rpm = best_freq * 60.0
    return estimated_rpm, best_freq, confidence


def estimate_train1_rpm(output_csv: Path = OUTPUT_CSV) -> Path:
    operation_df = read_operation_csv(TRAIN1_OPERATION_CSV)
    tdms_files = sorted(TRAIN1_VIBRATION_DIR.glob("*.tdms"))
    if not tdms_files:
        raise FileNotFoundError(f"No TDMS files found in {TRAIN1_VIBRATION_DIR}")

    operation_start = float(operation_df["time_sec"].min())
    operation_end = float(operation_df["time_sec"].max())
    rows = []

    print(f"Train1 TDMS files: {len(tdms_files)}")
    print(f"Operation time range: {operation_start:.3f} ~ {operation_end:.3f} sec")
    print(f"RPM search range: {MIN_RPM:.0f} ~ {MAX_RPM:.0f} RPM")

    for file_index, tdms_path in enumerate(tdms_files):
        mapped_time = operation_time_for_tdms_index(
            file_index=file_index,
            file_count=len(tdms_files),
            operation_start=operation_start,
            operation_end=operation_end,
        )
        operation_row = nearest_operation_row(operation_df, mapped_time)

        channels = read_tdms_channels(tdms_path)
        freqs, spectrum = combined_low_frequency_spectrum(channels)
        estimated_rpm, peak_freq_hz, confidence = estimate_rpm_from_spectrum(freqs, spectrum)

        true_rpm = float(operation_row["motor_speed_rpm"])
        rows.append(
            {
                "tdms_file": tdms_path.name,
                "tdms_file_index": file_index + 1,
                "mapped_operation_time_sec": mapped_time,
                "nearest_operation_time_sec": float(operation_row["time_sec"]),
                "true_motor_speed_rpm": true_rpm,
                "estimated_rpm": estimated_rpm,
                "rpm_error": estimated_rpm - true_rpm,
                "abs_rpm_error": abs(estimated_rpm - true_rpm),
                "peak_freq_hz": peak_freq_hz,
                "confidence": confidence,
            }
        )
        print(
            f"{tdms_path.name}: true={true_rpm:.1f}, "
            f"estimated={estimated_rpm:.1f}, peak={peak_freq_hz:.3f} Hz"
        )

    result_df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_csv, index=False)

    mae = float(result_df["abs_rpm_error"].mean())
    corr = float(result_df[["true_motor_speed_rpm", "estimated_rpm"]].corr().iloc[0, 1])
    print(f"\nSaved: {output_csv}")
    print(f"Rows: {len(result_df):,}")
    print(f"MAE: {mae:.3f} RPM")
    print(f"Correlation: {corr:.4f}")
    return output_csv


if __name__ == "__main__":
    estimate_train1_rpm()
