"""Physics-based STFT RUL 예측용 데이터 로딩.

Operation CSV 의존성을 제거하고, TDMS vibration에서
STFT 특징 + auxiliary(RPM, RMS) temporal sequence를 생성한다.

Train/Test 모두 동일한 TDMS-only 파이프라인을 사용하여
modality mismatch를 완전 해소한다.
"""

from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from nptdms import TdmsFile

from config import AUXILIARY_DIM, TRAIN_DIR
from features import vibration_stft_timestep
from features.rpm_estimator import extract_auxiliary_vector


CSV_ENCODINGS = ("utf-8-sig", "utf-8", "cp949")


def load_tdms_file(file_path):
    """작은 TDMS 테스트용 기본 로딩. 대용량 파일에는 사용하지 않는다."""
    tdms_file = TdmsFile.read(file_path)
    df = tdms_file.as_dataframe()
    return df


def load_tdms_channels(file_path):
    """대용량 대응 TDMS 로딩. 파일을 dataframe으로 만들지 않고 채널만 읽는다."""
    tdms_file = TdmsFile.read(file_path)
    data = {}
    for group in tdms_file.groups():
        for channel in group.channels():
            data[channel.name] = channel[:]
    return data


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
    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]
    for col in columns:
        col_lower = str(col).strip().lower()
        if any(candidate.lower() in col_lower for candidate in candidates):
            return col
    return None


def load_operation_csv(csv_path: Path) -> pd.DataFrame:
    """운전 CSV를 읽고 컬럼명을 표준화한다. RUL label 계산에만 사용."""
    df = _read_csv(csv_path)
    mapping = {
        _find_column(df.columns, ("time_sec", "time", "sec")): "time_sec",
        _find_column(df.columns, ("torque",)): "torque",
        _find_column(df.columns, ("speed", "rpm")): "speed",
        _find_column(df.columns, ("temp_front", "front")): "temp_front",
        _find_column(df.columns, ("temp_rear", "rear")): "temp_rear",
    }
    mapping = {old: new for old, new in mapping.items() if old is not None}
    df = df.rename(columns=mapping)

    if "time_sec" not in df.columns:
        raise ValueError(f"{csv_path} must contain time_sec or an equivalent time column.")

    for column in ("time_sec",):
        if column not in df.columns:
            df[column] = 0.0
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)

    return df.sort_values("time_sec").reset_index(drop=True)


def compute_rul(operation_df: pd.DataFrame, current_time: float) -> float:
    """RUL = max_time - current_time."""
    return float(operation_df["time_sec"].max() - current_time)


def _case_name_from_operation(csv_path: Path) -> str:
    return csv_path.stem.replace("_Operation", "")


def _find_vibration_dir(case_name: str, root_dir: Path) -> Optional[Path]:
    candidates = [
        root_dir / f"{case_name}_Vibration" / f"{case_name}_Vibration",
        root_dir / f"{case_name}_Vibration",
        root_dir / case_name / case_name,
        root_dir / case_name,
    ]
    for candidate in candidates:
        if candidate.exists() and any(candidate.glob("*.tdms")):
            return candidate
    for candidate in sorted(root_dir.rglob(f"{case_name}*")):
        if candidate.is_dir() and any(candidate.glob("*.tdms")):
            return candidate
    return None


def discover_cases(root_dir: Path) -> List[Tuple[str, Path, Path]]:
    """운전 CSV와 TDMS 진동 폴더가 모두 있는 case를 찾는다.
    
    Train 시에는 RUL label을 위해 operation CSV가 필요하다.
    (operation CSV에서 max_time 정보를 가져와 RUL을 계산)
    """
    root_dir = Path(root_dir)
    cases = []
    for csv_path in sorted(root_dir.glob("*_Operation.csv")):
        case_name = _case_name_from_operation(csv_path)
        vibration_dir = _find_vibration_dir(case_name, root_dir)
        if vibration_dir is not None:
            cases.append((case_name, csv_path, vibration_dir))
    return cases


def discover_vibration_cases(root_dir: Path) -> List[Tuple[str, Path]]:
    """Find validation/test cases that contain TDMS files without requiring operation CSVs."""
    root_dir = Path(root_dir)
    cases = []

    for child in sorted(root_dir.iterdir()):
        if not child.is_dir():
            continue
        vibration_dir = _find_vibration_dir(child.name, root_dir)
        if vibration_dir is not None:
            cases.append((child.name, vibration_dir))

    if cases:
        return cases

    if any(root_dir.glob("*.tdms")):
        return [(root_dir.name, root_dir)]

    return []


def _time_from_tdms_index(index: int, max_time: float, total_files: int) -> float:
    if total_files <= 1:
        return max_time
    return max_time * index / float(total_files - 1)


def _extract_auxiliary_from_tdms(tdms_path: Path) -> np.ndarray:
    """TDMS 파일 1개에서 auxiliary feature [RPM, RMS]를 추출한다."""
    channel_data = load_tdms_channels(tdms_path)
    return extract_auxiliary_vector(channel_data)


def _case_timesteps(
    operation_csv: Path,
    vibration_dir: Path,
    max_files: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    TDMS 파일별 timestep 생성.
    진동 shape: (num_steps, 4, freq_bins)
    auxiliary shape: (num_steps, auxiliary_dim)
    """
    operation_df = load_operation_csv(operation_csv)
    max_time = float(operation_df["time_sec"].max())
    all_tdms_files = sorted(Path(vibration_dir).glob("*.tdms"))
    if not all_tdms_files:
        raise ValueError(f"No TDMS files found in {vibration_dir}")

    total_files = len(all_tdms_files)
    tdms_files = all_tdms_files[:max_files] if max_files is not None else all_tdms_files

    vibration_steps = []
    auxiliary_steps = []
    times = []

    for index, tdms_path in enumerate(tdms_files):
        current_time = _time_from_tdms_index(index, max_time, total_files)
        vibration_steps.append(vibration_stft_timestep(tdms_path))
        auxiliary_steps.append(_extract_auxiliary_from_tdms(tdms_path))
        times.append(current_time)

    return (
        np.asarray(vibration_steps, dtype=np.float32),
        np.asarray(auxiliary_steps, dtype=np.float32),
        np.asarray(times, dtype=np.float32),
    )


def build_case_sequences(
    operation_csv: Path,
    vibration_dir: Path,
    case_name: str,
    window_size: int,
    stride: int,
    max_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """정렬된 진동/auxiliary timestep에 sliding window를 적용한다."""
    operation_df = load_operation_csv(operation_csv)
    max_files = None
    if max_samples is not None:
        max_files = window_size + max(0, max_samples - 1) * stride

    vibration_steps, auxiliary_steps, times = _case_timesteps(
        operation_csv,
        vibration_dir,
        max_files=max_files,
    )

    X_vibration = []
    X_auxiliary = []
    y = []
    metadata = []

    for start in range(0, len(times) - window_size + 1, stride):
        if max_samples is not None and len(y) >= max_samples:
            break
        end = start + window_size
        current_time = float(times[end - 1])
        X_vibration.append(vibration_steps[start:end])
        X_auxiliary.append(auxiliary_steps[start:end])
        y.append(compute_rul(operation_df, current_time))
        metadata.append({"case_name": case_name, "start_time": float(times[start]), "time_sec": current_time})

    if not y:
        raise ValueError(f"Window size {window_size} is larger than available TDMS files in {vibration_dir}")

    return (
        np.asarray(X_vibration, dtype=np.float32),
        np.asarray(X_auxiliary, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
        pd.DataFrame(metadata),
    )


def build_inference_sequence(
    vibration_dir: Path,
    case_name: str,
    window_size: int,
    max_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build one validation/test sequence from the latest available TDMS window.
    
    Operation CSV 없이 TDMS만으로 vibration + auxiliary를 생성한다.
    """
    tdms_files = sorted(Path(vibration_dir).glob("*.tdms"))
    if max_samples is not None:
        tdms_files = tdms_files[:max_samples]
    if len(tdms_files) < window_size:
        raise ValueError(f"Window size {window_size} is larger than available TDMS files in {vibration_dir}")

    selected_files = tdms_files[-window_size:]
    vibration_steps = []
    auxiliary_steps = []

    for tdms_path in selected_files:
        vibration_steps.append(vibration_stft_timestep(tdms_path))
        auxiliary_steps.append(_extract_auxiliary_from_tdms(tdms_path))

    X_vibration = np.asarray([vibration_steps], dtype=np.float32)
    X_auxiliary = np.asarray([auxiliary_steps], dtype=np.float32)
    metadata = pd.DataFrame(
        [
            {
                "case_name": case_name,
                "start_file": selected_files[0].stem,
                "end_file": selected_files[-1].stem,
                "num_files": len(tdms_files),
            }
        ]
    )
    return X_vibration, X_auxiliary, metadata


def apply_data_augmentation(X_vib_batch: np.ndarray, X_aux_batch: np.ndarray, y_batch: np.ndarray,
                          aug_prob: float = 0.3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """통합 데이터 증강 적용 (훈련 시에만 사용)

    Auxiliary(RPM/RMS)는 물리적 값이므로 시간축 조작만 적용하고
    값 자체는 변경하지 않는다.
    """
    from features import augment_stft_features, augment_sequence_level

    N = len(y_batch)
    # 최대 4배 증강(원본 + STFT aug + 시퀀스 aug 최대 2개)을 가정해 pre-allocate
    MAX_FACTOR = 4
    vib_buf = np.empty((N * MAX_FACTOR, *X_vib_batch.shape[1:]), dtype=np.float32)
    aux_buf = np.empty((N * MAX_FACTOR, *X_aux_batch.shape[1:]), dtype=np.float32)
    y_buf   = np.empty(N * MAX_FACTOR, dtype=np.float32)
    idx = 0

    for vib_seq, aux_seq, y in zip(X_vib_batch, X_aux_batch, y_batch):
        # 원본
        vib_buf[idx] = vib_seq
        aux_buf[idx] = aux_seq
        y_buf[idx]   = y
        idx += 1

        # STFT 특징 증강
        vib_stft_aug = vib_seq.copy()
        for ch in range(vib_seq.shape[1]):
            channel_stft = vib_seq[:, ch, :].T  # (freq_bins, window_size)
            vib_stft_aug[:, ch, :] = augment_stft_features(channel_stft, aug_prob).T
        vib_buf[idx] = vib_stft_aug
        aux_buf[idx] = aux_seq
        y_buf[idx]   = y
        idx += 1

        # 시퀀스 레벨 증강
        seq_augmented = augment_sequence_level(vib_seq, aux_seq, y, aug_prob)
        for vib_aug, aux_aug, y_aug in seq_augmented[1:]:
            if idx >= len(y_buf):
                break
            vib_buf[idx] = vib_aug
            aux_buf[idx] = aux_aug
            y_buf[idx]   = y_aug
            idx += 1

    return vib_buf[:idx], aux_buf[:idx], y_buf[:idx]


def load_inference_dataset(
    root_dir: Path,
    window_size: int = 32,
    max_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load validation/test TDMS-only cases for RUL submission inference."""
    vib_batches = []
    aux_batches = []
    metadata_frames = []

    for case_name, vibration_dir in discover_vibration_cases(root_dir):
        X_vib, X_aux, metadata = build_inference_sequence(
            vibration_dir=vibration_dir,
            case_name=case_name,
            window_size=window_size,
            max_samples=max_samples,
        )
        vib_batches.append(X_vib)
        aux_batches.append(X_aux)
        metadata_frames.append(metadata)

    if not vib_batches:
        raise ValueError(f"No TDMS validation/test cases found in {root_dir}")

    return (
        np.concatenate(vib_batches, axis=0),
        np.concatenate(aux_batches, axis=0),
        pd.concat(metadata_frames, ignore_index=True),
    )


def load_dataset(
    root_dir: Path = TRAIN_DIR,
    window_size: int = 32,
    stride: int = 4,
    max_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    전체 case를 로딩한다.
    X_vibration: (batch, seq_len, 4, freq_bins)
    X_auxiliary: (batch, seq_len, auxiliary_dim)
    y: (batch,)
    """
    vib_batches = []
    aux_batches = []
    targets = []
    metadata_frames = []

    for case_name, operation_csv, vibration_dir in discover_cases(root_dir):
        loaded = sum(len(batch) for batch in targets)
        remaining = None if max_samples is None else max_samples - loaded
        if remaining is not None and remaining <= 0:
            break

        X_vib, X_aux, y, metadata = build_case_sequences(
            operation_csv=operation_csv,
            vibration_dir=vibration_dir,
            case_name=case_name,
            window_size=window_size,
            stride=stride,
            max_samples=remaining,
        )
        vib_batches.append(X_vib)
        aux_batches.append(X_aux)
        targets.append(y)
        metadata_frames.append(metadata)

    if not vib_batches:
        raise ValueError(f"No cases with operation CSV + TDMS files found in {root_dir}")

    return (
        np.concatenate(vib_batches, axis=0),
        np.concatenate(aux_batches, axis=0),
        np.concatenate(targets, axis=0),
        pd.concat(metadata_frames, ignore_index=True),
    )
