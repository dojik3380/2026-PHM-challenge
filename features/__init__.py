"""Physics-based feature extraction 모듈.

RPM trajectory estimation, RMS extraction, auxiliary feature vector 생성 등
TDMS 진동 데이터에서 물리 기반 특징을 추출하는 함수들을 제공한다.
"""

from .rpm_estimator import (
    estimate_rpm_from_signal,
    estimate_rpm_trajectory,
    extract_rms_from_channels,
    extract_auxiliary_vector,
)

from .vibration import (
    vibration_stft_timestep,
    augment_stft_features,
    augment_sequence_level,
    stft_magnitude_vector,
)

__all__ = [
    "estimate_rpm_from_signal",
    "estimate_rpm_trajectory",
    "extract_rms_from_channels",
    "extract_auxiliary_vector",
    "vibration_stft_timestep",
    "augment_stft_features",
    "augment_sequence_level",
    "stft_magnitude_vector",
]
