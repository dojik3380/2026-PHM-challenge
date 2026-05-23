"""PHM RUL 예측 파이프라인 설정."""

from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DIR = DATA_DIR / "Train"
TEST_DIR = DATA_DIR / "Test"
VALIDATION_DIR = DATA_DIR / "Validation"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

MODEL_PATH = MODELS_DIR / "RUL_Baseline.pt"
PREDICTION_PATH = RESULTS_DIR / "evaluation_predictions.csv"
TEAM_NAME = "PHM"
VALIDATION_PREDICTION_PATH = RESULTS_DIR / f"{TEAM_NAME}_validation.xlsx"

VIBRATION_CHANNELS = ("CH1", "CH2", "CH3", "CH4")

# ── 베어링 30306 물리 규격 ──────────────────────────────────────
BEARING_30306 = {
    "BPFI_ref": 140.0,
    "BPFO_ref": 93.0,
    "BSF_ref": 78.0,
    "Cage_ref": 6.7,
}

# ── RPM Estimation 설정 (하위 호환용 — rpm_estimator.py 내부에서는 사용 안 함) ──
RPM_SEARCH_RANGE = (5.0, 30.0)
RPM_ESTIMATION_WINDOW = 4096

# ── Auxiliary Feature 설정 ─────────────────────────────────────
AUXILIARY_FEATURES = ("rpm", "rms")
AUXILIARY_DIM = len(AUXILIARY_FEATURES)

# STFT 설정
SAMPLING_RATE = 25_600
STFT_NPERSEG = 1024
STFT_NOVERLAP = STFT_NPERSEG // 2
STFT_FREQ_BINS = STFT_NPERSEG // 2 + 1
VIBRATION_FEATURES_PER_CHANNEL = STFT_FREQ_BINS * 2  # mean + std (513×2=1026)

# 모델 학습 설정
WINDOW_SIZE = 32
STRIDE = 1

EPOCHS = 80
BATCH_SIZE = 16
LEARNING_RATE = 5e-4
TEST_SIZE = 0.2
RANDOM_STATE = 42
DROPOUT = 0.4
WEIGHT_DECAY = 0.0

PRETRAIN_LR = 3e-4
PRETRAIN_EPOCHS = 20
PRETRAIN_BATCH_SIZE = 32
PRETRAIN_WEIGHT_DECAY = 1e-4
SCHEDULER_T0 = 10
EARLY_STOPPING_PATIENCE = 15
AUGMENTATION_PROB = 0.35
MIXUP_ALPHA = 0.2     # 0 = mixup 비활성. >0 = Beta(alpha, alpha) sampling
MIXUP_PROB = 0.5      # batch별 mixup 적용 확률

# Loss 설정
HUBER_WEIGHT = 0.7
ASYMMETRIC_WEIGHT = 0.3
OVER_EST_PENALTY_SCALE = 20.0
UNDER_EST_PENALTY_SCALE = 50.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# STFT 캐시 설정
STFT_CACHE_DIR = PROJECT_ROOT / "stft_cache"
STFT_CACHE_ENABLED = True
STFT_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def clear_stft_cache():
    """STFT 캐시 디렉토리 비우기"""
    import shutil
    if STFT_CACHE_DIR.exists():
        shutil.rmtree(STFT_CACHE_DIR)
        STFT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Cleared STFT cache: {STFT_CACHE_DIR}")
    else:
        print("STFT cache directory does not exist")
