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

MODEL_PATH = MODELS_DIR / "RUL_Baseline.pt" # 모델 체크포인트명 설정
PREDICTION_PATH = RESULTS_DIR / "evaluation_predictions.csv"   # 예측 결과 저장명 설정
TEAM_NAME = "PHM"
VALIDATION_PREDICTION_PATH = RESULTS_DIR / f"{TEAM_NAME}_validation.xlsx"

VIBRATION_CHANNELS = ("CH1", "CH2", "CH3", "CH4")

# ── 베어링 30306 물리 규격 ──────────────────────────────────────
# 1000 RPM 기준 결함 주파수 (Hz). 실제 사용 시 RPM(t)/1000 으로 스케일링.
BEARING_30306 = {
    "BPFI_ref": 140.0,   # Ball Pass Frequency Inner race
    "BPFO_ref": 93.0,    # Ball Pass Frequency Outer race
    "BSF_ref": 78.0,     # Ball Spin Frequency
    "Cage_ref": 6.7,     # Cage (FTF) Frequency
}

# ── RPM Estimation 설정 ────────────────────────────────────────
RPM_SEARCH_RANGE = (5.0, 30.0)   # shaft frequency 탐색 범위 (Hz) → 300~1800 RPM
RPM_ESTIMATION_WINDOW = 4096     # RPM 추정용 FFT window size

# ── Auxiliary Feature 설정 ─────────────────────────────────────
AUXILIARY_FEATURES = ("rpm", "rms")
AUXILIARY_DIM = len(AUXILIARY_FEATURES)

# STFT 설정: 25.6 kHz 진동 신호를 주파수 영역으로 변환한다.
SAMPLING_RATE = 25_600
STFT_NPERSEG = 1024
STFT_NOVERLAP = STFT_NPERSEG // 2
STFT_FREQ_BINS = STFT_NPERSEG // 2 + 1
VIBRATION_FEATURES_PER_CHANNEL = STFT_FREQ_BINS * 2  # mean + std concatenated (513×2=1026)

# 모델 학습 설정
WINDOW_SIZE = 32
<<<<<<< HEAD
STRIDE = 4
<<<<<<< HEAD
EPOCHS = 100
=======
EPOCHS = 60
>>>>>>> test
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
TEST_SIZE = 0.2
RANDOM_STATE = 42
DROPOUT = 0.2
WEIGHT_DECAY = 1e-4  # L2 regularization strength (0.0: 끔)
AUGMENTATION_PROB = 0.3  # Data augmentation probability (0.0: 증강 끔, 0.3: 기본, 0.5: 강한 증강)
=======
STRIDE = 1          # 4→1: 케이스당 윈도우 수 ~24→~95개, 훈련 샘플 4배 증가

EPOCHS = 80         # 도메인 갭 적응에 더 많은 epoch 필요
BATCH_SIZE = 16
LEARNING_RATE = 5e-4  
TEST_SIZE = 0.2
RANDOM_STATE = 42
DROPOUT = 0.3
WEIGHT_DECAY = 0.0          # L2 regularization strength (fine-tuning)

PRETRAIN_LR = 3e-4          # pretrain 전용 학습률
PRETRAIN_EPOCHS = 20        # pretrain 전용 epoch
PRETRAIN_BATCH_SIZE = 32    # pretrain 전용 배치 크기
PRETRAIN_WEIGHT_DECAY = 1e-4  # pretrain 전용 L2 regularization
SCHEDULER_T0 = 10           # CosineAnnealingWarmRestarts 첫 주기 (pretrain/train 공용)
EARLY_STOPPING_PATIENCE = 15  # val_loss 개선 없을 때 조기 종료 기준 epoch (pretrain/train 공용)
AUGMENTATION_PROB = 0.1     # 0.3→0.1: 데이터 적을 때 과한 증강이 train/val 괴리 유발
>>>>>>> test2

# Loss 설정
<<<<<<< HEAD
HUBER_WEIGHT = 0.5 # Huber loss의 가중치
ASYMMETRIC_WEIGHT = 0.5 # Asymmetric loss의 가중치
OVER_EST_PENALTY_SCALE = 50.0 # 과대평가에 대한 패널티를 (변경금지)
UNDER_EST_PENALTY_SCALE = 20.0 # 과소평가에 대한 패널티를 (변경금지)
=======
HUBER_WEIGHT = 0.7 # Huber loss의 가중치
ASYMMETRIC_WEIGHT = 0.3 # Asymmetric loss의 가중치
OVER_EST_PENALTY_SCALE = 20.0 # 과대평가에 대한 패널티 (엄격함)
UNDER_EST_PENALTY_SCALE = 50.0 # 과소평가에 대한 패널티 (관대함)
>>>>>>> test

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# STFT 캐시 설정
STFT_CACHE_DIR = PROJECT_ROOT / "stft_cache"
STFT_CACHE_ENABLED = True  # True: 캐시 사용, False: 매번 계산
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
