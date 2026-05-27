"""PHM Health-Indicator (HI) prediction pipeline config.

Stage 1 (this codebase): a window-level neural network regresses HI in [0, 1].
Stage 2 (inference.py):   HI sequence is curve-fit per case and extrapolated to
                          the failure threshold to compute RUL.

This design replaces direct RUL regression (which was capped by the
information-theoretic floor of a single short window vs. an 80,000 s lifetime).
HI is a local question (degradation level NOW); RUL becomes simple post-hoc
statistics. Asymmetric loss is only applied at evaluation time.
"""

from pathlib import Path

import torch


# Paths -----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DIR = DATA_DIR / "Train"
TEST_DIR = DATA_DIR / "Test"
VALIDATION_DIR = DATA_DIR / "Validation"

# KIMM data2 (parquet-keyed cases). Layout: Train_No_*/vibration.parquet.
DATA2_DIR = PROJECT_ROOT / "data2"
DATA2_FEATURE_CACHE_DIR = PROJECT_ROOT / "data2_features"

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

TEAM_NAME = "PHM"


# Vibration / bearing ---------------------------------------------------------
VIBRATION_CHANNELS = ("CH1", "CH2", "CH3", "CH4")


# Handcrafted features --------------------------------------------------------
# 10 RPM-independent statistics + 6 bearing fault frequency amplitudes.
# Fault freq amplitudes are computed from each chunk's own FFT-estimated RPM
# (not the operation CSV), so they are accurately aligned per 10-second chunk.
# Previous RPM ablation failed because it used the operation CSV motor_speed_rpm
# which has coarse temporal resolution and per-chunk misalignment.
HANDCRAFTED_FEATURES = (
    "RMS",
    "PEAK_TO_PEAK",
    "ABS_MEAN",
    "SKEW",
    "KURT",
    "CREST",
    "IMPULSE",
    "SHAPE",
    "ABS_MAX",
    "RMS_HIGH",
    # Bearing fault frequency amplitudes (RPM estimated per chunk via FFT)
    "BPFI_1X",
    "BPFI_2X",
    "BPFO_1X",
    "BPFO_2X",
    "BSF_1X",
    "FTF_1X",
    # Shaft harmonic order energy (1×/2×/3× shaft frequency)
    # Order-domain normalization: freq_hz = mult × shaft_hz, RPM-invariant
    "SHAFT_1X",
    "SHAFT_2X",
    "SHAFT_3X",
)
HANDCRAFTED_DIM = len(HANDCRAFTED_FEATURES)
DEGRADATION_BASELINE_TIMESTEPS = 10                # first N timesteps used as healthy baseline
HI_FEATURES_ENABLED = True                          # 19-dim -> 57-dim: [rel, cummax_rel, damage_rel]
AUGMENTED_HANDCRAFTED_DIM = (HANDCRAFTED_DIM * 3) if HI_FEATURES_ENABLED else HANDCRAFTED_DIM


# HI label generation ---------------------------------------------------------
# How the per-timestep HI ground-truth label is computed.
#   "linear" : HI = t / T_case                            (time-fraction, case-dependent)
#   "power"  : HI = (t/T_case) ** HI_LABEL_POWER          (bearing-like late ramp)
#   "damage" : HI = clip(RMS_growth_vs_baseline / scale, 0, 1)
#              physical degradation measure, case-invariant. Avoids the
#              short/long-lifetime outlier problem (Train3 / Train4) that
#              time-based labels suffer from.
#   "hybrid" : 0.5 * linear + 0.5 * damage                (best of both)
HI_LABEL_MODE = "exponential"
HI_ALPHA = 4.0               # degradation acceleration: HI(t) = (e^(α·t/T)-1)/(e^α-1)
HI_LABEL_POWER = 2.0
HI_DAMAGE_SCALE = 5.0        # RMS growth ratio that maps to HI=1.0 (tanh saturation)
HI_FAILURE_THRESHOLD = 1.0   # Stage 2 failure threshold (exponential label ends at 1.0)
HI_SMOOTH_WINDOW = 11        # moving-average window for cumulative-damage HI smoothing


# Bearing fault frequency multipliers — 30306 tapered roller bearing
# Reference: spec sheet at 1000 RPM → BPFI=140Hz, BPFO=93Hz, BSF=78Hz, FTF=6.7Hz
# mult = freq_hz / (1000/60).  At any RPM: fault_hz = MULT * RPM/60.
BEARING_BPFI_MULT = 8.40    # 140 / 16.667
BEARING_BPFO_MULT = 5.58    # 93  / 16.667
BEARING_BSF_MULT  = 4.68    # 78  / 16.667
BEARING_FTF_MULT  = 0.402   # 6.7 / 16.667
BEARING_RPM_MIN     = 600.0
BEARING_RPM_MAX     = 1_100.0
BEARING_RPM_DEFAULT = 800.0  # fallback when chunk is too short for RPM estimation


# STFT ------------------------------------------------------------------------
SAMPLING_RATE = 25_600
STFT_NPERSEG = 1024
STFT_NOVERLAP = STFT_NPERSEG // 2
STFT_FREQ_BINS = STFT_NPERSEG // 2 + 1
VIBRATION_FEATURES_PER_CHANNEL = STFT_FREQ_BINS * 2  # mean + std (513 x 2 = 1026)


# TDMS chunking (1 timestep = 10 s = 256k samples, matches data2 grid) -------
TDMS_CHUNK_SAMPLES = 256_000
TDMS_CHUNK_SECONDS = 10.0
TDMS_CHUNKS_PER_FILE = 6                             # 60 s TDMS file -> 6 chunks

# TDMS 측정 cycle: 1분 측정 + 9분 휴식 = 10분 wall-clock cycle.
# 한 TDMS file = 1분 측정 = 6 chunks (10s 단위). 이후 9분간 측정 없음.
# wall-clock(chunk i) = (i // 6) × TDMS_CYCLE_SECONDS + (i % 6) × TDMS_CHUNK_SECONDS
# 등간격 linspace 로 잡으면 file 내 6 chunks 의 t 가 약 10배 펴져 HI/stage 라벨이 부정확해짐.
TDMS_CYCLE_SECONDS = 600.0                            # 10분 cycle


# Training --------------------------------------------------------------------
# Window=32 (5 min). HI is a local question, so we don't need 256-step context.
# Stride=4 was chosen empirically: stride=1 generated near-duplicate windows
# (consecutive windows differ by only one 10s timestep) which the model
# memorized in ~5 epochs and never generalized. Stride=4 (40s shift) decorrelates
# windows enough to avoid trivial memorization and is 4x faster to train.
WINDOW_SIZE = 32
STRIDE = 4
EPOCHS = 40
BATCH_SIZE = 32
LEARNING_RATE = 2e-4
DROPOUT = 0.5
WEIGHT_DECAY = 1e-4
VIB_HIDDEN = 64
SCHEDULER_T0 = 12
EARLY_STOPPING_PATIENCE = 12

# Validation holdout (single-fold leave-one-TDMS-case-out).
RANDOM_VAL_CASE = True
VAL_CASE_DEFAULT = "Train2"
VAL_CASE_SEED = None   # None → truly random each run; set an int for reproducibility


# Loss ------------------------------------------------------------------------
# Training loss (Stage-1 HI prediction only — RUL comes from Stage-2 curve fitting)
#   L = HI_LOSS_WEIGHT * Huber(HI, δ=HUBER_DELTA, late×LATE_LIFE_WEIGHT) + HI_RANK_LOSS_WEIGHT * HIPairwiseRanking
#   - Huber(HI): 후반부 20% 구간 W_i=LATE_LIFE_WEIGHT 가중치 적용 (고장 직전 궤적 정밀도 향상)
#   - HIPairwiseRanking: pred_hi[후기] > pred_hi[전기] 단조성 강제 (상수 붕괴 방지)
#
# A_RUL (Stage-2 기반) is used ONLY in validation reporting, not in training.
HI_LOSS_WEIGHT      = 0.75
HI_RANK_LOSS_WEIGHT = 0.25
HUBER_DELTA         = 0.3
LATE_LIFE_WEIGHT    = 8.0

# A_RUL evaluation penalty scales — evaluation only, not used in training loss.
OVER_EST_PENALTY_SCALE  = 20.0
UNDER_EST_PENALTY_SCALE = 50.0

# DENORM_SCALE is retired. Kept at 1.0 so existing checkpoints still load.
DENORM_SCALE = 1.0

# Post-hoc conservative calibration at inference (multiply final RUL by this).
# 1.0 = neutral. Tune on OOF: sweep [0.6, 1.0] and pick α that maximises A_RUL.
CALIBRATION_SHRINK = 1.0


# Stage 2 (RUL Estimation) ----------------------------------------------------
STAGE2_FALLBACK_RUL_CAP   = 200_000.0   # 최후 fallback (변화점 못 찾고 신호 없을 때)
STAGE2_RUL_BEFORE_FDP     = 30_000.0    # FCP 발견 전 기본 RUL (steady stage)
STAGE2_FCP_MIN_HISTORY    = 20          # AIC 가동 최소 점 수
STAGE2_FCP_MIN_SEGMENT    = 15          # 각 stage 최소 점 수 (false-positive FCP 억제)
STAGE2_AIC_C_ALPHA        = 8.0        # 임계 ↑ : noise 데이터에서 false-positive FCP 억제

STAGE2_KF_Q               = 1e-4        # process noise variance (K_ss≈0.31). 5e-5 까지 줄였더니 over-smooth.
STAGE2_KF_R               = 5e-4        # measurement noise variance
STAGE2_KF_P0              = 1.0         # 초기 covariance
STAGE2_KF_OUTLIER_LO      = 0.10        # raw HI < 이 값으로 dip 하면 측정 update skip
STAGE2_KF_OUTLIER_STATE   = 0.20        # state > 이 값일 때만 outlier 판정 (초기 단계 정상 0 보호)

STAGE2_HI_LOG_EPS         = 1e-2        # log 변환 안정성. D ≈ 4.605.
STAGE2_RUL_QUANTILE_K     = 0.5         # 보수 quantile (0.5 ≈ 30% lower bound, k=1.0은 너무 공격적)
STAGE2_RUL_MAX_CV         = 1.0         # σ_RUL / l_mean 가 이보다 크면 신뢰 부족 → reject
STAGE2_BETA_MIN_FACTOR    = 1.0         # β > _BETA_MIN_FACTOR / lifetime 이상이어야 의미있는 drift

STAGE2_WLS_RECENCY_DECAY  = 5.0
STAGE2_WLS_ROLLING_WINDOW = 50           # FCP 이후 최근 N 점만 사용 (30 은 짧음, β 추정 noisy)

# FDP threshold: 정적 0.15 → **동적 (baseline noise 기반)** (학계 표준 3σ rule)
# 각 case 의 첫 N window 의 HI_kf mean+std 로 진입 임계 결정.
# Nguyen et al. 2025, Wang & Xiang 2021 등 사용. fold 별 HI 분포 변동에 강건.
STAGE2_FDP_BASELINE_N = 30               # 첫 N window 로 baseline 통계 계산
STAGE2_FDP_K_SIGMA    = 3.0              # mean + k·σ
STAGE2_FDP_MIN        = 0.08             # 최소 임계 (지나치게 낮은 noise spike 차단)
STAGE2_FDP_MAX        = 0.30             # 최대 임계 (FDP 너무 늦어지면 lifetime 대부분 fallback)
STAGE2_FDP_THRESHOLD  = 0.15             # legacy static fallback (lifetime_prior 없을 때만)

STAGE2_FCP_MIN_RATIO      = 0.10         # increment 인덱스 비율로 너무 이른 FCP 거부
STAGE2_FCP_MAX_RATIO      = 0.95         # 너무 늦은 FCP 도 거부 (외삽 데이터 부족)

STAGE2_ROLLING_WINDOW     = 30          # legacy: train.py plot 호환

# Stage 2 — log-linear curve fit (사용자 수정 stage2.py 의 magic number 들)
STAGE2_MACRO_ROLLING      = 300          # macro fitting 광역 윈도우 (50분, 노이즈 스파이크 저항)
STAGE2_LOG_EPS            = 1e-6         # log(HI) 안정화용 epsilon
STAGE2_BETA_MIN_FIT       = 1e-8         # log-linear slope 이 이보다 작으면 reject
STAGE2_FDP_FALLBACK_RUL   = 200_000.0     # FDP 미도달 시 global linear fit 의 safe upper bound (Long-life outlier 대응)
STAGE2_FDP_MIN_SLOPE      = 1e-8         # FDP 미도달 global fit 의 slope 임계

# Stage 1 — per-case local normalization (Train4 OOD 해소)
STAGE1_NORMALIZE_N_BASELINE = DEGRADATION_BASELINE_TIMESTEPS  # 첫 N window 가 healthy

# Stage 2 — Lifetime prior (training fold lifetime distribution)
# Track 1 fallback 의 60000s hardcoded 를 *학습 fold lifetime 의 lognormal MRL* 로 교체.
# 학계 표준 (Si et al. 2011 review). LOCO 각 fold 의 train cases lifetime fit → prior.
# inference 시 conditional MRL(t_now) = E[T-t|T>t] 사용.
STAGE2_LIFETIME_PRIOR_DIST = "lognormal"   # 분포 family
STAGE2_LIFETIME_PRIOR_QUANTILE = 0.35      # 보수적인 분위수(35%) 적용 (Risk-averse)
STAGE2_FALLBACK_FLOOR      = 2_000.0       # CQRL fallback 의 최소값 (degenerate prior 방지)

# Stage 1 — Degradation Stage Classification (auxiliary multi-task head)
# absolute lifetime position 학습 → case fingerprint representation
# 4-class CE: lifetime fraction 을 경계로 양자화
STAGE_CE_WEIGHT     = 0.3                           # loss 비중 (HI Huber + Rank 와 함께)
STAGE_BOUNDARIES    = (0.40, 0.70, 0.90)            # healthy/incipient/fault/severe (quantile, late-loaded)
STAGE_NUM_CLASSES   = len(STAGE_BOUNDARIES) + 1     # = 4


# Runtime ---------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA2_FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def clear_feature_cache():
    """Delete the per-case NPZ feature cache (STFT + handcrafted + meta)."""
    import shutil
    if DATA2_FEATURE_CACHE_DIR.exists():
        shutil.rmtree(DATA2_FEATURE_CACHE_DIR)
        DATA2_FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Cleared feature cache: {DATA2_FEATURE_CACHE_DIR}")
    else:
        print("Feature cache directory does not exist")
