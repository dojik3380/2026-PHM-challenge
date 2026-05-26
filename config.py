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
HI_FEATURES_ENABLED = True                          # 10-dim -> 30-dim: [rel, cummax_rel, damage_rel]
AUGMENTED_HANDCRAFTED_DIM = (HANDCRAFTED_DIM * 3 + 1) if HI_FEATURES_ENABLED else HANDCRAFTED_DIM


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


# Training --------------------------------------------------------------------
# Window=32 (5 min). HI is a local question, so we don't need 256-step context.
# Stride=4 was chosen empirically: stride=1 generated near-duplicate windows
# (consecutive windows differ by only one 10s timestep) which the model
# memorized in ~5 epochs and never generalized. Stride=4 (40s shift) decorrelates
# windows enough to avoid trivial memorization and is 4x faster to train.
WINDOW_SIZE = 32
STRIDE = 4
EPOCHS = 35
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
