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
# Only RPM-INDEPENDENT features. The 10 RPM-dependent ones (BPFO/BPFI/BSF/FTF,
# F_1X..F_3456X) were dropped after the RPM ablation showed RPM was not a
# usable signal and inferred-RPM features just added noise.
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
)
HANDCRAFTED_DIM = len(HANDCRAFTED_FEATURES)
DEGRADATION_BASELINE_TIMESTEPS = 10                # first N timesteps used as healthy baseline
HI_FEATURES_ENABLED = True                          # 10-dim -> 30-dim: [rel, cummax_rel, damage_rel]
AUGMENTED_HANDCRAFTED_DIM = HANDCRAFTED_DIM * 3 if HI_FEATURES_ENABLED else HANDCRAFTED_DIM


# HI label generation ---------------------------------------------------------
# How the per-timestep HI ground-truth label is computed.
#   "linear" : HI = t / T_case                            (time-fraction, case-dependent)
#   "power"  : HI = (t/T_case) ** HI_LABEL_POWER          (bearing-like late ramp)
#   "damage" : HI = clip(RMS_growth_vs_baseline / scale, 0, 1)
#              physical degradation measure, case-invariant. Avoids the
#              short/long-lifetime outlier problem (Train3 / Train4) that
#              time-based labels suffer from.
#   "hybrid" : 0.5 * linear + 0.5 * damage                (best of both)
HI_LABEL_MODE = "hybrid"
HI_LABEL_POWER = 2.0
HI_DAMAGE_SCALE = 5.0        # RMS growth ratio that maps to HI=1.0 (tanh saturation)
HI_FAILURE_THRESHOLD = 0.9   # HI value treated as end-of-life in stage 2


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
EPOCHS = 25
BATCH_SIZE = 32
LEARNING_RATE = 5e-4
DROPOUT = 0.5
WEIGHT_DECAY = 1e-4
VIB_HIDDEN = 64
SCHEDULER_T0 = 12
EARLY_STOPPING_PATIENCE = 8

# Validation holdout (single-fold leave-one-TDMS-case-out).
RANDOM_VAL_CASE = True
VAL_CASE_DEFAULT = "Train2"
VAL_CASE_SEED = None


# Loss ------------------------------------------------------------------------
# Phase 1 hybrid: the model has two heads sharing one encoder.
#   - RUL head:  log-space regression, trained with Huber(log) + Asymmetric(real).
#                This is the production output -- it directly matches the
#                competition metric (which heavily rewards conservative
#                under-prediction) so we let it learn that bias.
#   - HI head:   sigmoid in [0, 1], trained with MSE on the hybrid HI label.
#                Acts as an auxiliary task -- forces the shared encoder to learn
#                degradation trajectory features (which a pure RUL head, on tiny
#                data, can ignore by falling into safe-low collapse).
#
# Total loss = RUL_LOSS_WEIGHT * RUL_loss + HI_LOSS_WEIGHT * HI_loss.
RUL_LOSS_WEIGHT = 0.7
HI_LOSS_WEIGHT = 0.3

# Internal RUL loss = HUBER_WEIGHT * Huber(log) + ASYMMETRIC_WEIGHT * Asymmetric(real).
HUBER_WEIGHT = 0.15
ASYMMETRIC_WEIGHT = 0.85
OVER_EST_PENALTY_SCALE = 20.0
UNDER_EST_PENALTY_SCALE = 50.0

# Post-inference scale applied to raw RUL seconds: pred_final = pred_raw * DENORM_SCALE.
# Tuned on OOF parquets from seed42 full-CV: raw preds are systematically low
# (model under-predicts due to asymmetric loss bias), so scale > 1.0 recovers signal.
# OOF sweep (30-dim relative features, seed42): 1.0→0.4215, 1.80→0.4673 (peak). Set to 1.80.
DENORM_SCALE = 1.80


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
