"""Health-Indicator (degradation) features.

Augments per-timestep handcrafted features with four normalized views:

  - relative              = feat / baseline          : current vs healthy ratio
  - cummax_ratio          = cummax(feat) / baseline  : worst-so-far (monotonic, late-life)
  - cumulative_damage     = cumsum(max(feat-baseline,0)) / baseline  : per-feature damage
  - energy_cumdamage      = cumsum(max(Energy-1,0))  : cross-product damage scalar per channel
                            Energy(t) = (RMS(t)/RMS_base) × (Kurt(t)/Kurt_base)
                            Spec §2.2: fault specificity — both RMS AND kurtosis must be
                            elevated simultaneously, suppressing broadband noise false alarms.

For inference (Test cases with no healthy baseline available), the GLOBAL
baseline computed from training data is reused.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from config import HANDCRAFTED_FEATURES


EPS = 1e-6

_RMS_IDX  = list(HANDCRAFTED_FEATURES).index("RMS")
_KURT_IDX = list(HANDCRAFTED_FEATURES).index("KURT")


def compute_case_baseline(feat: np.ndarray, n_baseline: int = 10) -> np.ndarray:
    """feat: (T, C, F) → (C, F)  baseline from first n_baseline healthy timesteps."""
    if feat.ndim != 3:
        raise ValueError(f"expected (T, C, F), got {feat.shape}")
    n = min(int(n_baseline), feat.shape[0])
    if n <= 0:
        return np.zeros(feat.shape[1:], dtype=np.float32)
    return feat[:n].mean(axis=0).astype(np.float32)


def compute_global_baseline(
    per_case_feats: Sequence[np.ndarray],
    n_baseline: int = 10,
) -> np.ndarray:
    """Per-case baselines averaged → global (C, F) baseline.

    Uses the mean of per-case baselines (not a flat mean over all chunks) so
    each case contributes equally regardless of length.
    """
    baselines = [compute_case_baseline(f, n_baseline) for f in per_case_feats]
    if not baselines:
        raise ValueError("no cases to compute baseline from")
    return np.mean(np.stack(baselines, axis=0), axis=0).astype(np.float32)


def augment_with_degradation(
    feat: np.ndarray,
    baseline: np.ndarray,
    eps: float = EPS,
) -> np.ndarray:
    """feat: (T, C, F), baseline: (C, F) → (T, C, F*3+1).

    Returns [rel, cummax_rel, cumulative_damage_rel, energy_cumdamage]:
      - rel, cummax_rel, cumulative_damage_rel: per-feature relative views (F*3)
      - energy_cumdamage (1 scalar per channel): cross-product degradation per spec §2.2
          Energy(t) = (RMS(t)/RMS_base) × (Kurt(t)/Kurt_base)
          energy_cumdamage(t) = Σ_{i≤t} max(Energy(i) - 1, 0)
        Forces BOTH RMS AND kurtosis to be elevated → fault-specific, suppresses noise.

    Raw absolute features excluded: inter-case invariance (Train4 kurtosis baseline ≈68
    vs ≈3.5 for others caused OOD collapse when raw features were included).
    Uses abs(baseline) so signed features (skew) don't flip ratios.
    """
    if feat.ndim != 3:
        raise ValueError(f"expected (T, C, F), got {feat.shape}")
    if baseline.shape != feat.shape[1:]:
        raise ValueError(f"baseline shape {baseline.shape} != feat tail {feat.shape[1:]}")
    denom = np.abs(baseline)[None, ...] + eps
    abs_baseline = np.abs(baseline)[None, ...]

    rel = feat / denom
    cummax = np.maximum.accumulate(feat, axis=0)
    cummax_rel = cummax / denom
    damage = np.maximum(feat - abs_baseline, 0.0)
    cumulative_damage_rel = np.cumsum(damage, axis=0) / denom

    # Cross-product Energy CumDamage (spec §2.2)
    rms_base  = np.abs(baseline[:, _RMS_IDX])[None, :]  + eps  # (1, C)
    kurt_base = np.abs(baseline[:, _KURT_IDX])[None, :] + eps  # (1, C)
    energy = (feat[:, :, _RMS_IDX] / rms_base) * (feat[:, :, _KURT_IDX] / kurt_base)  # (T, C)
    energy_cumdamage = np.cumsum(np.maximum(energy - 1.0, 0.0), axis=0)[:, :, None]   # (T, C, 1)

    out = np.concatenate([rel, cummax_rel, cumulative_damage_rel, energy_cumdamage], axis=-1)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def augment_cases(
    per_case_feats: Iterable[np.ndarray],
    baseline: np.ndarray,
) -> list[np.ndarray]:
    """Apply augment_with_degradation to a list of per-case feature arrays."""
    return [augment_with_degradation(f, baseline) for f in per_case_feats]
