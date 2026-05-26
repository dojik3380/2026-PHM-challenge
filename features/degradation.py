"""Health-Indicator (degradation) features.

Augments per-timestep handcrafted features with three normalized views:

  - relative              = feat / baseline          : current vs healthy ratio
  - cummax_ratio          = cummax(feat) / baseline  : worst-so-far (monotonic, late-life)
  - cumulative_damage     = cumsum(max(feat-baseline,0)) / baseline  : per-feature damage

energy_cumdamage (cross-product (RMS/RMS_base)×(Kurt/Kurt_base)) 는 제거됨 —
Train4 baseline kurtosis OOD 가 곱셈식에서 폭주해 held-out fold mean-collapse 유발.

For inference (Test cases with no healthy baseline available), the GLOBAL
baseline computed from training data is reused.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


EPS = 1e-6


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
    """feat: (T, C, F), baseline: (C, F) → (T, C, F*3).

    Returns [rel, cummax_rel, cumulative_damage_rel]:
      - rel               : feat / |baseline|              (현재 / 건강 비율)
      - cummax_rel        : cummax(feat) / |baseline|      (worst-so-far, 단조)
      - cumulative_damage_rel: cumsum(max(feat-|baseline|, 0)) / |baseline|

    Cross-product Energy CumDamage 는 제거됨 — Train4 의 baseline kurtosis≈68
    (다른 케이스 ≈3.5) 가 곱셈식 (RMS/RMS_base)×(Kurt/Kurt_base) 에 들어가면
    held-out fold 에서 OOD 폭주 → 모델 mean-collapse (HI≈0.5 평탄). 곱셈식 제거 후
    각 feature 의 *비율* 차원에서만 학습되도록 함.

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

    out = np.concatenate([rel, cummax_rel, cumulative_damage_rel], axis=-1)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def augment_cases(
    per_case_feats: Iterable[np.ndarray],
    baseline: np.ndarray,
) -> list[np.ndarray]:
    """Apply augment_with_degradation to a list of per-case feature arrays."""
    return [augment_with_degradation(f, baseline) for f in per_case_feats]
