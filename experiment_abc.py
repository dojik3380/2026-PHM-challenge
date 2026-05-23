"""
experiment_abc.py

3가지 전략을 공정하게 비교한다.

  Experiment A : TDMS-only baseline
                 입력 = vibration STFT / auxiliary = zeros
  Experiment B : TDMS + Ground Truth RPM (upper bound)
                 입력 = vibration STFT + GT RPM from operation CSV
  Experiment C : TDMS + Predicted RPM  (test-time simulation)
                 입력 = vibration STFT + FFT-estimated RPM from TDMS
  Experiment C+: C with RPM Gaussian perturbation during training

Usage:
    python experiment_abc.py [--experiments A B C Cplus] [--epochs 80]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, TensorDataset

from config import (
    AUGMENTATION_PROB,
    BATCH_SIZE,
    DEVICE,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    LEARNING_RATE,
    MODELS_DIR,
    PROJECT_ROOT,
    SAMPLING_RATE,
    SCHEDULER_T0,
    STFT_FREQ_BINS,
    STFT_NOVERLAP,
    STFT_NPERSEG,
    STRIDE,
    TRAIN_DIR,
    WEIGHT_DECAY,
    WINDOW_SIZE,
)
from data_loader import (
    apply_data_augmentation,
    discover_cases,
    load_operation_csv,
    _extract_auxiliary_from_tdms,
)
from features import vibration_stft_timestep
from features.rpm_estimator import estimate_rpm_trajectory, extract_rms_from_channels
from model import CombinedLoss, asymmetric_rul_score_np, create_model

# ── 출력 디렉토리 ─────────────────────────────────────────────────
RESULTS_DIR = PROJECT_ROOT / "results" / "experiment_abc"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Regime 경계값 ─────────────────────────────────────────────────
REGIME_THRESHOLD = 850.0   # RPM < 850 → low, RPM >= 850 → high
REGIME_LOW_RANGE  = (700, 760)
REGIME_HIGH_RANGE = (940, 980)

RPM_NOISE_SIGMA = 20.0     # Experiment C+ RPM 노이즈 표준편차 (rpm)


# ══════════════════════════════════════════════════════════════════
#  1. 데이터 로딩
# ══════════════════════════════════════════════════════════════════

def _time_from_index(index: int, max_time: float, total: int) -> float:
    if total <= 1:
        return max_time
    return max_time * index / float(total - 1)


def _load_case_timesteps(
    operation_csv: Path,
    vibration_dir: Path,
    mode: str,          # 'A', 'B', 'C'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """TDMS 파일 시퀀스에서 (vib_steps, aux_steps, gt_rpm_steps, times)를 반환한다.

    Returns
    -------
    vib_steps    : (T, 4, freq_bins)  STFT 특징
    aux_steps    : (T, 2)             [RPM, RMS]  (mode에 따라 다름)
    gt_rpm_steps : (T,)               operation CSV에서 보간한 실제 RPM
    times        : (T,)               각 TDMS 파일의 절대 시간(초)
    """
    op_df = load_operation_csv(operation_csv)
    max_time = float(op_df["time_sec"].max())

    # GT RPM 보간용 레퍼런스
    has_speed = "speed" in op_df.columns
    if has_speed:
        csv_times = op_df["time_sec"].values
        csv_rpms  = pd.to_numeric(op_df["speed"], errors="coerce").fillna(0).values
    else:
        csv_times = np.array([0.0, max_time])
        csv_rpms  = np.zeros(2)

    all_tdms = sorted(Path(vibration_dir).glob("*.tdms"))
    if not all_tdms:
        raise ValueError(f"No TDMS files in {vibration_dir}")
    total = len(all_tdms)

    vib_steps:    List[np.ndarray] = []
    aux_steps:    List[np.ndarray] = []
    gt_rpm_steps: List[float]      = []
    times_list:   List[float]      = []

    for idx, tdms_path in enumerate(all_tdms):
        t = _time_from_index(idx, max_time, total)

        # STFT 특징 (캐시)
        vib_feat = vibration_stft_timestep(tdms_path)

        # estimated RPM & RMS (캐시) — mode B/C/Cplus 공통으로 미리 추출
        aux_cached = _extract_auxiliary_from_tdms(tdms_path)  # [est_rpm, est_rms]
        est_rpm, est_rms = float(aux_cached[0]), float(aux_cached[1])

        # GT RPM 보간
        gt_rpm = float(np.interp(t, csv_times, csv_rpms))

        # mode별 auxiliary 구성
        if mode == 'A':
            aux = np.array([0.0, 0.0], dtype=np.float32)
        elif mode == 'B':
            aux = np.array([gt_rpm, est_rms], dtype=np.float32)
        else:  # 'C' or 'Cplus'
            aux = np.array([est_rpm, est_rms], dtype=np.float32)

        vib_steps.append(vib_feat)
        aux_steps.append(aux)
        gt_rpm_steps.append(gt_rpm)
        times_list.append(t)

    return (
        np.asarray(vib_steps,    dtype=np.float32),  # (T, 4, freq_bins)
        np.asarray(aux_steps,    dtype=np.float32),  # (T, 2)
        np.asarray(gt_rpm_steps, dtype=np.float32),  # (T,)
        np.asarray(times_list,   dtype=np.float32),  # (T,)
    )


def load_dataset_for_experiment(
    mode: str,
    root_dir: Path = TRAIN_DIR,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """mode ∈ {'A','B','C','Cplus'} 에 맞게 데이터셋을 반환한다.

    Returns
    -------
    X_vib   : (N, window_size, 4, freq_bins)
    X_aux   : (N, window_size, 2)
    y       : (N,)                  RUL in seconds
    gt_rpms : (N, window_size)      각 창의 GT RPM (RPM 분석용)
    metadata: DataFrame
    """
    # 'Cplus'는 데이터 로딩은 'C'와 동일, 학습 시 RPM 노이즈 추가
    load_mode = 'C' if mode == 'Cplus' else mode

    cases = discover_cases(root_dir)
    vib_batches, aux_batches, y_batches, gt_rpm_batches, meta_frames = [], [], [], [], []

    for case_name, op_csv, vib_dir in cases:
        print(f"  Loading [{mode}] {case_name} ...", end=" ", flush=True)
        vib_steps, aux_steps, gt_rpm_steps, times = _load_case_timesteps(
            op_csv, vib_dir, mode=load_mode
        )
        op_df   = load_operation_csv(op_csv)
        max_t   = float(op_df["time_sec"].max())

        for start in range(0, len(times) - window_size + 1, stride):
            end = start + window_size
            current_time = float(times[end - 1])
            rul = max_t - current_time

            vib_batches.append(vib_steps[start:end])
            aux_batches.append(aux_steps[start:end])
            gt_rpm_batches.append(gt_rpm_steps[start:end])
            y_batches.append(rul)
            meta_frames.append({
                "case_name":  case_name,
                "time_sec":   current_time,
            })
        print(f"{len(vib_batches)} windows total")

    X_vib   = np.asarray(vib_batches,    dtype=np.float32)   # (N, W, 4, F)
    X_aux   = np.asarray(aux_batches,    dtype=np.float32)   # (N, W, 2)
    y       = np.asarray(y_batches,      dtype=np.float32)   # (N,)
    gt_rpms = np.asarray(gt_rpm_batches, dtype=np.float32)   # (N, W)
    meta    = pd.DataFrame(meta_frames)

    return X_vib, X_aux, y, gt_rpms, meta


# ══════════════════════════════════════════════════════════════════
#  2. 학습 유틸
# ══════════════════════════════════════════════════════════════════

def _standardize(
    train: np.ndarray, val: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor]:
    flat = train.reshape(-1, *train.shape[2:])
    mean = flat.mean(axis=0)
    std  = np.where(flat.std(axis=0) < 1e-8, 1.0, flat.std(axis=0))
    return (
        ((train - mean) / std).astype(np.float32),
        ((val   - mean) / std).astype(np.float32),
        torch.tensor(mean, dtype=torch.float32),
        torch.tensor(std,  dtype=torch.float32),
    )


def _add_rpm_noise(X_aux: np.ndarray, sigma: float) -> np.ndarray:
    """X_aux[..., 0] (RPM 차원)에 Gaussian noise를 추가한다."""
    noisy = X_aux.copy()
    noisy[..., 0] += np.random.normal(0, sigma, X_aux[..., 0].shape).astype(np.float32)
    noisy[..., 0] = np.clip(noisy[..., 0], 0, None)   # RPM ≥ 0
    return noisy


def train_one_fold(
    fold: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    X_vib: np.ndarray,
    X_aux: np.ndarray,
    y: np.ndarray,
    mode: str,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
) -> Tuple[torch.nn.Module, dict]:
    """단일 fold 학습. 학습된 model과 표준화 통계를 반환한다."""
    X_vib_tr, X_vib_val, vib_mean, vib_std = _standardize(X_vib[train_idx], X_vib[val_idx])
    X_aux_tr, X_aux_val, aux_mean, aux_std = _standardize(X_aux[train_idx], X_aux[val_idx])
    y_tr = y[train_idx].copy()
    y_val = y[val_idx].copy()

    # 데이터 증강 (기존 파이프라인 재사용)
    X_vib_tr, X_aux_tr, y_tr = apply_data_augmentation(X_vib_tr, X_aux_tr, y_tr, AUGMENTATION_PROB)

    # Experiment C+: 표준화 후 RPM 차원에 노이즈 추가
    if mode == 'Cplus':
        X_aux_tr = _add_rpm_noise(X_aux_tr, sigma=RPM_NOISE_SIGMA)

    y_tr  = np.log1p(y_tr).astype(np.float32)
    y_val = np.log1p(y_val).astype(np.float32)

    train_ds = TensorDataset(
        torch.from_numpy(X_vib_tr),
        torch.from_numpy(X_aux_tr),
        torch.from_numpy(y_tr).unsqueeze(1),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_vib_val),
        torch.from_numpy(X_aux_val),
        torch.from_numpy(y_val).unsqueeze(1),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    device = torch.device(DEVICE)
    model  = create_model(
        vibration_channels=X_vib.shape[2],
        auxiliary_dim=X_aux.shape[-1],
        vibration_features=X_vib.shape[3],
    ).to(device)

    # Transfer learning (있으면 로드)
    pretrained_path = MODELS_DIR / "RUL_pretrained.pt"
    if pretrained_path.exists():
        ckpt = torch.load(pretrained_path, map_location=device)
        weights = ckpt.get("model_state_dict", ckpt)
        ms = model.state_dict()
        n_ok = 0
        for k, v in weights.items():
            if k in ms and ms[k].shape == v.shape:
                ms[k] = v
                n_ok += 1
        model.load_state_dict(ms)
        print(f"  [Transfer] {n_ok} layers loaded from {pretrained_path.name}")

    criterion = CombinedLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=SCHEDULER_T0, T_mult=2
    )

    best_loss, patience_cnt, best_state = float("inf"), 0, None

    for epoch in range(1, epochs + 1):
        model.train()
        for bv, ba, by in train_loader:
            bv, ba, by = bv.to(device), ba.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bv, ba), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for bv, ba, by in val_loader:
                bv, ba, by = bv.to(device), ba.to(device), by.to(device)
                vl += criterion(model(bv, ba), by).item() * by.size(0)
        vl /= len(val_loader.dataset)
        scheduler.step()

        if vl < best_loss:
            best_loss, patience_cnt = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= EARLY_STOPPING_PATIENCE:
                print(f"    [Early Stop] fold {fold+1} epoch {epoch}")
                break

        if epoch % 10 == 0:
            print(f"    fold {fold+1} epoch {epoch:03d} | val_loss={vl:.5f} | patience={patience_cnt}")

    if best_state:
        model.load_state_dict(best_state)

    stats = {
        "vib_mean": vib_mean, "vib_std": vib_std,
        "aux_mean": aux_mean, "aux_std": aux_std,
    }
    return model, stats


# ══════════════════════════════════════════════════════════════════
#  3. 평가
# ══════════════════════════════════════════════════════════════════

def evaluate_fold(
    model: torch.nn.Module,
    stats: dict,
    X_vib_val: np.ndarray,
    X_aux_val: np.ndarray,
    y_val: np.ndarray,
) -> dict:
    """fold validation set 평가. MAE / RMSE / A_RUL 반환."""
    device = torch.device(DEVICE)
    model.eval()

    # 표준화
    def _scale(arr, mean, std):
        return ((arr - mean.numpy()) / std.numpy()).astype(np.float32)

    X_vib_s = _scale(X_vib_val, stats["vib_mean"], stats["vib_std"])
    X_aux_s = _scale(X_aux_val, stats["aux_mean"], stats["aux_std"])

    vib_t = torch.from_numpy(X_vib_s).to(device)
    aux_t = torch.from_numpy(X_aux_s).to(device)

    with torch.no_grad():
        preds_log = model(vib_t, aux_t).squeeze(1).cpu().numpy()

    preds_log = np.clip(preds_log, 0.0, 11.5)
    preds = np.expm1(preds_log)
    targets = y_val.astype(np.float64)

    mae  = float(np.mean(np.abs(preds - targets)))
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    arl  = float(np.mean(asymmetric_rul_score_np(preds, targets)))

    return {"mae": mae, "rmse": rmse, "a_rul": arl, "preds": preds, "targets": targets}


# ══════════════════════════════════════════════════════════════════
#  4. RPM 분석 (Experiment C/C+)
# ══════════════════════════════════════════════════════════════════

def compute_rpm_analysis(
    X_aux_raw: np.ndarray,   # (N, W, 2) estimated RPM (mode=C/Cplus)
    gt_rpms:   np.ndarray,   # (N, W) GT RPM
) -> dict:
    """predicted vs GT RPM 비교 분석."""
    pred_rpm = X_aux_raw[:, -1, 0].astype(np.float64)   # 마지막 timestep RPM
    true_rpm = gt_rpms[:, -1].astype(np.float64)

    mask = true_rpm > 0
    pred_rpm = pred_rpm[mask]
    true_rpm = true_rpm[mask]

    mae  = float(np.mean(np.abs(pred_rpm - true_rpm)))
    rmse = float(np.sqrt(np.mean((pred_rpm - true_rpm) ** 2)))

    # Regime classification (GT 기준)
    gt_regime   = (true_rpm >= REGIME_THRESHOLD).astype(int)   # 0=low, 1=high
    pred_regime = (pred_rpm >= REGIME_THRESHOLD).astype(int)

    acc = float(np.mean(gt_regime == pred_regime))

    return {
        "pred_rpm": pred_rpm,
        "true_rpm": true_rpm,
        "rpm_mae":  mae,
        "rpm_rmse": rmse,
        "regime_acc": acc,
        "gt_regime":  gt_regime,
        "pred_regime": pred_regime,
    }


# ══════════════════════════════════════════════════════════════════
#  5. 시각화
# ══════════════════════════════════════════════════════════════════

def _save_rpm_plots(rpm_analysis: dict, out_dir: Path) -> None:
    pred = rpm_analysis["pred_rpm"]
    true = rpm_analysis["true_rpm"]

    # Scatter plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(true, pred, alpha=0.4, s=10, color="steelblue")
    lim = (min(true.min(), pred.min()) * 0.95, max(true.max(), pred.max()) * 1.05)
    ax.plot(lim, lim, "r--", lw=1, label="ideal")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("GT RPM"); ax.set_ylabel("Predicted RPM")
    ax.set_title(f"RPM: Predicted vs GT\nMAE={rpm_analysis['rpm_mae']:.1f}  RMSE={rpm_analysis['rpm_rmse']:.1f}  RegimeAcc={rpm_analysis['regime_acc']:.3f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "rpm_scatter.png", dpi=150)
    plt.close(fig)

    # Histogram
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(600, 1050, 60)
    ax.hist(true, bins=bins, alpha=0.6, label="GT RPM", color="green")
    ax.hist(pred, bins=bins, alpha=0.6, label="Pred RPM", color="orange")
    ax.axvline(REGIME_LOW_RANGE[0],  color="gray", ls="--", lw=0.8)
    ax.axvline(REGIME_LOW_RANGE[1],  color="gray", ls="--", lw=0.8)
    ax.axvline(REGIME_HIGH_RANGE[0], color="navy", ls="--", lw=0.8)
    ax.axvline(REGIME_HIGH_RANGE[1], color="navy", ls="--", lw=0.8)
    ax.set_xlabel("RPM"); ax.set_ylabel("Count")
    ax.set_title("RPM Distribution: GT vs Predicted")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "rpm_histogram.png", dpi=150)
    plt.close(fig)


def _save_comparison_table(all_results: Dict[str, list], out_dir: Path) -> None:
    rows = []
    for exp_name, fold_results in all_results.items():
        maes  = [r["mae"]   for r in fold_results]
        rmses = [r["rmse"]  for r in fold_results]
        arls  = [r["a_rul"] for r in fold_results]
        rows.append({
            "Experiment":  exp_name,
            "MAE mean":    f"{np.mean(maes):.1f}",
            "MAE std":     f"{np.std(maes):.1f}",
            "RMSE mean":   f"{np.mean(rmses):.1f}",
            "RMSE std":    f"{np.std(rmses):.1f}",
            "A_RUL mean":  f"{np.mean(arls):.4f}",
            "A_RUL std":   f"{np.std(arls):.4f}",
        })
    df = pd.DataFrame(rows)
    print("\n" + "=" * 70)
    print(df.to_string(index=False))
    print("=" * 70)
    df.to_csv(out_dir / "comparison_table.csv", index=False)

    # Bar chart
    exp_names = [r["Experiment"] for r in rows]
    mae_means = [float(r["MAE mean"]) for r in rows]
    arl_means = [float(r["A_RUL mean"]) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["steelblue", "darkorange", "green", "red"][:len(exp_names)]

    axes[0].bar(exp_names, mae_means, color=colors)
    axes[0].set_title("MAE (lower = better)"); axes[0].set_ylabel("MAE (seconds)")

    axes[1].bar(exp_names, arl_means, color=colors)
    axes[1].set_title("A_RUL Score (lower = better)"); axes[1].set_ylabel("A_RUL")

    for ax in axes:
        ax.tick_params(axis="x", rotation=15)
    fig.suptitle("Experiment A / B / C Comparison")
    fig.tight_layout()
    fig.savefig(out_dir / "comparison_bar.png", dpi=150)
    plt.close(fig)

    # Fold box plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    mae_data = [[r["mae"]   for r in v] for v in all_results.values()]
    arl_data = [[r["a_rul"] for r in v] for v in all_results.values()]
    for ax, data, title in zip(axes, [mae_data, arl_data], ["MAE per Fold", "A_RUL per Fold"]):
        bp = ax.boxplot(data, labels=list(all_results.keys()), patch_artist=True)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_dir / "comparison_boxplot.png", dpi=150)
    plt.close(fig)


def _save_rpm_metrics_table(all_rpm: Dict[str, dict], out_dir: Path) -> None:
    rows = []
    for exp_name, rpm_info in all_rpm.items():
        rows.append({
            "Experiment":    exp_name,
            "RPM MAE":       f"{rpm_info['rpm_mae']:.2f}",
            "RPM RMSE":      f"{rpm_info['rpm_rmse']:.2f}",
            "Regime Acc":    f"{rpm_info['regime_acc']:.4f}",
        })
    df = pd.DataFrame(rows)
    print("\n--- RPM Analysis ---")
    print(df.to_string(index=False))
    df.to_csv(out_dir / "rpm_metrics.csv", index=False)


# ══════════════════════════════════════════════════════════════════
#  6. 실험 실행
# ══════════════════════════════════════════════════════════════════

def run_experiment(
    mode: str,
    epochs: int,
    root_dir: Path = TRAIN_DIR,
) -> Tuple[List[dict], Optional[dict]]:
    """단일 experiment 실행. (fold_results, rpm_analysis|None) 반환."""
    print(f"\n{'='*60}")
    print(f"  Experiment {mode}")
    print(f"{'='*60}")

    X_vib, X_aux, y, gt_rpms, meta = load_dataset_for_experiment(
        mode=mode, root_dir=root_dir
    )
    print(f"  Dataset: {len(y)} samples | vib={X_vib.shape} | aux={X_aux.shape}")

    groups         = meta["case_name"].values
    unique_groups  = np.unique(groups)
    n_splits       = len(unique_groups)
    gkf            = GroupKFold(n_splits=n_splits)

    fold_results: List[dict] = []

    for fold, (train_idx, val_idx) in enumerate(
        gkf.split(np.arange(len(y)), groups=groups)
    ):
        print(f"\n  ── Fold {fold+1}/{n_splits} (train={len(train_idx)}, val={len(val_idx)}) ──")
        model, stats = train_one_fold(
            fold=fold,
            train_idx=train_idx,
            val_idx=val_idx,
            X_vib=X_vib,
            X_aux=X_aux,
            y=y,
            mode=mode,
            epochs=epochs,
        )

        result = evaluate_fold(
            model=model,
            stats=stats,
            X_vib_val=X_vib[val_idx],
            X_aux_val=X_aux[val_idx],
            y_val=y[val_idx],
        )
        fold_results.append(result)
        print(f"  Fold {fold+1} | MAE={result['mae']:.1f}  RMSE={result['rmse']:.1f}  A_RUL={result['a_rul']:.4f}")

        # 모델 저장
        model_name = f"Exp{mode}_fold{fold+1}.pt"
        torch.save({"model_state_dict": model.state_dict()}, MODELS_DIR / model_name)

    # RPM 분석은 C/C+ 에서만
    rpm_analysis = None
    if mode in ('C', 'Cplus'):
        rpm_analysis = compute_rpm_analysis(X_aux, gt_rpms)
        print(f"\n  RPM MAE={rpm_analysis['rpm_mae']:.2f}  RMSE={rpm_analysis['rpm_rmse']:.2f}  RegimeAcc={rpm_analysis['regime_acc']:.4f}")

    return fold_results, rpm_analysis


# ══════════════════════════════════════════════════════════════════
#  7. main
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="A/B/C Experiment Runner")
    parser.add_argument(
        "--experiments", nargs="+", default=["A", "B", "C", "Cplus"],
        choices=["A", "B", "C", "Cplus"],
        help="실행할 실험 목록 (기본: A B C Cplus)",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--data-dir", type=Path, default=TRAIN_DIR)
    args = parser.parse_args()

    all_results: Dict[str, List[dict]] = {}
    all_rpm: Dict[str, dict] = {}

    for mode in args.experiments:
        fold_results, rpm_analysis = run_experiment(
            mode=mode, epochs=args.epochs, root_dir=args.data_dir
        )
        all_results[f"Exp {mode}"] = fold_results
        if rpm_analysis is not None:
            all_rpm[f"Exp {mode}"] = rpm_analysis

    # ── 비교표 & 시각화 저장
    _save_comparison_table(all_results, RESULTS_DIR)
    if all_rpm:
        _save_rpm_metrics_table(all_rpm, RESULTS_DIR)
        for exp_label, rpm_info in all_rpm.items():
            exp_dir = RESULTS_DIR / exp_label.replace(" ", "_")
            exp_dir.mkdir(exist_ok=True)
            _save_rpm_plots(rpm_info, exp_dir)

    # ── fold별 결과 JSON 저장
    summary = {}
    for exp_label, fold_results in all_results.items():
        maes  = [r["mae"]   for r in fold_results]
        rmses = [r["rmse"]  for r in fold_results]
        arls  = [r["a_rul"] for r in fold_results]
        summary[exp_label] = {
            "mae_per_fold":   maes,
            "rmse_per_fold":  rmses,
            "a_rul_per_fold": arls,
            "mae_mean":   float(np.mean(maes)),
            "mae_std":    float(np.std(maes)),
            "rmse_mean":  float(np.mean(rmses)),
            "rmse_std":   float(np.std(rmses)),
            "a_rul_mean": float(np.mean(arls)),
            "a_rul_std":  float(np.std(arls)),
        }
        if exp_label in all_rpm:
            ri = all_rpm[exp_label]
            summary[exp_label]["rpm_mae"]     = ri["rpm_mae"]
            summary[exp_label]["rpm_rmse"]    = ri["rpm_rmse"]
            summary[exp_label]["regime_acc"]  = ri["regime_acc"]

    with open(RESULTS_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n[Done] 모든 결과가 {RESULTS_DIR} 에 저장되었습니다.")
    print("  - comparison_table.csv")
    print("  - comparison_bar.png")
    print("  - comparison_boxplot.png")
    print("  - rpm_metrics.csv  (C/C+)")
    print("  - Exp_C/rpm_scatter.png")
    print("  - Exp_C/rpm_histogram.png")
    print("  - summary.json")


if __name__ == "__main__":
    main()
