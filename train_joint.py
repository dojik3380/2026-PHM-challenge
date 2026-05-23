"""
train_joint.py

합성 데이터 + 실제 4개 케이스를 동시에 학습한다 (joint training).

- pretrain/finetune 2단계 없이 한번에 학습
- GroupKFold: 실제 케이스 기준 split
  - train fold = 실제 3개 케이스 + 합성 전체
  - val fold   = 실제 1개 케이스만 (합성 없음)
- 사전학습 가중치 로드 없음 (scratch)

Usage:
    python train_joint.py
    python train_joint.py --sim-weight 0.3  # 합성 데이터 샘플 비율 조정
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from config import (
    AUGMENTATION_PROB,
    BATCH_SIZE,
    DEVICE,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    LEARNING_RATE,
    MIXUP_ALPHA,
    MIXUP_PROB,
    MODELS_DIR,
    SCHEDULER_T0,
    STFT_FREQ_BINS,
    STFT_NOVERLAP,
    STFT_NPERSEG,
    STRIDE,
    TRAIN_DIR,
    WEIGHT_DECAY,
    WINDOW_SIZE,
)
from data_loader import apply_data_augmentation, load_dataset
from model import CombinedLoss, asymmetric_rul_score_np, create_model


def _mixup_batch(x_vib, x_aux, y, alpha=MIXUP_ALPHA, prob=MIXUP_PROB):
    """train.py와 동일한 batch-level Mixup."""
    if alpha <= 0.0 or np.random.random() > prob:
        return x_vib, x_aux, y
    lam = float(np.random.beta(alpha, alpha))
    lam = max(lam, 1.0 - lam)
    perm = torch.randperm(x_vib.size(0), device=x_vib.device)
    return (
        lam * x_vib + (1.0 - lam) * x_vib[perm],
        lam * x_aux + (1.0 - lam) * x_aux[perm],
        lam * y + (1.0 - lam) * y[perm],
    )

JOINT_MODEL_PATH = MODELS_DIR / "RUL_Joint.pt"
SIM_DATA_PATH = Path("outputs/synthetic/pretrain_data.npz")


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


def load_sim_data(sim_weight: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """합성 데이터 로드. sim_weight < 1이면 랜덤 샘플링."""
    if not SIM_DATA_PATH.exists():
        raise FileNotFoundError(f"합성 데이터 없음: {SIM_DATA_PATH}\npython generate_synthetic_data.py 먼저 실행")
    d = np.load(SIM_DATA_PATH)
    X_vib = d["X_vib"].astype(np.float32)
    X_aux = d["X_aux"].astype(np.float32)
    y     = d["y_rul"].astype(np.float32)

    if sim_weight < 1.0:
        n = max(1, int(len(y) * sim_weight))
        idx = np.random.choice(len(y), n, replace=False)
        X_vib, X_aux, y = X_vib[idx], X_aux[idx], y[idx]

    print(f"  합성 데이터: {len(y)} samples")
    return X_vib, X_aux, y


def train_joint(
    sim_weight: float = 1.0,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    model_path: Path = JOINT_MODEL_PATH,
    target_synth_ratio: float = 0.10,
    epoch_oversample: int = 5,
) -> None:
    print("=" * 60)
    print("  Joint Training (Synthetic + Real, No Pretrain)")
    print("=" * 60)

    # ── 실제 데이터 로드
    print("\n[1] 실제 데이터 로딩...")
    X_vib_real, X_aux_real, y_real, meta = load_dataset(
        root_dir=TRAIN_DIR, window_size=WINDOW_SIZE, stride=STRIDE
    )
    print(f"  실제 데이터: {len(y_real)} samples")

    # ── 합성 데이터 로드
    print("\n[2] 합성 데이터 로딩...")
    X_vib_sim, X_aux_sim, y_sim = load_sim_data(sim_weight)

    groups = meta["case_name"].values
    unique_groups = np.unique(groups)
    n_splits = len(unique_groups)
    gkf = GroupKFold(n_splits=n_splits)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(
        gkf.split(np.arange(len(y_real)), groups=groups)
    ):
        print(f"\n{'='*50}")
        print(f"  Fold {fold+1}/{n_splits}  (real train={len(train_idx)}, val={len(val_idx)})")
        print(f"{'='*50}")

        # val: 실제 데이터만
        X_vib_val_r = X_vib_real[val_idx]
        X_aux_val_r = X_aux_real[val_idx]
        y_val_r     = y_real[val_idx]

        # train: 실제 train fold + 합성 전체 합치기
        X_vib_tr = np.concatenate([X_vib_real[train_idx], X_vib_sim], axis=0)
        X_aux_tr = np.concatenate([X_aux_real[train_idx], X_aux_sim], axis=0)
        y_tr     = np.concatenate([y_real[train_idx],     y_sim],     axis=0)
        print(f"  train 총: {len(y_tr)} (real {len(train_idx)} + sim {len(y_sim)})")

        # 표준화 (train 통계로)
        X_vib_tr, X_vib_val, vib_mean, vib_std = _standardize(X_vib_tr, X_vib_val_r)
        X_aux_tr, X_aux_val, aux_mean, aux_std = _standardize(X_aux_tr, X_aux_val_r)
        y_val = y_val_r.copy()

        # 데이터 증강 (실제 train 부분만)
        n_real_tr = len(train_idx)
        X_vib_real_tr_aug, X_aux_real_tr_aug, y_real_tr_aug = apply_data_augmentation(
            X_vib_tr[:n_real_tr], X_aux_tr[:n_real_tr], y_tr[:n_real_tr], AUGMENTATION_PROB
        )
        # 실제 aug + 합성 원본 합치기
        n_real_aug = len(y_real_tr_aug)
        X_vib_tr = np.concatenate([X_vib_real_tr_aug, X_vib_tr[n_real_tr:]], axis=0)
        X_aux_tr = np.concatenate([X_aux_real_tr_aug, X_aux_tr[n_real_tr:]], axis=0)
        y_tr     = np.concatenate([y_real_tr_aug,     y_tr[n_real_tr:]],     axis=0)
        n_synth_tr = len(y_tr) - n_real_aug

        y_tr  = np.log1p(y_tr).astype(np.float32)
        y_val = np.log1p(y_val).astype(np.float32)

        def _t(arr):
            return torch.tensor(np.ascontiguousarray(arr), dtype=torch.float32)

        # ── WeightedRandomSampler: real이 (1 - target_synth_ratio), synth가 target_synth_ratio 비중으로 sampling ──
        # 각 sample weight = target_proportion / count_in_class
        real_weight = (1.0 - target_synth_ratio) / max(n_real_aug, 1)
        synth_weight = target_synth_ratio / max(n_synth_tr, 1)
        sample_weights = np.concatenate([
            np.full(n_real_aug, real_weight, dtype=np.float64),
            np.full(n_synth_tr, synth_weight, dtype=np.float64),
        ])
        num_samples_per_epoch = n_real_aug * epoch_oversample
        sampler = WeightedRandomSampler(
            weights=sample_weights.tolist(),
            num_samples=num_samples_per_epoch,
            replacement=True,
        )
        print(f"  [Sampler] target_synth_ratio={target_synth_ratio} | num_samples/epoch={num_samples_per_epoch}")
        print(f"  [Sampler] real_aug={n_real_aug}, synth={n_synth_tr}")

        train_loader = DataLoader(
            TensorDataset(_t(X_vib_tr), _t(X_aux_tr), _t(y_tr).unsqueeze(1)),
            batch_size=batch_size, sampler=sampler,
        )
        val_loader = DataLoader(
            TensorDataset(_t(X_vib_val), _t(X_aux_val), _t(y_val).unsqueeze(1)),
            batch_size=batch_size, shuffle=False,
        )

        device = torch.device(DEVICE)
        model = create_model(
            vibration_channels=X_vib_real.shape[2],
            auxiliary_dim=X_aux_real.shape[-1],
            vibration_features=X_vib_real.shape[3],
        ).to(device)
        print("  [Joint] Training from scratch - no pretrained weights.")

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
                # Mixup: real/synth가 섞인 batch에 적용 — 도메인 boundary smooth
                bv, ba, by = _mixup_batch(bv, ba, by)
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
                    print(f"  [Early Stop] epoch {epoch}")
                    break

            if epoch % 10 == 0:
                print(f"  epoch {epoch:03d} | val_loss={vl:.5f} | patience={patience_cnt}")

        # ── Fold 평가 (실제 val 데이터만)
        if best_state:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            vib_t = torch.tensor(np.ascontiguousarray(X_vib_val), dtype=torch.float32).to(device)
            aux_t = torch.tensor(np.ascontiguousarray(X_aux_val), dtype=torch.float32).to(device)
            preds_log = model(vib_t, aux_t).squeeze(1).cpu().numpy()

        preds   = np.expm1(np.clip(preds_log, 0, 11.5))
        targets = np.expm1(y_val)
        mae  = float(np.mean(np.abs(preds - targets)))
        rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
        arl  = float(np.mean(asymmetric_rul_score_np(preds, targets)))
        fold_results.append({"mae": mae, "rmse": rmse, "a_rul": arl})
        print(f"\n  Fold {fold+1} 결과 | MAE={mae:.1f}  RMSE={rmse:.1f}  A_RUL={arl:.4f}")

        # 모델 저장
        fp = Path(str(model_path).replace(".pt", f"_fold{fold+1}.pt"))
        torch.save({
            "model_state_dict": model.state_dict(),
            "window_size": WINDOW_SIZE, "stride": STRIDE,
            "stft_nperseg": STFT_NPERSEG, "stft_noverlap": STFT_NOVERLAP,
            "stft_freq_bins": STFT_FREQ_BINS,
            "vibration_channels": X_vib_real.shape[2],
            "auxiliary_dim": X_aux_real.shape[-1],
            "vibration_features": X_vib_real.shape[3],
            "vibration_mean": vib_mean, "vibration_std": vib_std,
            "auxiliary_mean": aux_mean, "auxiliary_std": aux_std,
        }, fp)

    # ── 최종 요약
    maes  = [r["mae"]   for r in fold_results]
    rmses = [r["rmse"]  for r in fold_results]
    arls  = [r["a_rul"] for r in fold_results]

    print("\n" + "=" * 60)
    print("  Joint Training 최종 결과")
    print("=" * 60)
    print(f"  MAE  : {np.mean(maes):.1f} ± {np.std(maes):.1f}")
    print(f"  RMSE : {np.mean(rmses):.1f} ± {np.std(rmses):.1f}")
    print(f"  A_RUL: {np.mean(arls):.4f} ± {np.std(arls):.4f}")
    print(f"\n  fold별 A_RUL: {[f'{a:.4f}' for a in arls]}")
    print("\n  비교:")
    print(f"  OFF only (best) : A_RUL ≈ 0.577")
    print(f"  Joint           : A_RUL = {np.mean(arls):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-weight", type=float, default=1.0,
                        help="합성 데이터 로딩 비율 (0~1, 기본 1.0=전체)")
    parser.add_argument("--target-synth-ratio", type=float, default=0.10,
                        help="batch당 synth sample 목표 비중 (0~1, 기본 0.10=10%%)")
    parser.add_argument("--epoch-oversample", type=int, default=5,
                        help="epoch당 real을 N배 본다 (기본 5)")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--model-path", type=Path, default=JOINT_MODEL_PATH)
    args = parser.parse_args()
    train_joint(
        sim_weight=args.sim_weight,
        epochs=args.epochs,
        model_path=args.model_path,
        target_synth_ratio=args.target_synth_ratio,
        epoch_oversample=args.epoch_oversample,
    )
