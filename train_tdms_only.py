"""
train_tdms_only.py

TDMS 실제 데이터만 사용하는 순수 지도학습 베이스라인.
사전학습(pretrain) 없이 처음부터 학습한다.

성능 비교 목적:
  python train_tdms_only.py          → models/RUL_TDMSOnly_fold*.pt
  python train.py                    → models/RUL_Baseline_fold*.pt (pretrain 포함)
  python evaluate.py --model-name RUL_TDMSOnly
  python evaluate.py --model-name RUL_Baseline
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from config import (
    AUGMENTATION_PROB,
    BATCH_SIZE,
    DEVICE,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    LEARNING_RATE,
    MODELS_DIR,
    RANDOM_STATE,
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
from model import CombinedLoss, create_model

# 베이스라인 전용 모델 저장 경로
TDMS_ONLY_MODEL_PATH = MODELS_DIR / "RUL_TDMSOnly.pt"


def _standardize_temporal_array(
    train_array: np.ndarray,
    val_array: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor]:
    feature_shape = train_array.shape[2:]
    train_flat = train_array.reshape(-1, *feature_shape)
    mean = train_flat.mean(axis=0)
    std  = np.where(train_flat.std(axis=0) < 1e-8, 1.0, train_flat.std(axis=0))
    return (
        ((train_array - mean) / std).astype(np.float32),
        ((val_array  - mean) / std).astype(np.float32),
        torch.tensor(mean, dtype=torch.float32),
        torch.tensor(std,  dtype=torch.float32),
    )


def train_tdms_only(
    data_dir: Path = TRAIN_DIR,
    model_path: Path = TDMS_ONLY_MODEL_PATH,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
    random_state: int = RANDOM_STATE,
    max_samples: Optional[int] = None,
) -> Path:
    """
    실제 TDMS 데이터만으로 4-Fold GroupKFold 학습.
    사전학습 가중치를 로드하지 않으며 완전히 처음부터(scratch) 학습한다.
    """
    print("=" * 60)
    print("  TDMS-Only Baseline Training (No Pretraining)")
    print("=" * 60)

    X_vib, X_aux, y, metadata = load_dataset(
        root_dir=data_dir,
        window_size=window_size,
        stride=stride,
        max_samples=max_samples,
    )
    if len(y) < 2:
        raise ValueError("At least two samples required for train/val split.")

    from sklearn.model_selection import GroupKFold
    groups       = metadata["case_name"].values
    unique_groups = np.unique(groups)
    n_splits      = len(unique_groups)
    gkf           = GroupKFold(n_splits=n_splits)

    saved_models = []

    for fold, (train_idx, val_idx) in enumerate(
        gkf.split(X=np.arange(len(y)), y=None, groups=groups)
    ):
        print(f"\n========== Fold {fold+1}/{n_splits} ==========")

        X_vib_train, X_vib_val, vib_mean, vib_std = _standardize_temporal_array(
            X_vib[train_idx], X_vib[val_idx]
        )
        X_aux_train, X_aux_val, aux_mean, aux_std = _standardize_temporal_array(
            X_aux[train_idx], X_aux[val_idx]
        )
        y_train = y[train_idx].astype(np.float32)
        y_val   = y[val_idx].astype(np.float32)

        X_vib_train, X_aux_train, y_train = apply_data_augmentation(
            X_vib_train, X_aux_train, y_train, aug_prob=AUGMENTATION_PROB
        )

        y_train = np.log1p(y_train)
        y_val   = np.log1p(y_val)

        train_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(X_vib_train),
                torch.from_numpy(X_aux_train),
                torch.from_numpy(y_train).unsqueeze(1),
            ),
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(X_vib_val),
                torch.from_numpy(X_aux_val),
                torch.from_numpy(y_val).unsqueeze(1),
            ),
            batch_size=batch_size,
            shuffle=False,
        )

        device = torch.device(DEVICE)
        model  = create_model(
            vibration_channels=X_vib.shape[2],
            auxiliary_dim=X_aux.shape[-1],
            vibration_features=X_vib.shape[3],
        ).to(device)

        # ── 사전학습 없음: 완전히 처음부터 학습 ────────────────────────────
        print("[Baseline] Training from scratch — no pretrained weights loaded.")
        # ─────────────────────────────────────────────────────────────────

        criterion = CombinedLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=SCHEDULER_T0, T_mult=2
        )

        print(f"Device: {device} | Train: {len(train_idx)} | Val: {len(val_idx)}")

        best_val_loss  = float("inf")
        patience_count = 0
        best_state     = None

        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = 0.0
            for batch_vib, batch_aux, batch_y in train_loader:
                batch_vib = batch_vib.to(device)
                batch_aux = batch_aux.to(device)
                batch_y   = batch_y.to(device)

                optimizer.zero_grad()
                loss = criterion(model(batch_vib, batch_aux), batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item() * batch_y.size(0)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_vib, batch_aux, batch_y in val_loader:
                    batch_vib = batch_vib.to(device)
                    batch_aux = batch_aux.to(device)
                    batch_y   = batch_y.to(device)
                    val_loss += criterion(
                        model(batch_vib, batch_aux), batch_y
                    ).item() * batch_y.size(0)

            train_loss /= len(train_loader.dataset)
            val_loss   /= len(val_loader.dataset)
            scheduler.step()

            if val_loss < best_val_loss:
                best_val_loss  = val_loss
                patience_count = 0
                best_state     = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_count += 1
                if patience_count >= EARLY_STOPPING_PATIENCE:
                    print(
                        f"[Early Stop] Fold {fold+1} — no improvement for "
                        f"{EARLY_STOPPING_PATIENCE} epochs at epoch {epoch}."
                    )
                    break

            if epoch % 10 == 0:
                print(
                    f"Fold {fold+1} - Epoch {epoch:03d}/{epochs} | "
                    f"train={train_loss:.6f} | val={val_loss:.6f} | "
                    f"patience={patience_count}/{EARLY_STOPPING_PATIENCE}"
                )

        fold_model_path = Path(str(model_path).replace(".pt", f"_fold{fold+1}.pt"))
        fold_model_path.parent.mkdir(parents=True, exist_ok=True)
        if best_state is not None:
            model.load_state_dict(best_state)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "window_size":       window_size,
                "stride":            stride,
                "stft_nperseg":      STFT_NPERSEG,
                "stft_noverlap":     STFT_NOVERLAP,
                "stft_freq_bins":    STFT_FREQ_BINS,
                "vibration_channels": X_vib.shape[2],
                "auxiliary_dim":     X_aux.shape[-1],
                "vibration_features": X_vib.shape[3],
                "vibration_mean":    vib_mean,
                "vibration_std":     vib_std,
                "auxiliary_mean":    aux_mean,
                "auxiliary_std":     aux_std,
            },
            fold_model_path,
        )
        print(f"Saved fold {fold+1} model → {fold_model_path}")
        saved_models.append(fold_model_path)

    print("\n[Done] TDMS-only baseline training complete.")
    print("=" * 60)
    return saved_models[0]


if __name__ == "__main__":
    train_tdms_only()
