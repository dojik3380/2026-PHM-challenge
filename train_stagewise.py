"""Stage-wise fine-tuning — 사전학습 가중치를 더 효과적으로 활용.

Stage 1 (head warm-up):
    - Vibration backbone(CNN/projection/LSTM) freeze
    - Head(attention/aux_encoder/fusion)만 학습
    - LR: ft_lr
    - Epoch: stage1_epochs (5~8)

Stage 2 (full unfreeze):
    - 전체 unfreeze, layer-wise LR
    - Backbone LR: ft_lr × backbone_lr_mult (예: 0.1)
    - Head LR: ft_lr
    - Epoch: stage2_epochs (15~20)

train.py와 동일한 4-fold + 사전학습 가중치 로드 + 저장 형식 유지.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, TensorDataset

from config import (
    BATCH_SIZE,
    DEVICE,
    EARLY_STOPPING_PATIENCE,
    LEARNING_RATE,
    MODELS_DIR,
    RANDOM_STATE,
    SCHEDULER_T0,
    STFT_FREQ_BINS,
    STFT_NOVERLAP,
    STFT_NPERSEG,
    STRIDE,
    TEST_SIZE,
    TRAIN_DIR,
    WEIGHT_DECAY,
    WINDOW_SIZE,
    AUGMENTATION_PROB,
)
from data_loader import apply_data_augmentation, load_dataset
from model import CombinedLoss, create_model


def _standardize_temporal_array(train_array, val_array):
    feature_shape = train_array.shape[2:]
    train_flat = train_array.reshape(-1, *feature_shape)
    mean = train_flat.mean(axis=0)
    std = train_flat.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    train_scaled = ((train_array - mean) / std).astype(np.float32)
    val_scaled = ((val_array - mean) / std).astype(np.float32)
    return (
        train_scaled,
        val_scaled,
        torch.tensor(mean, dtype=torch.float32),
        torch.tensor(std, dtype=torch.float32),
    )


def _freeze_backbone(model: torch.nn.Module, freeze: bool) -> None:
    """vibration CNN/projection/LSTM/pos_encoder를 freeze/unfreeze."""
    backbone_modules = ["vibration_cnn", "vibration_projection", "vibration_lstm", "pos_encoder"]
    for name in backbone_modules:
        if hasattr(model, name):
            for p in getattr(model, name).parameters():
                p.requires_grad = (not freeze)


def _trainable_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _layerwise_param_groups(model: torch.nn.Module, base_lr: float, backbone_lr_mult: float = 0.1):
    """Backbone과 head에 다른 LR 적용."""
    backbone_names = {"vibration_cnn", "vibration_projection", "vibration_lstm", "pos_encoder"}
    backbone_params, head_params = [], []
    for name, module in model.named_children():
        params = list(module.parameters())
        if name in backbone_names:
            backbone_params.extend(params)
        else:
            head_params.extend(params)
    return [
        {"params": backbone_params, "lr": base_lr * backbone_lr_mult},
        {"params": head_params, "lr": base_lr},
    ]


def train_stagewise(
    data_dir: Path = TRAIN_DIR,
    model_path: Path = MODELS_DIR / "RUL_Stagewise.pt",
    stage1_epochs: int = 6,
    stage2_epochs: int = 20,
    stage1_lr: float = 3e-4,
    stage2_lr: float = 1e-4,
    backbone_lr_mult: float = 0.1,
    batch_size: int = BATCH_SIZE,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
    random_state: int = RANDOM_STATE,
) -> list[Path]:
    print("=" * 60)
    print("  Stage-wise Fine-tuning")
    print("=" * 60)

    X_vib, X_aux, y, metadata = load_dataset(
        root_dir=data_dir,
        window_size=window_size,
        stride=stride,
    )

    groups = metadata["case_name"].values
    unique_groups = np.unique(groups)
    n_splits = len(unique_groups)
    gkf = GroupKFold(n_splits=n_splits)

    pretrained_path = MODELS_DIR / "RUL_pretrained.pt"
    has_pretrained = pretrained_path.exists()
    print(f"Pretrained available: {has_pretrained} ({pretrained_path})")

    saved_models = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(np.arange(len(y)), None, groups)):
        print(f"\n========== Fold {fold + 1}/{n_splits} ==========")
        X_vib_train, X_vib_val, vib_mean, vib_std = _standardize_temporal_array(X_vib[train_idx], X_vib[val_idx])
        X_aux_train, X_aux_val, aux_mean, aux_std = _standardize_temporal_array(X_aux[train_idx], X_aux[val_idx])
        y_train_raw = y[train_idx].astype(np.float32)
        y_val_raw = y[val_idx].astype(np.float32)

        X_vib_train, X_aux_train, y_train_raw = apply_data_augmentation(
            X_vib_train, X_aux_train, y_train_raw, aug_prob=AUGMENTATION_PROB
        )

        y_train = np.log1p(y_train_raw)
        y_val = np.log1p(y_val_raw)

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
        model = create_model(
            vibration_channels=X_vib.shape[2],
            auxiliary_dim=X_aux.shape[-1],
            vibration_features=X_vib.shape[3],
        ).to(device)

        # ── 사전학습 가중치 로드 ──
        if has_pretrained:
            state = torch.load(pretrained_path, map_location=device)
            pre_state = state.get("model_state_dict", state)
            model_state = model.state_dict()
            matched = 0
            for k, v in pre_state.items():
                if k in model_state and model_state[k].shape == v.shape:
                    model_state[k] = v
                    matched += 1
            model.load_state_dict(model_state)
            print(f"[Transfer] Matched {matched} layers")

        criterion = CombinedLoss()

        # ─────────────────────────────────────────────
        # Stage 1: backbone freeze, head only
        # ─────────────────────────────────────────────
        print(f"[Stage 1] Backbone FROZEN, head warm-up | LR={stage1_lr} | epochs={stage1_epochs}")
        _freeze_backbone(model, freeze=True)
        print(f"  Trainable params (stage 1): {_trainable_params(model):,}")
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=stage1_lr, weight_decay=WEIGHT_DECAY,
        )

        for epoch in range(1, stage1_epochs + 1):
            model.train()
            train_loss = 0.0
            n = 0
            for batch_vib, batch_aux, batch_y in train_loader:
                batch_vib = batch_vib.to(device); batch_aux = batch_aux.to(device); batch_y = batch_y.to(device)
                optimizer.zero_grad()
                pred = model(batch_vib, batch_aux)
                loss = criterion(pred, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item() * batch_y.size(0); n += batch_y.size(0)

            model.eval()
            val_loss = 0.0; nv = 0
            with torch.no_grad():
                for batch_vib, batch_aux, batch_y in val_loader:
                    batch_vib = batch_vib.to(device); batch_aux = batch_aux.to(device); batch_y = batch_y.to(device)
                    val_loss += criterion(model(batch_vib, batch_aux), batch_y).item() * batch_y.size(0); nv += batch_y.size(0)
            print(f"  [S1] Epoch {epoch:02d}/{stage1_epochs} | train={train_loss/n:.4f} | val={val_loss/nv:.4f}")

        # ─────────────────────────────────────────────
        # Stage 2: all unfreeze, layer-wise LR
        # ─────────────────────────────────────────────
        print(f"[Stage 2] All UNFREEZE, layer-wise LR | base={stage2_lr} | backbone_mult={backbone_lr_mult} | epochs={stage2_epochs}")
        _freeze_backbone(model, freeze=False)
        print(f"  Trainable params (stage 2): {_trainable_params(model):,}")
        optimizer = torch.optim.AdamW(
            _layerwise_param_groups(model, base_lr=stage2_lr, backbone_lr_mult=backbone_lr_mult),
            weight_decay=WEIGHT_DECAY,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=SCHEDULER_T0, T_mult=2)

        best_val = float("inf")
        patience_count = 0
        best_state = None

        for epoch in range(1, stage2_epochs + 1):
            model.train()
            train_loss = 0.0; n = 0
            for batch_vib, batch_aux, batch_y in train_loader:
                batch_vib = batch_vib.to(device); batch_aux = batch_aux.to(device); batch_y = batch_y.to(device)
                optimizer.zero_grad()
                pred = model(batch_vib, batch_aux)
                loss = criterion(pred, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item() * batch_y.size(0); n += batch_y.size(0)

            scheduler.step()

            model.eval()
            val_loss = 0.0; nv = 0
            with torch.no_grad():
                for batch_vib, batch_aux, batch_y in val_loader:
                    batch_vib = batch_vib.to(device); batch_aux = batch_aux.to(device); batch_y = batch_y.to(device)
                    val_loss += criterion(model(batch_vib, batch_aux), batch_y).item() * batch_y.size(0); nv += batch_y.size(0)
            val_loss /= nv
            train_loss /= n

            if val_loss < best_val:
                best_val = val_loss
                patience_count = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_count += 1
                if patience_count >= EARLY_STOPPING_PATIENCE:
                    print(f"  [S2] Early stop at epoch {epoch}")
                    break

            if epoch % 5 == 0 or epoch == 1:
                print(f"  [S2] Epoch {epoch:02d}/{stage2_epochs} | train={train_loss:.4f} | val={val_loss:.4f} | best={best_val:.4f}")

        # 저장
        fold_path = Path(str(model_path).replace(".pt", f"_fold{fold + 1}.pt"))
        fold_path.parent.mkdir(parents=True, exist_ok=True)
        if best_state is not None:
            model.load_state_dict(best_state)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "window_size": window_size,
                "stride": stride,
                "stft_nperseg": STFT_NPERSEG,
                "stft_noverlap": STFT_NOVERLAP,
                "stft_freq_bins": STFT_FREQ_BINS,
                "vibration_channels": X_vib.shape[2],
                "auxiliary_dim": X_aux.shape[-1],
                "vibration_features": X_vib.shape[3],
                "vibration_mean": vib_mean,
                "vibration_std": vib_std,
                "auxiliary_mean": aux_mean,
                "auxiliary_std": aux_std,
                "stage1_epochs": stage1_epochs,
                "stage2_epochs": stage2_epochs,
                "best_val_loss": best_val,
            },
            fold_path,
        )
        print(f"  Saved fold {fold + 1} -> {fold_path.name} | best_val={best_val:.4f}")
        saved_models.append(fold_path)

    return saved_models


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=Path, default=MODELS_DIR / "RUL_Stagewise.pt")
    p.add_argument("--stage1-epochs", type=int, default=6)
    p.add_argument("--stage2-epochs", type=int, default=20)
    p.add_argument("--stage1-lr", type=float, default=3e-4)
    p.add_argument("--stage2-lr", type=float, default=1e-4)
    p.add_argument("--backbone-lr-mult", type=float, default=0.1)
    args = p.parse_args()
    train_stagewise(
        model_path=args.model_path,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        stage1_lr=args.stage1_lr,
        stage2_lr=args.stage2_lr,
        backbone_lr_mult=args.backbone_lr_mult,
    )