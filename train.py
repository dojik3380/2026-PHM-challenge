"""STFT + CNN + LSTM PHM RUL 모델 학습."""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, TensorDataset

from config import (
    BATCH_SIZE,
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    MODEL_PATH,
    MODELS_DIR,
    RANDOM_STATE,
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
from data_loader import load_dataset, operation_feature_names
from model import AsymmetricRULLoss, CombinedLoss, create_model


def _standardize_temporal_array(
    train_array: np.ndarray,
    val_array: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor]:
    """
    batch와 sequence 축을 합쳐 표준화한다.
    진동은 (4, freq_bins), 운전은 (features,) 단위로 평균/표준편차를 가진다.
    """
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


def train_model(
    data_dir: Path = TRAIN_DIR,
    model_path: Path = MODEL_PATH,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
    random_state: int = RANDOM_STATE,
    max_samples: Optional[int] = None,
) -> Path:
    """Case-level split, DataLoader, AdamW, 비대칭 RUL loss로 학습한다."""
    X_vib, X_op, y, metadata = load_dataset(
        root_dir=data_dir,
        window_size=window_size,
        stride=stride,
        max_samples=max_samples,
    )
    if len(y) < 2:
        raise ValueError("At least two sequence samples are required for train/validation split.")

    # Case-level split: metadata['case_name']을 group으로 사용
    from sklearn.model_selection import GroupKFold
    
    groups = metadata['case_name'].values
    unique_groups = np.unique(groups)
    n_splits = len(unique_groups)
    
    gkf = GroupKFold(n_splits=n_splits)
    
    saved_models = []
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X=np.arange(len(y)), y=None, groups=groups)):
        print(f"\n========== Fold {fold+1}/{n_splits} ==========")
        X_vib_train, X_vib_val, vib_mean, vib_std = _standardize_temporal_array(X_vib[train_idx], X_vib[val_idx])
        X_op_train, X_op_val, op_mean, op_std = _standardize_temporal_array(X_op[train_idx], X_op[val_idx])
        y_train = y[train_idx].astype(np.float32)
        y_val = y[val_idx].astype(np.float32)

        # 데이터 증강 적용 (훈련 데이터만)
        from data_loader import apply_data_augmentation
        X_vib_train, X_op_train, y_train = apply_data_augmentation(
            X_vib_train, X_op_train, y_train, aug_prob=AUGMENTATION_PROB
        )

        # RUL log scaling 다시 적용 (원본 스케일이 너무 커서 학습 불가)
        y_train = np.log1p(y_train)
        y_val = np.log1p(y_val)

        train_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(X_vib_train),
                torch.from_numpy(X_op_train),
                torch.from_numpy(y_train).unsqueeze(1),
            ),
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(X_vib_val),
                torch.from_numpy(X_op_val),
                torch.from_numpy(y_val).unsqueeze(1),
            ),
            batch_size=batch_size,
            shuffle=False,
        )

        device = torch.device(DEVICE)
        model = create_model(
            vibration_channels=X_vib.shape[2],
            operation_features=X_op.shape[-1],
            vibration_features=X_vib.shape[3],
        ).to(device)

        # ── 사전학습 가중치 로드 (Transfer Learning) ─────────────────────
        pretrained_path = MODELS_DIR / "RUL_pretrained.pt"
        if pretrained_path.exists():
            print(f"[Transfer] Loading pre-trained weights from {pretrained_path.name}...")
            pretrained_state = torch.load(pretrained_path, map_location=device)
            model_state = model.state_dict()
            # shape이 일치하는 레이어만 로드 (채널 수 불일치 레이어는 건너뜀)
            matched, skipped = 0, 0
            for k, v in pretrained_state.items():
                if k in model_state and model_state[k].shape == v.shape:
                    model_state[k] = v
                    matched += 1
                else:
                    skipped += 1
            model.load_state_dict(model_state)
            print(f"[Transfer] Matched={matched} layers loaded | Skipped={skipped} layers (shape mismatch)")
        else:
            print("[Transfer] No pre-trained weights found. Training from scratch.")
        # ─────────────────────────────────────────────────────────────────

        criterion = CombinedLoss()
        # 전이학습 시 Fine-tuning을 위해 학습률을 더 낮게 설정
        ft_lr = learning_rate * 0.5 if pretrained_path.exists() else learning_rate
        optimizer = torch.optim.AdamW(model.parameters(), lr=ft_lr, weight_decay=WEIGHT_DECAY)
        # 스케줄러를 CosineAnnealingWarmRestarts로 교체하여 수렴 속도 및 성능 향상
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

        print(f"Device: {device} | Train: {len(train_idx)} | Val: {len(val_idx)}")


        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = 0.0
            for batch_vib, batch_op, batch_y in train_loader:
                batch_vib = batch_vib.to(device)
                batch_op = batch_op.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                predictions = model(batch_vib, batch_op)
                loss = criterion(predictions, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item() * batch_y.size(0)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_vib, batch_op, batch_y in val_loader:
                    batch_vib = batch_vib.to(device)
                    batch_op = batch_op.to(device)
                    batch_y = batch_y.to(device)
                    val_loss += criterion(model(batch_vib, batch_op), batch_y).item() * batch_y.size(0)

            train_loss /= len(train_loader.dataset)
            val_loss /= len(val_loader.dataset)
            scheduler.step()
            
            if epoch % 10 == 0:
                print(f"Fold {fold+1} - Epoch {epoch:03d}/{epochs} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

            # 마지막 Fold에서만 시각화 저장 (성능 저하 방지)
            if fold == n_splits - 1 and epoch % 10 == 0:
                from config import PROJECT_ROOT
                from visualization import save_epoch_dashboard
                val_run_ids = metadata.iloc[val_idx]['case_name'].values
                save_epoch_dashboard(
                    epoch=epoch,
                    model=model,
                    val_loader=val_loader,
                    device=device,
                    run_ids=val_run_ids,
                    output_dir=PROJECT_ROOT / "outputs" / "visualization"
                )

        fold_model_path = Path(str(model_path).replace(".pt", f"_fold{fold+1}.pt"))
        fold_model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "window_size": window_size,
                "stride": stride,
                "stft_nperseg": STFT_NPERSEG,
                "stft_noverlap": STFT_NOVERLAP,
                "stft_freq_bins": STFT_FREQ_BINS,
                "vibration_channels": X_vib.shape[2],
                "operation_features": X_op.shape[-1],
                "vibration_features": X_vib.shape[3],
                "operation_feature_names": operation_feature_names(),
                "vibration_mean": vib_mean,
                "vibration_std": vib_std,
                "operation_mean": op_mean,
                "operation_std": op_std,
            },
            fold_model_path,
        )
        print(f"Saved fold {fold+1} model to {fold_model_path}")
        saved_models.append(fold_model_path)
        
    return saved_models[0]


if __name__ == "__main__":
    train_model()
