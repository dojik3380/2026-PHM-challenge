"""
pretrain.py

Physics-based Synthetic Data로 모델을 사전학습(Pre-training)하는 스크립트.
생성된 outputs/synthetic/pretrain_data.npz를 로드하여 학습한다.

사용법:
    python pretrain.py
    python pretrain.py --epochs 80 --lr 3e-4 --batch-size 32
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from config import (
    DEVICE, MODELS_DIR,
    PRETRAIN_LR, PRETRAIN_EPOCHS, PRETRAIN_BATCH_SIZE,
    PRETRAIN_WEIGHT_DECAY, SCHEDULER_T0, EARLY_STOPPING_PATIENCE,
)
from model import create_model, CombinedLoss

# ============================================================
# 경로 설정
# ============================================================
PROJECT_ROOT   = Path(__file__).parent
PRETRAIN_DATA  = PROJECT_ROOT / "outputs" / "synthetic" / "pretrain_data.npz"
PRETRAINED_OUT = MODELS_DIR / "RUL_pretrained.pt"


# ============================================================
# 데이터 로드
# ============================================================

def load_pretrain_data(
    data_path: Path = PRETRAIN_DATA,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    pretrain_data.npz를 로드하고 Train/Val DataLoader 및 표준화 통계량을 반환한다.
    """
    print(f"Loading pretrain data from: {data_path}")
    data = np.load(data_path)
    X_vib = data["X_vib"].astype(np.float32)  # (N, W, C, F)
    X_aux = data["X_aux"].astype(np.float32)  # (N, W, 2)
    y_rul = data["y_rul"].astype(np.float32)   # (N,)

    print(f"  X_vib: {X_vib.shape} | X_aux: {X_aux.shape} | y_rul: {y_rul.shape}")
    print(f"  RUL range: {y_rul.min():.1f} ~ {y_rul.max():.1f}")

    # 로그 스케일링 (실제 학습과 동일한 전처리)
    y_log = np.log1p(y_rul)
    
    # ── 표준화 통계량 계산 ──
    # 전체 데이터 기준으로 통계량 계산 (Transfer learning 호환성을 위해)
    vib_flat = X_vib.reshape(-1, X_vib.shape[2], X_vib.shape[3])
    vib_mean = torch.tensor(vib_flat.mean(axis=0), dtype=torch.float32)
    vib_std = torch.tensor(np.where(vib_flat.std(axis=0) < 1e-8, 1.0, vib_flat.std(axis=0)), dtype=torch.float32)
    
    aux_flat = X_aux.reshape(-1, X_aux.shape[2])
    aux_mean = torch.tensor(aux_flat.mean(axis=0), dtype=torch.float32)
    aux_std = torch.tensor(np.where(aux_flat.std(axis=0) < 1e-8, 1.0, aux_flat.std(axis=0)), dtype=torch.float32)
    
    X_vib_scaled = ((X_vib - vib_mean.numpy()) / vib_std.numpy()).astype(np.float32)
    X_aux_scaled = ((X_aux - aux_mean.numpy()) / aux_std.numpy()).astype(np.float32)

    X_vib_tensor = torch.from_numpy(X_vib_scaled)          # (N, W, C, F)
    X_aux_tensor = torch.from_numpy(X_aux_scaled)          # (N, W, 2)
    y_tensor = torch.from_numpy(y_log).unsqueeze(1)  # (N, 1)

    dataset = TensorDataset(X_vib_tensor, X_aux_tensor, y_tensor)
    n_val = int(len(dataset) * val_ratio)
    n_train = len(dataset) - n_val

    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=gen)
    print(f"  Train: {n_train} | Val: {n_val}")
    return train_ds, val_ds, vib_mean, vib_std, aux_mean, aux_std


# ============================================================
# 학습 루프
# ============================================================

def pretrain(args: argparse.Namespace) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(DEVICE)

    print("=" * 60)
    print("  Physics-informed Pre-training")
    print("=" * 60)

    # 데이터 로드
    train_ds, val_ds, vib_mean, vib_std, aux_mean, aux_std = load_pretrain_data(
        data_path=PRETRAIN_DATA,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

    # 모델 생성
    sample_vib, sample_aux, _ = train_ds[0]
    n_channels     = sample_vib.shape[1]
    vib_features   = sample_vib.shape[2]
    aux_features   = sample_aux.shape[-1]
    window_size    = sample_vib.shape[0]

    print(f"\nModel config: channels={n_channels} | vib_features={vib_features} | aux_features={aux_features}")
    model = create_model(
        vibration_channels=n_channels,
        auxiliary_dim=aux_features,
        vibration_features=vib_features,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    criterion = CombinedLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=SCHEDULER_T0, T_mult=2
    )

    best_val_loss = float("inf")
    patience_count = 0
    patience = args.patience

    print(f"\nStarting pre-training for {args.epochs} epochs...\n")

    for epoch in range(1, args.epochs + 1):
        # ── Train ──────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for batch_vib, batch_aux, batch_y in train_loader:
            batch_vib = batch_vib.to(device)      # (B, W, C, F)
            batch_aux = batch_aux.to(device)      # (B, W, 2)
            batch_y = batch_y.to(device)          # (B, 1)

            optimizer.zero_grad()
            pred = model(batch_vib, batch_aux)
            loss = criterion(pred, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(train_loader)

        # ── Val ────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_vib, batch_aux, batch_y in val_loader:
                batch_vib = batch_vib.to(device)
                batch_aux = batch_aux.to(device)
                batch_y = batch_y.to(device)
                pred = model(batch_vib, batch_aux)
                val_loss += criterion(pred, batch_y).item()
        val_loss /= len(val_loader)

        if epoch % 10 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch:03d}/{args.epochs} | "
                  f"train={train_loss:.4f} | val={val_loss:.4f} | lr={lr_now:.2e}")

        # ── Early Stopping & Best Model 저장 ────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
            
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "window_size": window_size,
                    "stride": 4,  # Synthetic data default
                    "vibration_channels": n_channels,
                    "auxiliary_dim": aux_features,
                    "vibration_features": vib_features,
                    "vibration_mean": vib_mean,
                    "vibration_std": vib_std,
                    "auxiliary_mean": aux_mean,
                    "auxiliary_std": aux_std,
                },
                PRETRAINED_OUT
            )
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"\n[Early Stop] No improvement for {patience} epochs. Stopping.")
                break

    print(f"\n[Done] Best val_loss = {best_val_loss:.4f}")
    print(f"Pre-trained model saved -> {PRETRAINED_OUT}")
    print("=" * 60)
    print("\nNext step: run 'python train.py' for fine-tuning with real data.")
    print("  (train.py will automatically load the pre-trained weights)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Physics-informed Pre-training")
    parser.add_argument("--epochs",       type=int,   default=PRETRAIN_EPOCHS,        help="학습 에포크 수")
    parser.add_argument("--lr",           type=float, default=PRETRAIN_LR,             help="학습률")
    parser.add_argument("--batch-size",   type=int,   default=PRETRAIN_BATCH_SIZE,     help="배치 크기")
    parser.add_argument("--weight-decay", type=float, default=PRETRAIN_WEIGHT_DECAY,   help="L2 regularization")
    parser.add_argument("--val-ratio",    type=float, default=0.15,                    help="검증 비율")
    parser.add_argument("--patience",     type=int,   default=EARLY_STOPPING_PATIENCE, help="Early Stopping 기준 에포크")
    parser.add_argument("--seed",         type=int,   default=42,                      help="랜덤 시드")
    args = parser.parse_args()
    pretrain(args)
