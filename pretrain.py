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

from config import DEVICE, MODELS_DIR
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
) -> tuple[DataLoader, DataLoader]:
    """
    pretrain_data.npz를 로드하고 Train/Val DataLoader를 반환한다.

    X_vib: (N, window_size, n_channels, vib_features)
    y_rul: (N,)

    모델 입력은 (batch, window_size, n_channels, vib_features)이므로
    DataLoader가 자동으로 배치 차원을 추가한다.
    """
    print(f"Loading pretrain data from: {data_path}")
    data = np.load(data_path)
    X_vib = data["X_vib"].astype(np.float32)  # (N, W, C, F)
    y_rul = data["y_rul"].astype(np.float32)   # (N,)

    print(f"  X_vib: {X_vib.shape} | y_rul: {y_rul.shape}")
    print(f"  RUL range: {y_rul.min():.1f} ~ {y_rul.max():.1f}")

    # 로그 스케일링 (실제 학습과 동일한 전처리)
    y_log = np.log1p(y_rul)
    print(f"  Log RUL range: {y_log.min():.3f} ~ {y_log.max():.3f}")

    # 채널 순서 변환: (N, W, C, F) -> (N, W, C, F) 유지
    # 단, 모델은 (batch, W, C, F) 형태로 받음. 여기서 W=window_size
    X_tensor = torch.from_numpy(X_vib)          # (N, W, C, F)
    y_tensor = torch.from_numpy(y_log).unsqueeze(1)  # (N, 1)

    dataset = TensorDataset(X_tensor, y_tensor)
    n_val = int(len(dataset) * val_ratio)
    n_train = len(dataset) - n_val

    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=gen)
    print(f"  Train: {n_train} | Val: {n_val}")
    return train_ds, val_ds


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
    train_ds, val_ds = load_pretrain_data(
        data_path=PRETRAIN_DATA,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

    # 모델 생성 (X_vib의 채널/특징 크기를 자동 감지)
    sample_x, _ = train_ds[0]
    # sample_x shape: (window_size, n_channels, vib_features)
    n_channels     = sample_x.shape[1]
    vib_features   = sample_x.shape[2]
    op_features    = 1  # 합성 데이터에는 운전 피처 없음 → 더미 1차원 사용

    print(f"\nModel config: channels={n_channels} | vib_features={vib_features} | op_features={op_features}")
    model = create_model(
        vibration_channels=n_channels,
        operation_features=op_features,
        vibration_features=vib_features,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    criterion = CombinedLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(10, args.epochs // 4), T_mult=2
    )

    best_val_loss = float("inf")
    patience_count = 0

    print(f"\nStarting pre-training for {args.epochs} epochs...\n")

    for epoch in range(1, args.epochs + 1):
        # ── Train ──────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)      # (B, W, C, F)
            batch_y = batch_y.to(device)      # (B, 1)

            # 합성 데이터에는 운전 피처가 없으므로 더미 0 텐서 사용
            batch_op = torch.zeros(batch_x.shape[0], batch_x.shape[0], op_features, device=device)
            # 실제로는 batch_op shape이 (B, W, op_features) 여야 함
            batch_op = torch.zeros(batch_x.shape[0], batch_x.shape[1], op_features, device=device)

            optimizer.zero_grad()
            pred = model(batch_x, batch_op)
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
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                batch_op = torch.zeros(batch_x.shape[0], batch_x.shape[1], op_features, device=device)
                pred = model(batch_x, batch_op)
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
            torch.save(model.state_dict(), PRETRAINED_OUT)
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"\n[Early Stop] No improvement for {args.patience} epochs. Stopping.")
                break

    print(f"\n[Done] Best val_loss = {best_val_loss:.4f}")
    print(f"Pre-trained model saved -> {PRETRAINED_OUT}")
    print("=" * 60)
    print("\nNext step: run 'python train.py' for fine-tuning with real data.")
    print("  (train.py will automatically load the pre-trained weights)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Physics-informed Pre-training")
    parser.add_argument("--epochs",     type=int,   default=60,   help="학습 에포크 수")
    parser.add_argument("--lr",         type=float, default=3e-4, help="학습률")
    parser.add_argument("--batch-size", type=int,   default=32,   help="배치 크기")
    parser.add_argument("--val-ratio",  type=float, default=0.15, help="검증 비율")
    parser.add_argument("--patience",   type=int,   default=15,   help="Early Stopping 기준 에포크")
    parser.add_argument("--seed",       type=int,   default=42,   help="랜덤 시드")
    args = parser.parse_args()
    pretrain(args)
