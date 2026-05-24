"""Train HI regression model with leave-one-TDMS-case-out CV.

Target: Health Indicator in [0, 1].
Loss:   MSE (symmetric, no asymmetric pressure -> no safe-low collapse).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from config import (
    BATCH_SIZE,
    DATA2_DIR,
    DEVICE,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    LEARNING_RATE,
    MODELS_DIR,
    RANDOM_VAL_CASE,
    SCHEDULER_T0,
    STFT_FREQ_BINS,
    STFT_NOVERLAP,
    STFT_NPERSEG,
    STRIDE,
    TRAIN_DIR,
    VAL_CASE_DEFAULT,
    VAL_CASE_SEED,
    WEIGHT_DECAY,
    WINDOW_SIZE,
)
from data_loader import load_dataset
from inference import hi_sequence_to_rul
from model import asymmetric_rul_score_np, create_model


MODEL_PATH = MODELS_DIR / "HI.pt"


def _standardize(train_array: np.ndarray, val_array: np.ndarray):
    feature_shape = train_array.shape[2:]
    flat = train_array.reshape(-1, *feature_shape)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return (
        ((train_array - mean) / std).astype(np.float32),
        ((val_array - mean) / std).astype(np.float32),
        torch.tensor(mean, dtype=torch.float32),
        torch.tensor(std, dtype=torch.float32),
    )


def _tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.tensor(np.ascontiguousarray(arr), dtype=torch.float32)


def _balanced_weights(case_names: np.ndarray, sources: np.ndarray) -> torch.Tensor:
    keys = np.asarray([f"{s}::{c}" for s, c in zip(sources, case_names)])
    unique, counts = np.unique(keys, return_counts=True)
    count_map = dict(zip(unique, counts))
    w = np.asarray([1.0 / count_map[k] for k in keys], dtype=np.float64)
    return torch.tensor(w, dtype=torch.double)


def _arul(preds: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean(asymmetric_rul_score_np(preds, targets)))


def train(
    original_dir: Path = TRAIN_DIR,
    data2_dir: Path = DATA2_DIR,
    model_path: Path = MODEL_PATH,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
    max_samples: Optional[int] = None,
    balanced: bool = True,
    use_cache: bool = True,
    full_cv: bool = False,
    val_case: Optional[str] = None,
    seed: Optional[int] = None,
    include_data2: bool = True,
) -> list[Path]:
    """Train HI regression model.

    Single-fold by default. Pass full_cv=True for leave-one-TDMS-case-out
    4-fold ensemble training.
    """
    if seed is not None:
        import random as _r
        _r.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"[seed] all RNGs set to {seed}")

    print("=" * 72)
    print(f"HI regression training | seed={seed} | window={window_size} stride={stride}")
    print("=" * 72)

    X_vib, X_feat, hi, rul, metadata, baseline = load_dataset(
        original_dir=original_dir, data2_dir=data2_dir,
        include_original=True, include_data2=include_data2,
        window_size=window_size, stride=stride,
        max_samples=max_samples, use_cache=use_cache,
    )
    print(f"Dataset: {len(hi)} windows")
    print(f"  vib={X_vib.shape} feat={X_feat.shape}")
    print(f"  HI range: [{hi.min():.3f}, {hi.max():.3f}]  RUL range: [{rul.min():.0f}, {rul.max():.0f}]")
    print(metadata.groupby(["source", "case_name"]).size().to_string())

    sources = metadata["source"].astype(str).to_numpy()
    case_names = metadata["case_name"].astype(str).to_numpy()
    tdms_cases = sorted(np.unique(case_names[sources == "original"]).tolist())
    if not tdms_cases:
        raise ValueError("No TDMS (source='original') cases found.")

    if full_cv:
        held_out_cases = tdms_cases
        print(f"\nCV strategy: full leave-one-TDMS-case-out ({len(tdms_cases)} folds).")
    else:
        if val_case is None:
            if RANDOM_VAL_CASE:
                import random
                rng = random.Random(VAL_CASE_SEED)
                val_case = rng.choice(tdms_cases)
                seed_note = f"seed={VAL_CASE_SEED}" if VAL_CASE_SEED is not None else "non-reproducible"
                print(f"\n[config] RANDOM_VAL_CASE=True -> picked val_case={val_case} ({seed_note})")
            else:
                val_case = VAL_CASE_DEFAULT
                print(f"\n[config] RANDOM_VAL_CASE=False -> using VAL_CASE_DEFAULT={val_case}")
        if val_case not in tdms_cases:
            raise ValueError(f"val_case={val_case!r} not in {tdms_cases}")
        held_out_cases = [val_case]
        print(f"CV strategy: SINGLE-FOLD (val_case={val_case}).")

    saved: list[Path] = []
    fold_summaries = []

    for fold, held_out in enumerate(held_out_cases, start=1):
        val_mask = (sources == "original") & (case_names == held_out)
        train_mask = ~val_mask
        val_idx = np.where(val_mask)[0]
        train_idx = np.where(train_mask)[0]
        print(f"\n========== Fold {fold}/{len(held_out_cases)} (held-out: {held_out}) ==========")
        print(f"Train={len(train_idx)} | Val={len(val_idx)}")

        Xv_tr, Xv_va, vib_mean, vib_std = _standardize(X_vib[train_idx], X_vib[val_idx])
        Xf_tr, Xf_va, feat_mean, feat_std = _standardize(X_feat[train_idx], X_feat[val_idx])
        y_tr = hi[train_idx].astype(np.float32)
        y_va = hi[val_idx].astype(np.float32)

        train_ds = TensorDataset(_tensor(Xv_tr), _tensor(Xf_tr), _tensor(y_tr).unsqueeze(1))
        if balanced:
            w = _balanced_weights(
                metadata.iloc[train_idx]["case_name"].astype(str).to_numpy(),
                metadata.iloc[train_idx]["source"].astype(str).to_numpy(),
            )
            sampler = WeightedRandomSampler(w, num_samples=len(train_idx), replacement=True)
            train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
        else:
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(
            TensorDataset(_tensor(Xv_va), _tensor(Xf_va), _tensor(y_va).unsqueeze(1)),
            batch_size=batch_size, shuffle=False,
        )

        device = torch.device(DEVICE)
        model = create_model(
            vibration_channels=X_vib.shape[2],
            vibration_features=X_vib.shape[3],
            handcrafted_dim=X_feat.shape[-1],
        ).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=SCHEDULER_T0, T_mult=2)

        best_val = float("inf")
        best_state = None
        patience = 0
        for epoch in range(1, epochs + 1):
            model.train()
            tr_loss = 0.0
            for bv, bf, by in train_loader:
                bv, bf, by = bv.to(device), bf.to(device), by.to(device)
                optimizer.zero_grad()
                loss = criterion(model(bv, bf), by)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                tr_loss += loss.item() * by.size(0)

            model.eval()
            va_loss = 0.0
            with torch.no_grad():
                for bv, bf, by in val_loader:
                    bv, bf, by = bv.to(device), bf.to(device), by.to(device)
                    va_loss += criterion(model(bv, bf), by).item() * by.size(0)
            tr_loss /= len(train_loader.dataset)
            va_loss /= len(val_loader.dataset)
            scheduler.step()

            if va_loss < best_val:
                best_val = va_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= EARLY_STOPPING_PATIENCE:
                    print(f"  [early-stop] epoch={epoch}")
                    break
            if epoch % 5 == 0 or epoch == 1:
                print(f"  epoch {epoch:03d}/{epochs} | train_mse={tr_loss:.5f} val_mse={va_loss:.5f} patience={patience}")

        if best_state is not None:
            model.load_state_dict(best_state)

        # Inference on val set: predict HI, then convert to RUL per case (stage 2)
        model.eval()
        hi_preds = []
        with torch.no_grad():
            for bv, bf, _ in val_loader:
                hi_preds.append(model(bv.to(device), bf.to(device)).squeeze(1).cpu().numpy())
        hi_preds = np.concatenate(hi_preds).astype(np.float64)

        val_meta = metadata.iloc[val_idx].reset_index(drop=True)
        true_rul = rul[val_idx].astype(np.float64)
        true_hi = hi[val_idx].astype(np.float64)
        pred_rul = _convert_val_hi_to_rul(val_meta, hi_preds)

        hi_mae = float(np.mean(np.abs(hi_preds - true_hi)))
        rul_mae = float(np.mean(np.abs(pred_rul - true_rul)))
        rul_arul = _arul(pred_rul, true_rul)
        print("\nValidation:")
        print(f"  HI  MAE={hi_mae:.4f}")
        print(f"  RUL MAE={rul_mae:.1f}  A_RUL={rul_arul:.4f}")
        fold_summaries.append({"fold": fold, "held_out": held_out, "hi_mae": hi_mae,
                               "rul_mae": rul_mae, "a_rul": rul_arul})

        seed_tag = f"_seed{seed}" if seed is not None else ""
        fold_path = Path(str(model_path).replace(".pt", f"{seed_tag}_fold{fold}.pt"))
        fold_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "window_size": window_size,
                "stride": stride,
                "stft_nperseg": STFT_NPERSEG,
                "stft_noverlap": STFT_NOVERLAP,
                "stft_freq_bins": STFT_FREQ_BINS,
                "vibration_channels": X_vib.shape[2],
                "vibration_features": X_vib.shape[3],
                "handcrafted_dim": X_feat.shape[-1],
                "vibration_mean": vib_mean,
                "vibration_std": vib_std,
                "feature_mean": feat_mean,
                "feature_std": feat_std,
                "degradation_baseline": baseline,
                "fold_summary": fold_summaries[-1],
                "seed": seed,
            },
            fold_path,
        )
        print(f"Saved {fold_path}")
        saved.append(fold_path)

        # Save per-fold OOF predictions for offline diagnosis
        oof_df = val_meta.copy()
        oof_df["hi_true"] = true_hi
        oof_df["hi_pred"] = hi_preds
        oof_df["rul_true"] = true_rul
        oof_df["rul_pred"] = pred_rul
        oof_df["fold"] = fold
        oof_df["held_out_case"] = held_out
        oof_path = Path(str(fold_path).replace(".pt", "_oof.parquet"))
        try:
            oof_df.to_parquet(oof_path, index=False)
        except Exception:
            oof_path = Path(str(oof_path).replace(".parquet", ".csv"))
            oof_df.to_csv(oof_path, index=False)
        print(f"Saved OOF predictions {oof_path}")

    print("\nFold summary:")
    for s in fold_summaries:
        print(f"  fold {s['fold']:>2} ({s['held_out']:>10}): "
              f"HI_MAE={s['hi_mae']:.4f}  RUL_MAE={s['rul_mae']:7.0f}  A_RUL={s['a_rul']:.4f}")
    if len(fold_summaries) > 1:
        avg_arul = np.mean([s["a_rul"] for s in fold_summaries])
        avg_hi = np.mean([s["hi_mae"] for s in fold_summaries])
        print(f"  AVERAGE: HI_MAE={avg_hi:.4f}  A_RUL={avg_arul:.4f}")
    return saved


def _convert_val_hi_to_rul(val_meta, hi_preds):
    """Convert per-window HI predictions to per-window RUL predictions.
    Group by case, fit curve to that case's HI sequence, then for each window
    compute RUL = predicted t_failure - window time_sec.
    """
    rul_pred = np.zeros(len(val_meta), dtype=np.float64)
    val_meta = val_meta.reset_index(drop=True)
    for case in val_meta["case_name"].unique():
        mask = (val_meta["case_name"] == case).to_numpy()
        idx = np.where(mask)[0]
        case_times = val_meta.loc[mask, "time_sec"].to_numpy(dtype=np.float64)
        case_max = float(val_meta.loc[mask, "case_max"].iloc[0])
        case_hi = hi_preds[mask]

        order = np.argsort(case_times)
        t_ordered = case_times[order]
        h_ordered = case_hi[order]
        for j, t_now in enumerate(t_ordered):
            t_fail = hi_sequence_to_rul(t_ordered[: j + 1], h_ordered[: j + 1], t_now, case_max)
            rul_pred[idx[order[j]]] = max(0.0, t_fail - t_now)
    return rul_pred


def main() -> None:
    p = argparse.ArgumentParser(description="Train HI regression model")
    p.add_argument("--original-dir", type=Path, default=TRAIN_DIR)
    p.add_argument("--data2-dir", type=Path, default=DATA2_DIR)
    p.add_argument("--model-path", type=Path, default=MODEL_PATH)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
    p.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    p.add_argument("--stride", type=int, default=STRIDE)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--no-balanced", action="store_true")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--full-cv", action="store_true")
    p.add_argument("--val-case", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no-data2", action="store_true",
                   help="Exclude data2 cases from training (TDMS-only).")
    args = p.parse_args()
    train(
        original_dir=args.original_dir, data2_dir=args.data2_dir,
        model_path=args.model_path, epochs=args.epochs,
        batch_size=args.batch_size, learning_rate=args.lr,
        window_size=args.window_size, stride=args.stride,
        max_samples=args.max_samples, balanced=not args.no_balanced,
        use_cache=not args.no_cache, full_cv=args.full_cv,
        val_case=args.val_case, seed=args.seed,
        include_data2=not args.no_data2,
    )


if __name__ == "__main__":
    main()
