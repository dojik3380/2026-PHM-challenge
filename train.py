"""Train DualHeadModel (RUL + HI) with leave-one-TDMS-case-out CV.

Training loss = 0.5*MSE(RUL_log) + 0.3*MSE(HI) + 0.2*PairwiseRanking
  - No asymmetric bias → no DENORM_SCALE post-hoc correction needed.
  - A_RUL (asymmetric metric) is reported at validation time only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

sys.stdout.reconfigure(line_buffering=True)  # flush every line when redirected to file

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
    HI_LOSS_WEIGHT,
    LEARNING_RATE,
    MODELS_DIR,
    RANDOM_VAL_CASE,
    RANKING_LOSS_WEIGHT,
    RESULTS_DIR,
    RUL_LOSS_WEIGHT,
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
from model import TrainingLoss, asymmetric_rul_score_np, create_model


MODEL_PATH = MODELS_DIR / "RUL.pt"
PLOT_EPOCHS = frozenset({1, 3, 5, 10, 15, 20, 25, 30})


def _make_trajectory_plot(
    true_rul: np.ndarray,
    pred_rul: np.ndarray,
    true_hi: np.ndarray,
    pred_hi: np.ndarray,
    epoch_label: str,
    save_dir: Path,
) -> None:
    """4-panel trajectory plot: RUL, HI, histogram, Spearman rank scatter."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy.stats import spearmanr
    except ImportError:
        return

    rho, _ = spearmanr(true_rul, pred_rul)
    t = np.arange(len(true_rul))
    n = len(true_rul)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    case_label = save_dir.parent.name  # e.g. "Train1_fold1"
    fig.suptitle(f"Val: {case_label} — Epoch {epoch_label}  |  Spearman ρ={rho:.3f}", fontsize=12)

    # 1. True vs Pred RUL (hours)
    ax = axes[0, 0]
    ax.plot(t, true_rul / 3600, label="True", color="C0", lw=1.5)
    ax.plot(t, pred_rul / 3600, label="Pred", color="C1", lw=1.2, alpha=0.85)
    ax.set_xlabel("Window index")
    ax.set_ylabel("RUL (h)")
    ax.set_title("True vs Pred RUL")
    ax.legend()

    # 2. HI trajectory
    ax = axes[0, 1]
    ax.plot(t, true_hi, label="True HI", color="C0", lw=1.5)
    ax.plot(t, pred_hi, label="Pred HI", color="C2", lw=1.2, alpha=0.85)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Window index")
    ax.set_ylabel("HI [0, 1]")
    ax.set_title("HI Trajectory")
    ax.legend()

    # 3. RUL prediction distribution
    ax = axes[1, 0]
    bins = min(30, max(10, n // 5))
    ax.hist(true_rul / 3600, bins=bins, alpha=0.6, label="True", color="C0", density=True)
    ax.hist(pred_rul / 3600, bins=bins, alpha=0.6, label="Pred", color="C1", density=True)
    ax.set_xlabel("RUL (h)")
    ax.set_ylabel("Density")
    ax.set_title("RUL Distribution")
    ax.legend()

    # 4. Spearman monotonicity: true rank vs pred rank
    ax = axes[1, 1]
    true_rank = np.argsort(np.argsort(-true_rul))
    pred_rank = np.argsort(np.argsort(-pred_rul))
    ax.scatter(true_rank, pred_rank, s=6, alpha=0.4, color="C3")
    ax.plot([0, n - 1], [0, n - 1], "k--", lw=1, alpha=0.4, label="Perfect")
    ax.set_xlabel("True RUL rank")
    ax.set_ylabel("Pred RUL rank")
    ax.set_title(f"Rank Monotonicity  (ρ={rho:.3f})")
    ax.legend(fontsize=8)

    plt.tight_layout()
    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / f"epoch_{epoch_label}.png"
    plt.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] {out.name}")


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
    """1/count per case so every case contributes equally regardless of window count."""
    keys = np.asarray([f"{s}::{c}" for s, c in zip(sources, case_names)])
    unique, counts = np.unique(keys, return_counts=True)
    count_map = dict(zip(unique, counts))
    w = np.asarray([1.0 / count_map[k] for k in keys], dtype=np.float64)
    w = w / w.sum() * len(w)
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
    """Train DualHeadModel.

    Single-fold by default. Pass full_cv=True for 4-fold leave-one-TDMS-case-out.
    """
    if seed is not None:
        import random as _r
        _r.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"[seed] all RNGs set to {seed}")

    print("=" * 72)
    print(f"Dual-head training | seed={seed} | window={window_size} stride={stride}")
    print(f"  loss: {RUL_LOSS_WEIGHT}*Huber(RUL_log) + {HI_LOSS_WEIGHT}*MSE(HI) + {RANKING_LOSS_WEIGHT}*PairwiseRank")
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
                _val_seed = seed if seed is not None else VAL_CASE_SEED
                rng = random.Random(_val_seed)  # None → system time (truly random)
                val_case = rng.choice(tdms_cases)
                note = f"seed={_val_seed}" if _val_seed is not None else "non-reproducible"
                print(f"\n[config] RANDOM_VAL_CASE=True -> picked val_case={val_case} ({note})")
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

        # RUL target in log space; HI target directly.
        rul_tr_log = np.log1p(rul[train_idx]).astype(np.float32)
        rul_va_log = np.log1p(rul[val_idx]).astype(np.float32)
        hi_tr = hi[train_idx].astype(np.float32)
        hi_va = hi[val_idx].astype(np.float32)

        # Integer case IDs for within-case ranking loss (train only)
        tr_case_names = metadata.iloc[train_idx]["case_name"].astype(str).to_numpy()
        unique_tr_cases = sorted(np.unique(tr_case_names).tolist())
        case_to_int = {c: i for i, c in enumerate(unique_tr_cases)}
        case_int_tr = torch.tensor(
            [case_to_int[c] for c in tr_case_names], dtype=torch.long
        )

        train_ds = TensorDataset(
            _tensor(Xv_tr), _tensor(Xf_tr),
            _tensor(rul_tr_log).unsqueeze(1),
            _tensor(hi_tr).unsqueeze(1),
            case_int_tr,
        )
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
            TensorDataset(_tensor(Xv_va), _tensor(Xf_va),
                          _tensor(rul_va_log).unsqueeze(1),
                          _tensor(hi_va).unsqueeze(1)),
            batch_size=batch_size, shuffle=False,
        )

        device = torch.device(DEVICE)
        model = create_model(
            vibration_channels=X_vib.shape[2],
            vibration_features=X_vib.shape[3],
            handcrafted_dim=X_feat.shape[-1],
        ).to(device)
        criterion = TrainingLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

        # Early stopping: maximise validation A_RUL (closer to 1.0 = better)
        best_val_arul = -float("inf")
        best_arul_epoch = 0
        best_state = None
        patience = 0
        plot_dir = RESULTS_DIR / "trajectory" / f"{held_out}_fold{fold}"
        # Pre-compute fixed val targets once
        ep_true_rul = rul[val_idx].astype(np.float64)
        ep_true_hi  = hi[val_idx].astype(np.float64)
        ep_pred_rul = np.zeros_like(ep_true_rul)
        ep_pred_hi  = np.zeros(len(val_idx), dtype=np.float64)

        for epoch in range(1, epochs + 1):
            model.train()
            tr_rul = tr_hi = tr_rank = tr_tot = 0.0
            for bv, bf, by_rul, by_hi, by_case in train_loader:
                bv, bf = bv.to(device), bf.to(device)
                by_rul, by_hi = by_rul.to(device), by_hi.to(device)
                by_case = by_case.to(device)
                optimizer.zero_grad()
                pred_rul, pred_hi = model(bv, bf)
                loss, l_rul, l_hi, l_rank = criterion(pred_rul, pred_hi, by_rul, by_hi, by_case)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                n = by_rul.size(0)
                tr_rul += l_rul.item() * n
                tr_hi += l_hi.item() * n
                tr_rank += l_rank.item() * n
                tr_tot += loss.item() * n

            model.eval()
            va_rul = va_hi = va_rank = va_tot = 0.0
            _ep_rul_log, _ep_hi = [], []
            with torch.no_grad():
                for bv, bf, by_rul, by_hi in val_loader:
                    bv, bf = bv.to(device), bf.to(device)
                    by_rul, by_hi = by_rul.to(device), by_hi.to(device)
                    pred_rul_b, pred_hi_b = model(bv, bf)
                    loss, l_rul, l_hi, l_rank = criterion(pred_rul_b, pred_hi_b, by_rul, by_hi)
                    n = by_rul.size(0)
                    va_rul += l_rul.item() * n
                    va_hi += l_hi.item() * n
                    va_rank += l_rank.item() * n
                    va_tot += loss.item() * n
                    _ep_rul_log.append(np.array(pred_rul_b.squeeze(1).cpu().tolist(), dtype=np.float32))
                    _ep_hi.append(np.array(pred_hi_b.squeeze(1).cpu().tolist(), dtype=np.float32))
            n_tr, n_va = len(train_loader.dataset), len(val_loader.dataset)
            tr_rul /= n_tr; tr_hi /= n_tr; tr_rank /= n_tr; tr_tot /= n_tr
            va_rul /= n_va; va_hi /= n_va; va_rank /= n_va; va_tot /= n_va
            scheduler.step()
            # Convert log-preds → seconds for A_RUL and plots
            ep_pred_rul = np.maximum(np.expm1(np.clip(np.concatenate(_ep_rul_log), 0.0, 11.5)), 0.0)
            ep_pred_hi  = np.concatenate(_ep_hi).astype(np.float64)
            epoch_arul  = _arul(ep_pred_rul, ep_true_rul)

            # Early stopping: maximise A_RUL — this is the competition metric.
            if epoch_arul > best_val_arul:
                best_val_arul = epoch_arul
                best_arul_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
                _make_trajectory_plot(ep_true_rul, ep_pred_rul, ep_true_hi, ep_pred_hi, "best", plot_dir)
            else:
                patience += 1
                if patience >= EARLY_STOPPING_PATIENCE:
                    print(f"  [early-stop] epoch={epoch}  best_A_RUL={best_val_arul:.4f} @ ep{best_arul_epoch}")
                    _make_trajectory_plot(ep_true_rul, ep_pred_rul, ep_true_hi, ep_pred_hi, "last", plot_dir)
                    break

            # Trajectory plots at fixed epochs
            if epoch in PLOT_EPOCHS:
                _make_trajectory_plot(ep_true_rul, ep_pred_rul, ep_true_hi, ep_pred_hi,
                                      str(epoch).zfill(2), plot_dir)

            if epoch % 5 == 0 or epoch == 1:
                print(f"  epoch {epoch:03d}/{epochs} | "
                      f"tr(rul={tr_rul:.4f} hi={tr_hi:.4f} rank={tr_rank:.4f} tot={tr_tot:.4f}) "
                      f"va(rul={va_rul:.4f} hi={va_hi:.4f} rank={va_rank:.4f} tot={va_tot:.4f}) "
                      f"A_RUL={epoch_arul:.4f} patience={patience}")

        if best_state is not None:
            model.load_state_dict(best_state)

        # "last" plot when training completed all epochs without early-stop
        if patience < EARLY_STOPPING_PATIENCE:
            _make_trajectory_plot(ep_true_rul, ep_pred_rul, ep_true_hi, ep_pred_hi, "last", plot_dir)

        print(f"  [best A_RUL={best_val_arul:.4f} @ epoch={best_arul_epoch}]")

        # Inference on val set
        model.eval()
        rul_log_preds, hi_preds = [], []
        with torch.no_grad():
            for bv, bf, _, _ in val_loader:
                pr, ph = model(bv.to(device), bf.to(device))
                rul_log_preds.append(np.array(pr.squeeze(1).cpu().tolist(), dtype=np.float32))
                hi_preds.append(np.array(ph.squeeze(1).cpu().tolist(), dtype=np.float32))
        rul_log_preds = np.concatenate(rul_log_preds).astype(np.float64)
        hi_preds = np.concatenate(hi_preds).astype(np.float64)

        # RUL head: log → seconds (no post-hoc scaling).
        rul_pred = np.maximum(np.expm1(np.clip(rul_log_preds, 0.0, 11.5)), 0.0)
        true_rul = rul[val_idx].astype(np.float64)
        true_hi = hi[val_idx].astype(np.float64)

        arul = _arul(rul_pred, true_rul)
        rul_mae = float(np.mean(np.abs(rul_pred - true_rul)))
        hi_mae = float(np.mean(np.abs(hi_preds - true_hi)))
        print("\nValidation:")
        print(f"  RUL  MAE={rul_mae:7.0f}  A_RUL={arul:.4f}")
        print(f"  HI   MAE={hi_mae:.4f}  (auxiliary head)")
        fold_summaries.append({"fold": fold, "held_out": held_out,
                               "rul_mae": rul_mae, "a_rul": arul,
                               "hi_mae": hi_mae})

        seed_tag = f"_seed{seed}" if seed is not None else ""
        fold_path = Path(str(model_path).replace(".pt", f"{seed_tag}_fold{fold}.pt"))
        fold_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),   # best A_RUL weights
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
                "denorm_scale": 1.0,
                "best_arul": best_val_arul,
                "best_arul_epoch": best_arul_epoch,
                "fold_summary": fold_summaries[-1],
                "seed": seed,
            },
            fold_path,
        )
        print(f"Saved {fold_path}  (best A_RUL={best_val_arul:.4f} @ ep{best_arul_epoch})")
        saved.append(fold_path)

        # Save per-fold OOF predictions
        val_meta = metadata.iloc[val_idx].reset_index(drop=True)
        oof_df = val_meta.copy()
        oof_df["rul_true"] = true_rul
        oof_df["rul_pred"] = rul_pred
        oof_df["hi_true"] = true_hi
        oof_df["hi_pred"] = hi_preds
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
              f"RUL_MAE={s['rul_mae']:7.0f}  A_RUL={s['a_rul']:.4f}  HI_MAE={s['hi_mae']:.4f}")
    if len(fold_summaries) > 1:
        avg_arul = np.mean([s["a_rul"] for s in fold_summaries])
        avg_rul_mae = np.mean([s["rul_mae"] for s in fold_summaries])
        avg_hi_mae = np.mean([s["hi_mae"] for s in fold_summaries])
        print(f"  AVERAGE: A_RUL={avg_arul:.4f}  RUL_MAE={avg_rul_mae:.0f}  HI_MAE={avg_hi_mae:.4f}")
    return saved


def main() -> None:
    p = argparse.ArgumentParser(description="Train dual-head (RUL + HI aux) model")
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
