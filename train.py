"""Train HI regression model with leave-one-TDMS-case-out CV.

Training loss = HI_LOSS_WEIGHT * Huber(HI) + HI_RANK_LOSS_WEIGHT * PairwiseRanking + λ1*Mono + λ2*Smooth
  - No direct RUL regression!
  - RUL is computed solely via Stage-2 extrapolation.
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
    HI_FAILURE_THRESHOLD,
    HI_LOSS_WEIGHT,
    HI_RANK_LOSS_WEIGHT,
    LATE_LIFE_WEIGHT,
    LEARNING_RATE,
    MODELS_DIR,
    RANDOM_VAL_CASE,
    RESULTS_DIR,
    SCHEDULER_T0,
    STAGE_BOUNDARIES,
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
from stage2 import (
    _FALLBACK_RUL_CAP,
    _RUL_BEFORE_FDP,
    compute_stage2_trajectory,
    kalman_filter_hi,
    _FDP_THRESHOLD,
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
    times: np.ndarray,
    epoch_label: str,
    save_dir: Path,
) -> None:
    """Generates 3 required plots: HI trajectory, Stage-2 fitting, RUL trajectory."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    epoch_dir = save_dir / f"epoch_{epoch_label}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    case_label = save_dir.name

    t = times
    pred_hi = np.clip(pred_hi, 0.0, 1.0)
    hi_raw_kf = kalman_filter_hi(pred_hi)
    hi_kf = np.maximum.accumulate(hi_raw_kf)

    # 1. HI Trajectory Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, true_hi, label="GT HI", color="k", lw=2, linestyle="--")
    ax.plot(t, pred_hi, label="Raw Pred HI", color="C0", alpha=0.4)
    ax.plot(t, hi_kf, label="KF Filtered (Monotonic) HI", color="C1", lw=2)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("HI [0, 1]")
    ax.set_title(f"[{case_label}] HI Trajectory (Epoch {epoch_label})")
    ax.legend()
    fig.tight_layout()
    plt.savefig(epoch_dir / "hi_traj.png", dpi=120)
    plt.close(fig)

    # 2. Stage-2 Fitting Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, pred_hi, color="C0", alpha=0.3, label="Raw Pred HI")
    ax.plot(t, hi_kf, color="C1", lw=2, label="KF Filtered HI")
    ax.axhline(HI_FAILURE_THRESHOLD, color="r", linestyle="--", label="Failure Threshold")
    ax.axhline(_FDP_THRESHOLD, color="gray", linestyle=":", label="FDP Threshold")
    t_now = float(t[-1])

    if hi_kf[-1] > _FDP_THRESHOLD:
        from stage2 import _WLS_RECENCY_DECAY
        n = len(t)
        rolling = min(300, n)
            
        t_window = t[-rolling:]
        hi_window = hi_raw_kf[-rolling:]
        hi_window = np.maximum.accumulate(hi_window)
        t_offset = t_window[0]
        t_norm = t_window - t_offset
        w = np.exp(_WLS_RECENCY_DECAY * np.arange(rolling, dtype=np.float64) / max(rolling - 1, 1))

        try:
            epsilon = 1e-6
            y_log = np.log(np.clip(hi_window, epsilon, None))
            coeffs = np.polyfit(t_norm, y_log, 1, w=w)
            b1, b0 = coeffs[0], coeffs[1]
            
            if b1 > 1e-8:
                y_target = np.log(HI_FAILURE_THRESHOLD)
                t_fail_norm = (y_target - b0) / b1
                t_fail = t_fail_norm + t_offset
                
                t_ext = np.linspace(t_offset, max(t_fail * 1.05, t_now * 1.2), 120)
                hi_ext = np.exp(b1 * (t_ext - t_offset) + b0)
                ax.plot(t_ext, hi_ext, color="C3", lw=2, label="Log-Linear Extrapolation")
                
                rul = max(t_fail - t_now, 0.0)
                ax.scatter([t_fail], [HI_FAILURE_THRESHOLD], color="r", marker="*",
                           s=200, zorder=5,
                           label=f"Pred Failure (t={t_fail:.0f}, RUL={rul:.0f})")
                ax.set_xlim(float(t[0]) * 0.9, max(t_fail * 1.1, t_now * 1.2))
        except:
            pass
    else:
        ax.text(0.05, 0.92, "FDP not triggered (steady stage)",
                transform=ax.transAxes, color="gray", fontsize=10)

    ax.set_ylim(-0.05, 1.15)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("HI")
    ax.set_title(f"[{case_label}] Stage-2 Exp Fit (Epoch {epoch_label})")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    plt.savefig(epoch_dir / "stage2_fit.png", dpi=120)
    plt.close(fig)

    # 3. RUL Trajectory Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    # Replace fallback caps with NaN for cleaner plotting
    plot_pred_rul = np.where(pred_rul >= _FALLBACK_RUL_CAP * 0.9, np.nan, pred_rul)
    
    ax.plot(t, true_rul / 3600, label="GT RUL", color="k", lw=2, linestyle="--")
    ax.plot(t, plot_pred_rul / 3600, label="Predicted RUL", color="C0", lw=2)
    
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RUL (h)")
    ax.set_title(f"[{case_label}] RUL Trajectory (Epoch {epoch_label})")
    ax.legend()
    fig.tight_layout()
    plt.savefig(epoch_dir / "rul_traj.png", dpi=120)
    plt.close(fig)


def _tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.tensor(np.ascontiguousarray(arr), dtype=torch.float32)

def _balanced_weights(
    case_names: np.ndarray,
    sources: np.ndarray,
) -> torch.Tensor:
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
    if seed is not None:
        import random as _r
        _r.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"[seed] all RNGs set to {seed}")

    print("=" * 72)
    print(f"HI Trajectory training | seed={seed} | window={window_size} stride={stride}")
    print("=" * 72)

    X_vib, X_feat, hi, rul, metadata, baseline = load_dataset(
        original_dir=original_dir, data2_dir=data2_dir,
        include_original=True, include_data2=include_data2,
        window_size=window_size, stride=stride,
        max_samples=max_samples, use_cache=use_cache,
    )

    sources = metadata["source"].astype(str).to_numpy()
    case_names = metadata["case_name"].astype(str).to_numpy()
    tdms_cases = sorted(np.unique(case_names[sources == "original"]).tolist())
    if not tdms_cases:
        raise ValueError("No TDMS cases found.")

    if full_cv:
        held_out_cases = tdms_cases
    else:
        if val_case is None:
            val_case = VAL_CASE_DEFAULT
        held_out_cases = [val_case]

    saved: list[Path] = []
    fold_summaries = []

    for fold, held_out in enumerate(held_out_cases, start=1):
        val_mask = (sources == "original") & (case_names == held_out)
        train_mask = ~val_mask
        val_idx = np.where(val_mask)[0]
        train_idx = np.where(train_mask)[0]
        print(f"\n========== Fold {fold}/{len(held_out_cases)} (held-out: {held_out}) ==========")

        Xv_tr = X_vib[train_idx]
        Xv_va = X_vib[val_idx]
        Xf_tr = X_feat[train_idx]
        Xf_va = X_feat[val_idx]

        hi_tr = hi[train_idx].astype(np.float32)
        hi_va = hi[val_idx].astype(np.float32)

        case_max_tr = np.maximum(metadata.iloc[train_idx]["case_max"].to_numpy(np.float32), 1.0)
        case_max_va = np.maximum(metadata.iloc[val_idx]["case_max"].to_numpy(np.float32), 1.0)
        elapsed_tr = metadata.iloc[train_idx]["time_sec"].to_numpy(np.float32) / case_max_tr
        elapsed_va = metadata.iloc[val_idx]["time_sec"].to_numpy(np.float32) / case_max_va

        # Stage 2 lifetime prior — train fold cases 의 lognormal fit.
        # Si et al. 2011 review 의 표준. LOCO 각 fold 별로 *학습 데이터만* 사용 → leak-free.
        # Track 1 의 60000s static cap 을 conditional MRL 로 교체.
        train_case_max = (
            metadata.iloc[train_idx][["case_name", "case_max"]]
            .drop_duplicates(subset="case_name")["case_max"]
            .to_numpy(np.float64)
        )
        from scipy.stats import lognorm as _lognorm
        try:
            _shape, _loc, _scale = _lognorm.fit(train_case_max, floc=0)
            lifetime_prior = {"mu": float(np.log(_scale)), "sigma": float(_shape)}
            _med = float(_scale)
            _mn  = float(np.exp(lifetime_prior["mu"] + lifetime_prior["sigma"] ** 2 / 2))
            print(f"  Lifetime prior (lognormal): μ={lifetime_prior['mu']:.2f}  "
                  f"σ={lifetime_prior['sigma']:.3f}  median={_med:.0f}s  mean={_mn:.0f}s "
                  f"(N={len(train_case_max)} train cases)")
        except Exception as _e:
            print(f"  Lifetime prior fit failed: {_e} — fallback to legacy static cap")
            lifetime_prior = None

        life_frac_tr = metadata.iloc[train_idx]["time_sec"].to_numpy(np.float32) / case_max_tr
        life_frac_va = metadata.iloc[val_idx]["time_sec"].to_numpy(np.float32) / case_max_va
        late_weight_tr = np.where(life_frac_tr >= 0.8, LATE_LIFE_WEIGHT, 1.0).astype(np.float32)
        # Stage classification labels: lifetime quantile → 0~3 class
        stage_tr = np.digitize(life_frac_tr, STAGE_BOUNDARIES).astype(np.int64)
        stage_va = np.digitize(life_frac_va, STAGE_BOUNDARIES).astype(np.int64)

        tr_case_names = metadata.iloc[train_idx]["case_name"].astype(str).to_numpy()
        unique_tr_cases = sorted(np.unique(tr_case_names).tolist())
        case_to_int = {c: i for i, c in enumerate(unique_tr_cases)}
        case_int_tr = torch.tensor([case_to_int[c] for c in tr_case_names], dtype=torch.long)
        
        va_case_names = metadata.iloc[val_idx]["case_name"].astype(str).to_numpy()
        case_int_va = torch.tensor([case_to_int.get(c, -1) for c in va_case_names], dtype=torch.long)

        train_ds = TensorDataset(
            _tensor(Xv_tr), _tensor(Xf_tr),
            _tensor(hi_tr).unsqueeze(1),
            case_int_tr,
            _tensor(elapsed_tr).unsqueeze(1),
            _tensor(late_weight_tr).unsqueeze(1),
            torch.tensor(stage_tr, dtype=torch.long),
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
                          _tensor(hi_va).unsqueeze(1),
                          case_int_va,
                          _tensor(elapsed_va).unsqueeze(1),
                          torch.tensor(stage_va, dtype=torch.long)),
            batch_size=batch_size, shuffle=False,
        )

        device = torch.device(DEVICE)
        model = create_model(
            vibration_channels=X_vib.shape[2],
            vibration_features=X_vib.shape[3],
            handcrafted_dim=X_feat.shape[-1],
        ).to(device)
        criterion = TrainingLoss(lambda1=0.1, lambda2=0.01)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
        
        warmup_epochs = max(1, epochs // 5)
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=1e-5)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

        best_val_score = -float("inf")
        best_epoch = 0
        best_state = None
        patience = 0
        recent_scores: list[float] = []
        plot_dir = RESULTS_DIR / "trajectory" / f"{held_out}_fold{fold}"
        
        val_meta = metadata.iloc[val_idx].reset_index(drop=True)
        ep_true_rul = rul[val_idx].astype(np.float64)
        ep_true_hi  = hi[val_idx].astype(np.float64)
        val_times_sec = val_meta["time_sec"].to_numpy(np.float64)

        for epoch in range(1, epochs + 1):
            model.train()
            tr_hi = tr_hi_rank = tr_mono = tr_smooth = tr_stage = tr_tot = 0.0
            for bv, bf, by_hi, by_case, by_elapsed, by_weight, by_stage in train_loader:
                bv, bf = bv.to(device), bf.to(device)
                by_hi = by_hi.to(device)
                by_case = by_case.to(device)
                by_elapsed = by_elapsed.to(device)
                by_weight = by_weight.to(device)
                by_stage = by_stage.to(device)

                optimizer.zero_grad()
                pred_hi, stage_logits = model(bv, bf, by_elapsed, return_stage=True)
                loss, l_hi, l_hi_rank, l_mono, l_smooth, l_stage = criterion(
                    pred_hi, by_hi, by_case, by_elapsed, by_weight,
                    stage_logits=stage_logits, stage_labels=by_stage,
                )

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                n = by_hi.size(0)
                tr_hi      += l_hi.item()      * n
                tr_hi_rank += l_hi_rank.item() * n
                tr_mono    += l_mono.item()    * n
                tr_smooth  += l_smooth.item()  * n
                tr_stage   += l_stage.item()   * n
                tr_tot     += loss.item()      * n

            model.eval()
            va_hi = va_hi_rank = va_mono = va_smooth = va_stage = va_tot = 0.0
            _ep_hi = []
            with torch.no_grad():
                for bv, bf, by_hi, by_case, by_elapsed, by_stage in val_loader:
                    bv, bf = bv.to(device), bf.to(device)
                    by_hi = by_hi.to(device)
                    by_case = by_case.to(device)
                    by_elapsed = by_elapsed.to(device)
                    by_stage = by_stage.to(device)

                    pred_hi_b, stage_logits_b = model(bv, bf, by_elapsed, return_stage=True)
                    loss, l_hi, l_hi_rank, l_mono, l_smooth, l_stage = criterion(
                        pred_hi_b, by_hi, by_case, by_elapsed, None,
                        stage_logits=stage_logits_b, stage_labels=by_stage,
                    )

                    n = by_hi.size(0)
                    va_hi      += l_hi.item()      * n
                    va_hi_rank += l_hi_rank.item() * n
                    va_mono    += l_mono.item()    * n
                    va_smooth  += l_smooth.item()  * n
                    va_stage   += l_stage.item()   * n
                    va_tot     += loss.item()      * n
                    _ep_hi.append(np.array(pred_hi_b.squeeze(1).cpu().tolist(), dtype=np.float32))

            n_tr, n_va = len(train_loader.dataset), len(val_loader.dataset)
            tr_hi /= n_tr; tr_hi_rank /= n_tr; tr_mono /= n_tr; tr_smooth /= n_tr; tr_stage /= n_tr; tr_tot /= n_tr
            va_hi /= n_va; va_hi_rank /= n_va; va_mono /= n_va; va_smooth /= n_va; va_stage /= n_va; va_tot /= n_va
            scheduler.step()

            ep_pred_hi  = np.concatenate(_ep_hi).astype(np.float64)
            
            s2_rul = compute_stage2_trajectory(val_times_sec, ep_pred_hi, HI_FAILURE_THRESHOLD,
                                                lifetime_prior=lifetime_prior)
            
            # --- STAGE-2 STABILITY MONITORING ---
            # 1. RUL Stability
            valid_rul_idx = np.where(s2_rul < 199_999)[0]
            rul_stability = float(np.std(np.diff(s2_rul[valid_rul_idx]))) if len(valid_rul_idx) > 1 else 10000.0
            
            # 2. HI Monotonicity (raw pred)
            neg_diffs = float(np.sum(np.diff(ep_pred_hi) < 0))
            
            # 3. Fit Success Ratio
            num_rejected = len(s2_rul) - len(valid_rul_idx)
            fit_success_ratio = 1.0 - (num_rejected / max(1, len(s2_rul)))
            
            # 4. HI MAE
            hi_mae = float(np.mean(np.abs(ep_pred_hi - ep_true_hi)))
            
            # 5. Late-life A_RUL
            n_val = len(ep_true_rul)
            n_late = max(1, n_val // 10)
            late_arul = _arul(s2_rul[-n_late:], ep_true_rul[-n_late:])

            # Checkpoint Selection Score (Prioritizing Stability & Physics)
            # Higher score is better
            score = (
                - hi_mae * 10.0 
                - neg_diffs * 0.01 
                - rul_stability * 0.0001 
                + fit_success_ratio * 2.0 
                + late_arul * 1.0
            )

            recent_scores.append(score)
            if len(recent_scores) > 3:
                recent_scores.pop(0)
            smoothed_score = float(np.mean(recent_scores))

            if smoothed_score > best_val_score:
                best_val_score = smoothed_score
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
                _make_trajectory_plot(ep_true_rul, s2_rul, ep_true_hi, ep_pred_hi, val_times_sec, "best", plot_dir)
            else:
                patience += 1
                if patience >= EARLY_STOPPING_PATIENCE:
                    print(f"  [early-stop] epoch={epoch}  best_score={best_val_score:.4f} @ ep{best_epoch}")
                    _make_trajectory_plot(ep_true_rul, s2_rul, ep_true_hi, ep_pred_hi, val_times_sec, "last", plot_dir)
                    break

            if epoch in PLOT_EPOCHS:
                _make_trajectory_plot(ep_true_rul, s2_rul, ep_true_hi, ep_pred_hi, val_times_sec, str(epoch).zfill(2), plot_dir)

            if epoch % 5 == 0 or epoch == 1:
                print(f"  epoch {epoch:03d}/{epochs} | va(tot={va_tot:.4f} hi={va_hi:.4f} mono={va_mono:.4f} sm={va_smooth:.4f})")
                print(f"    -> Metric: HI_MAE={hi_mae:.3f} | NegDiffs={neg_diffs:.0f} | RUL_Std={rul_stability:.0f} | FitRatio={fit_success_ratio:.2f} | Score={score:.3f}")

        if best_state is not None:
            model.load_state_dict(best_state)

        if patience < EARLY_STOPPING_PATIENCE:
            _make_trajectory_plot(ep_true_rul, s2_rul, ep_true_hi, ep_pred_hi, val_times_sec, "last", plot_dir)

        print(f"  [best Score={best_val_score:.4f} @ epoch={best_epoch}]")

        # Final Evaluation
        model.eval()
        hi_preds = []
        with torch.no_grad():
            for bv, bf, _, _, be, _ in val_loader:
                ph = model(bv.to(device), bf.to(device), be.to(device))
                hi_preds.append(np.array(ph.squeeze(1).cpu().tolist(), dtype=np.float32))
        hi_preds = np.concatenate(hi_preds).astype(np.float64)

        true_rul = rul[val_idx].astype(np.float64)
        true_hi = hi[val_idx].astype(np.float64)
        
        hi_mae = float(np.mean(np.abs(hi_preds - true_hi)))

        s2_rul = compute_stage2_trajectory(val_times_sec, hi_preds, HI_FAILURE_THRESHOLD)
        s2_arul_full = _arul(s2_rul, true_rul)
        s2_arul_late = _arul(s2_rul[-n_late:], true_rul[-n_late:])
        s2_arul_last = _arul(s2_rul[-1:], true_rul[-1:])
        s2_rul_mae   = float(np.mean(np.abs(s2_rul - true_rul)))
        
        valid_rul_idx = np.where(s2_rul < 199_999)[0]
        rul_stability = float(np.std(np.diff(s2_rul[valid_rul_idx]))) if len(valid_rul_idx) > 1 else 10000.0

        print("\nValidation Results:")
        print(f"  HI MAE = {hi_mae:.4f} | RUL Stability (std) = {rul_stability:.1f}")
        print(f"  [Stage-2] MAE={s2_rul_mae:7.0f} | A_RUL(full)={s2_arul_full:.4f} | A_RUL(last)={s2_arul_last:.4f}")

        fold_summaries.append({"fold": fold, "held_out": held_out,
                               "hi_mae": hi_mae,
                               "s2_a_rul": s2_arul_full,
                               "s2_last_arul": s2_arul_last,
                               "s2_rul_mae": s2_rul_mae,
                               "rul_stability": rul_stability})

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
                "degradation_baseline": baseline,
                "best_score": best_val_score,
                "best_epoch": best_epoch,
                "fold_summary": fold_summaries[-1],
                "seed": seed,
                "lifetime_prior": lifetime_prior,   # {"mu", "sigma"} or None — Stage 2 MRL fallback
            },
            fold_path,
        )
        print(f"Saved {fold_path}")

    print("\nFold summary:")
    for s in fold_summaries:
        print(f"  fold {s['fold']:>2} ({s['held_out']:>10}): "
              f"HI_MAE={s['hi_mae']:.4f}  "
              f"RUL_Stability={s['rul_stability']:.1f}  "
              f"[S2] RUL_MAE={s['s2_rul_mae']:7.0f}  "
              f"A_RUL={s['s2_a_rul']:.4f}")
    return saved


def main() -> None:
    p = argparse.ArgumentParser(description="Train single-head HI trajectory model")
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
