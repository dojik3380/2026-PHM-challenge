import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import itertools

from config import DATA2_DIR, TRAIN_DIR, DEVICE, BATCH_SIZE, HI_FAILURE_THRESHOLD
from data_loader import load_dataset
from model import create_model, asymmetric_rul_score_np
from stage2 import compute_stage2_trajectory

def _arul(preds: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean(asymmetric_rul_score_np(preds, targets)))

def _tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.tensor(np.ascontiguousarray(arr), dtype=torch.float32)

def cache_predictions():
    print("Loading dataset...")
    X_vib, X_feat, hi, rul, metadata, baseline = load_dataset(
        original_dir=TRAIN_DIR, data2_dir=DATA2_DIR,
        include_original=True, include_data2=True,
    )
    
    sources = metadata["source"].astype(str).to_numpy()
    case_names = metadata["case_name"].astype(str).to_numpy()
    tdms_cases = sorted(np.unique(case_names[sources == "original"]).tolist())
    
    cached_folds = []
    
    device = torch.device(DEVICE)
    
    for fold, held_out in enumerate(tdms_cases, start=1):
        model_path = Path(f"models/RUL_seed42_fold{fold}.pt")
        if not model_path.exists():
            print(f"Skipping fold {fold}: {model_path} not found.")
            continue
            
        print(f"Loading predictions for fold {fold} (held_out: {held_out})")
        val_mask = (sources == "original") & (case_names == held_out)
        val_idx = np.where(val_mask)[0]
        
        Xv_va = X_vib[val_idx]
        Xf_va = X_feat[val_idx]
        case_max_va = np.maximum(metadata.iloc[val_idx]["case_max"].to_numpy(np.float32), 1.0)
        elapsed_va = metadata.iloc[val_idx]["time_sec"].to_numpy(np.float32) / case_max_va
        
        val_loader = DataLoader(
            TensorDataset(_tensor(Xv_va), _tensor(Xf_va), _tensor(elapsed_va).unsqueeze(1)),
            batch_size=BATCH_SIZE, shuffle=False
        )
        
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        model = create_model(
            vibration_channels=Xv_va.shape[2],
            vibration_features=Xv_va.shape[3],
            handcrafted_dim=Xf_va.shape[-1],
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        
        hi_preds = []
        with torch.no_grad():
            for bv, bf, be in val_loader:
                ph = model(bv.to(device), bf.to(device), be.to(device))
                hi_preds.append(np.array(ph.squeeze(1).cpu().tolist(), dtype=np.float32))
        hi_preds = np.concatenate(hi_preds).astype(np.float64)
        
        val_times_sec = metadata.iloc[val_idx]["time_sec"].to_numpy(np.float64)
        true_rul = rul[val_idx].astype(np.float64)
        
        cached_folds.append({
            "fold": fold,
            "held_out": held_out,
            "times": val_times_sec,
            "hi_preds": hi_preds,
            "true_rul": true_rul,
            "lifetime_prior": ckpt.get("lifetime_prior")
        })
        
    return cached_folds

def grid_search(cached_folds):
    print("\nStarting Fast Grid Search for Stage 2 Hyperparameters...")
    
    # Search Space
    grid = {
        "fdp_k_sigma": [2.0, 3.0, 4.0],
        "cqrl_quantile": [0.20, 0.25, 0.30, 0.35],
        "q_scale": [0.1, 1.0, 10.0],
        "r_scale": [0.1, 1.0, 10.0],
    }
    
    keys = list(grid.keys())
    values = list(grid.values())
    combinations = list(itertools.product(*values))
    
    best_score = -1.0
    best_params = None
    best_fold_scores = []
    
    print(f"Total combinations to evaluate: {len(combinations)}")
    
    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        
        fold_scores = []
        for c in cached_folds:
            s2_rul = compute_stage2_trajectory(
                c["times"], c["hi_preds"], 
                failure_threshold=HI_FAILURE_THRESHOLD,
                lifetime_prior=c["lifetime_prior"],
                q_scale=params["q_scale"],
                r_scale=params["r_scale"],
                fdp_k_sigma=params["fdp_k_sigma"],
                cqrl_quantile=params["cqrl_quantile"]
            )
            score = _arul(s2_rul, c["true_rul"])
            fold_scores.append(score)
            
        mean_score = float(np.mean(fold_scores))
        
        if mean_score > best_score:
            best_score = mean_score
            best_params = params
            best_fold_scores = fold_scores
            print(f"[{i+1}/{len(combinations)}] New Best! Mean Score: {best_score:.4f} | Params: {params}")

    print("\n" + "="*50)
    print("Optimization Complete!")
    print(f"Best Mean A_RUL Score: {best_score:.4f}")
    print(f"Best Parameters: {best_params}")
    print(f"Fold Scores: {best_fold_scores}")
    print("="*50)

if __name__ == "__main__":
    cached = cache_predictions()
    grid_search(cached)
