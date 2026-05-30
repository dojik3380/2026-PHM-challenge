import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from config import DATA2_DIR, TRAIN_DIR, DEVICE, BATCH_SIZE, HI_FAILURE_THRESHOLD
from data_loader import load_dataset
from model import create_model, asymmetric_rul_score_np
from stage2 import compute_stage2_trajectory

def _tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.tensor(np.ascontiguousarray(arr), dtype=torch.float32)

def main():
    X_vib, X_feat, hi, rul, metadata, baseline = load_dataset(
        original_dir=TRAIN_DIR, data2_dir=DATA2_DIR,
        include_original=True, include_data2=True,
    )
    
    sources = metadata["source"].astype(str).to_numpy()
    case_names = metadata["case_name"].astype(str).to_numpy()
    tdms_cases = sorted(np.unique(case_names[sources == "original"]).tolist())
    device = torch.device(DEVICE)
    
    for factor in [1.0, 0.95, 0.90, 0.85, 0.80]:
        total_scores = []
        for fold, held_out in enumerate(tdms_cases, start=1):
            model_path = Path(f"models/RUL_seed42_fold{fold}.pt")
            if not model_path.exists(): continue
                
            val_mask = (sources == "original") & (case_names == held_out)
            val_idx = np.where(val_mask)[0]
            
            Xv_va = X_vib[val_idx]; Xf_va = X_feat[val_idx]
            case_max_va = np.maximum(metadata.iloc[val_idx]["case_max"].to_numpy(np.float32), 1.0)
            elapsed_va = metadata.iloc[val_idx]["time_sec"].to_numpy(np.float32) / case_max_va
            
            val_loader = DataLoader(
                TensorDataset(_tensor(Xv_va), _tensor(Xf_va), _tensor(elapsed_va).unsqueeze(1)),
                batch_size=BATCH_SIZE, shuffle=False
            )
            
            ckpt = torch.load(model_path, map_location=device, weights_only=False)
            model = create_model(Xv_va.shape[2], Xv_va.shape[3], Xf_va.shape[-1]).to(device)
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
            
            s2_rul = compute_stage2_trajectory(
                val_times_sec, hi_preds, 
                failure_threshold=HI_FAILURE_THRESHOLD,
                lifetime_prior=ckpt.get("lifetime_prior")
            )
            
            # APPLY SAFETY FACTOR
            s2_rul_adjusted = s2_rul * factor
            score = np.mean(asymmetric_rul_score_np(s2_rul_adjusted, true_rul))
            total_scores.append(score)
            
        print(f"Factor {factor:.2f} => Mean A_RUL Score: {np.mean(total_scores):.4f}")

if __name__ == '__main__':
    main()
