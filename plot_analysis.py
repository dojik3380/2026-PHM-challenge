import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from config import DATA2_DIR, TRAIN_DIR, DEVICE, BATCH_SIZE, HI_FAILURE_THRESHOLD
from data_loader import load_dataset
from model import create_model
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
    
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for fold, held_out in enumerate(tdms_cases, start=1):
        ax = axes[fold-1]
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
        
        val_times_h = val_times_sec / 3600.0
        true_rul_h = true_rul / 3600.0
        s2_rul_h = s2_rul / 3600.0
        
        ax.plot(val_times_h, true_rul_h, 'w--', label='True RUL', linewidth=2)
        ax.plot(val_times_h, s2_rul_h, 'c-', label='Predicted RUL', linewidth=2)
        ax.set_title(f"Fold {fold} ({held_out}) RUL Trajectory", fontsize=14, fontweight='bold')
        ax.set_xlabel("Elapsed Time (Hours)", fontsize=12)
        ax.set_ylabel("Remaining Useful Life (Hours)", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig("C:/Users/kdj33/.gemini/antigravity-ide/brain/c3cd5fe6-7195-4b52-957c-e347559fa326/rul_analysis_plot.png", dpi=150)
    print("Plot saved.")

if __name__ == '__main__':
    main()
