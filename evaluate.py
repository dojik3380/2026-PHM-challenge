"""Ensemble inference on data/Test using Single-Head fold checkpoints.

For each Test case:
  1. Build all sliding windows.
  2. Run each fold's model; collect HI predictions.
  3. Ensemble (mean) the HI predictions across folds.
  4. Pass ensembled HI trajectory to Stage-2 curve fitting to get final RUL.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from config import CALIBRATION_SHRINK, DEVICE, HI_FAILURE_THRESHOLD, RESULTS_DIR, TEAM_NAME, TEST_DIR, WINDOW_SIZE
from stage2 import fit_stage2_rul
from data_loader import load_inference_dataset
from model import create_model
from train import MODEL_PATH





def _discover_fold_checkpoints(model_path: Path) -> list[Path]:
    model_path = Path(model_path)
    pattern_seed = f"{model_path.stem}_seed*_fold*.pt"
    pattern_plain = f"{model_path.stem}_fold*.pt"
    import re
    folds = sorted(set(model_path.parent.glob(pattern_seed)) | set(model_path.parent.glob(pattern_plain)))
    folds = [f for f in folds if re.search(r"_fold\d+\.pt$", f.name)]
    if folds:
        return folds
    if model_path.exists():
        return [model_path]
    raise FileNotFoundError(f"No fold checkpoints near {model_path}")


def _load_fold(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = create_model(
        vibration_channels=ckpt["vibration_channels"],
        vibration_features=ckpt["vibration_features"],
        handcrafted_dim=ckpt["handcrafted_dim"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    return model, ckpt


def _predict(model: torch.nn.Module, X_vib: np.ndarray, X_feat: np.ndarray,
             device: torch.device, batch_size: int = 16,
             elapsed_frac: Optional[np.ndarray] = None) -> np.ndarray:
    hi = []
    n = len(X_vib)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            vb = torch.from_numpy(np.ascontiguousarray(X_vib[start:end])).to(device)
            fb = torch.from_numpy(np.ascontiguousarray(X_feat[start:end])).to(device)
            ef = None
            if elapsed_frac is not None:
                ef = torch.tensor(
                    elapsed_frac[start:end].reshape(-1, 1), dtype=torch.float32
                ).to(device)
            ph = model(vb, fb, ef)
            hi.append(np.array(ph.squeeze(1).cpu().tolist(), dtype=np.float32))
    return np.concatenate(hi, axis=0)


def evaluate_test(
    test_dir: Path = TEST_DIR,
    model_path: Path = MODEL_PATH,
    output_path: Optional[Path] = None,
    window_size: int = WINDOW_SIZE,
    use_cache: bool = True,
) -> pd.DataFrame:
    test_dir = Path(test_dir)
    model_path = Path(model_path)
    if output_path is None:
        output_path = RESULTS_DIR / f"{TEAM_NAME}_validation.xlsx"
    output_path = Path(output_path)

    fold_paths = _discover_fold_checkpoints(model_path)
    print(f"Found {len(fold_paths)} fold checkpoint(s):")
    for fp in fold_paths:
        print(f"  - {fp.name}")

    first_ckpt = torch.load(fold_paths[0], map_location="cpu", weights_only=False)
    baseline = first_ckpt.get("degradation_baseline")
    if baseline is None:
        raise RuntimeError(
            f"Checkpoint {fold_paths[0].name} has no 'degradation_baseline'. Retrain."
        )

    print(f"\nBuilding inference dataset from {test_dir} (window_size={window_size}) ...")
    X_vib_raw, X_feat_raw, metadata = load_inference_dataset(
        test_dir, window_size=window_size, stride=1,
        use_cache=use_cache, degradation_baseline=baseline,
    )
    print(f"Inference dataset: vib={X_vib_raw.shape} feat={X_feat_raw.shape}")
    print(metadata.groupby("case_name").size().to_string())

    elapsed_frac = (
        metadata["time_sec"].to_numpy(np.float32)
        / np.maximum(metadata["case_max"].to_numpy(np.float32), 1.0)
    )

    device = torch.device(DEVICE)
    X_vib = X_vib_raw
    X_feat = X_feat_raw

    fold_hi: list[np.ndarray] = []
    for fp in fold_paths:
        model, _ckpt = _load_fold(fp, device)
        hi = _predict(model, X_vib, X_feat, device, elapsed_frac=elapsed_frac)
        print(f"  {fp.name}: HI mean={hi.mean():.3f}")
        fold_hi.append(hi)

    hi_ens = np.mean(np.stack(fold_hi, axis=0), axis=0)
    print(f"\n[ensemble] HI mean={hi_ens.mean():.3f}")

    # Stage 2: 케이스별 HI 궤적 → 지수 피팅 → RUL 외삽
    results = []
    for case_name in sorted(metadata["case_name"].unique()):
        mask = (metadata["case_name"] == case_name).to_numpy()
        case_global_idx = np.where(mask)[0]

        # time_sec 기준 정렬 (fit_stage2_rul 내부에서도 정렬하지만 명시적으로)
        sort_order      = np.argsort(metadata.iloc[case_global_idx]["time_sec"].to_numpy())
        sorted_idx      = case_global_idx[sort_order]
        case_times      = metadata.iloc[sorted_idx]["time_sec"].to_numpy(np.float64)
        case_hi         = hi_ens[sorted_idx]

        rul_s2 = fit_stage2_rul(case_times, case_hi, failure_threshold=HI_FAILURE_THRESHOLD)
        rul_s2 = float(max(rul_s2 * CALIBRATION_SHRINK, 0.0))

        last_global = sorted_idx[-1]
        results.append({
            "File": case_name,
            "RUL_Score": int(round(rul_s2)),
            "rul_seconds_raw": rul_s2,
            "hi_last": float(hi_ens[last_global]),
            "n_windows": int(mask.sum()),
        })

    df = pd.DataFrame(results)
    print("\nPredictions:")
    print(df.to_string(index=False))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out = df[["File", "RUL_Score"]]
    if output_path.suffix.lower() in (".xlsx", ".xls"):
        df_out.to_excel(output_path, index=False)
    else:
        df_out.to_csv(output_path, index=False)
    print(f"\nSaved {output_path}")
    return df


def main() -> None:
    p = argparse.ArgumentParser(description="Single-head ensemble inference on TDMS Test set")
    p.add_argument("--test-dir", type=Path, default=TEST_DIR)
    p.add_argument("--model-path", type=Path, default=MODEL_PATH)
    p.add_argument("--output", type=Path, default=None,
                   help=f"default: results/{TEAM_NAME}_validation.xlsx")
    p.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    p.add_argument("--no-cache", action="store_true")
    args = p.parse_args()
    evaluate_test(
        test_dir=args.test_dir, model_path=args.model_path,
        output_path=args.output, window_size=args.window_size,
        use_cache=not args.no_cache,
    )


if __name__ == "__main__":
    main()
