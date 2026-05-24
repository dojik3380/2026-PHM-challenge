"""Ensemble inference on data/Test using HI fold checkpoints.

For each Test case:
  1. Build all sliding windows (the full HI trajectory).
  2. Average HI predictions across fold checkpoints.
  3. Curve-fit the averaged HI sequence and extrapolate to threshold.
  4. RUL = t_failure - t_last_window.

Writes File / RUL_Score Excel.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from config import DEVICE, RESULTS_DIR, TEAM_NAME, TEST_DIR, WINDOW_SIZE
from data_loader import load_inference_dataset
from inference import hi_sequence_to_rul
from model import create_model
from train import MODEL_PATH


def _standardize(arr: np.ndarray, mean, std) -> np.ndarray:
    m = mean.cpu().numpy() if isinstance(mean, torch.Tensor) else np.asarray(mean)
    s = std.cpu().numpy() if isinstance(std, torch.Tensor) else np.asarray(std)
    s = np.where(s < 1e-8, 1.0, s)
    return ((arr - m) / s).astype(np.float32)


def _discover_fold_checkpoints(model_path: Path) -> list[Path]:
    model_path = Path(model_path)
    pattern_seed = f"{model_path.stem}_seed*_fold*.pt"
    pattern_plain = f"{model_path.stem}_fold*.pt"
    folds = sorted(set(model_path.parent.glob(pattern_seed)) | set(model_path.parent.glob(pattern_plain)))
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
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def _predict_hi(model: torch.nn.Module, X_vib: np.ndarray, X_feat: np.ndarray,
                device: torch.device, batch_size: int = 16) -> np.ndarray:
    preds: list[np.ndarray] = []
    n = len(X_vib)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            vb = torch.from_numpy(np.ascontiguousarray(X_vib[start:end])).to(device)
            fb = torch.from_numpy(np.ascontiguousarray(X_feat[start:end])).to(device)
            out = model(vb, fb).squeeze(1).cpu().numpy()
            preds.append(out)
    return np.concatenate(preds, axis=0)


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
            f"Checkpoint {fold_paths[0].name} has no 'degradation_baseline'. "
            f"Retrain with the current pipeline."
        )

    print(f"\nBuilding inference dataset from {test_dir} (window_size={window_size}) ...")
    X_vib_raw, X_feat_raw, metadata = load_inference_dataset(
        test_dir, window_size=window_size, stride=1,
        use_cache=use_cache, degradation_baseline=baseline,
    )
    print(f"Inference dataset: vib={X_vib_raw.shape} feat={X_feat_raw.shape}")
    print(metadata.groupby("case_name").size().to_string())

    device = torch.device(DEVICE)
    fold_hi: list[np.ndarray] = []
    for fp in fold_paths:
        model, ckpt = _load_fold(fp, device)
        X_vib = _standardize(X_vib_raw, ckpt["vibration_mean"], ckpt["vibration_std"])
        X_feat = _standardize(X_feat_raw, ckpt["feature_mean"], ckpt["feature_std"])
        hi = _predict_hi(model, X_vib, X_feat, device)
        print(f"  {fp.name}: HI mean={hi.mean():.3f} range=[{hi.min():.3f}, {hi.max():.3f}]")
        fold_hi.append(hi)
    hi_ensemble = np.mean(np.stack(fold_hi, axis=0), axis=0)

    # Stage 2: per-case curve fit -> RUL at the LAST window.
    results = []
    for case_name in sorted(metadata["case_name"].unique()):
        mask = (metadata["case_name"] == case_name).to_numpy()
        case_meta = metadata[mask].sort_values("end_timestep").reset_index(drop=True)
        case_times = case_meta["time_sec"].to_numpy(dtype=np.float64)
        case_hi = hi_ensemble[mask][case_meta.index.to_numpy()]
        # case_meta is already sorted; align hi by re-applying argsort if needed
        order = np.argsort(case_meta["end_timestep"].to_numpy())
        case_times = case_times[order]
        case_hi = case_hi[order]
        t_now = float(case_times[-1])
        t_fail = hi_sequence_to_rul(case_times, case_hi, t_now=t_now, case_max=None)
        rul = max(0.0, t_fail - t_now)
        results.append({
            "File": case_name,
            "RUL_Score": int(round(rul)),
            "hi_last": float(case_hi[-1]),
            "hi_mean": float(case_hi.mean()),
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
    p = argparse.ArgumentParser(description="HI ensemble inference on TDMS Test set")
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
