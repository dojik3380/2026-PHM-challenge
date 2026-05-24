"""Per-case HI trajectory plot."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_hi_trajectory(
    hi_pred: np.ndarray,
    hi_true: np.ndarray,
    val_meta: pd.DataFrame,
    output_path: Union[str, Path],
    title: str = "HI trajectory",
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = val_meta.reset_index(drop=True).copy()
    df["pred"] = np.asarray(hi_pred)
    df["true"] = np.asarray(hi_true)
    if "time_sec" not in df.columns:
        df["time_sec"] = np.arange(len(df), dtype=np.float32)

    cases = sorted(df["case_name"].astype(str).unique())
    n = len(cases)
    cols = min(2, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 3.5 * rows), squeeze=False)

    for i, case in enumerate(cases):
        ax = axes[i // cols][i % cols]
        sub = df[df["case_name"] == case].sort_values("time_sec")
        ax.plot(sub["time_sec"] / 3600.0, sub["true"], "k-", label="true HI", linewidth=1.5)
        ax.plot(sub["time_sec"] / 3600.0, sub["pred"], "r--", label="pred HI", linewidth=1.2, alpha=0.85)
        ax.set_title(f"{case} (n={len(sub)})")
        ax.set_xlabel("time (h)")
        ax.set_ylabel("HI")
        ax.set_ylim(-0.05, 1.10)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)

    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=110)
    plt.close(fig)
