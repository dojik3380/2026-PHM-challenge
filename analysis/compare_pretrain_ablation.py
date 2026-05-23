"""사전학습 ON vs OFF 효과 비교 - log 파싱 + checkpoint 평가."""
from __future__ import annotations

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import re
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LOG_ON  = PROJECT_ROOT / "outputs" / "train_with_pretrain.log"
LOG_OFF = PROJECT_ROOT / "outputs" / "train_no_pretrain.log"


def parse_fold_results(log_path: Path) -> dict:
    """train 로그에서 fold별 (모든 epoch의 val_loss 리스트, best, early-stop 정보) 추출."""
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    folds = {}
    current_fold = None

    for line in text.splitlines():
        m_fold = re.search(r"Fold (\d+)/\d+", line)
        if m_fold and "====" in line:
            current_fold = int(m_fold.group(1))
            folds.setdefault(current_fold, {"epochs": [], "early_stop_epoch": None})
            continue

        m_epoch = re.search(r"Fold (\d+) - Epoch (\d+)/\d+ \| train_loss=([\d\.]+) \| val_loss=([\d\.]+) \| patience=(\d+)/(\d+)", line)
        if m_epoch:
            f, ep, tl, vl, p, _ = m_epoch.groups()
            f = int(f)
            folds.setdefault(f, {"epochs": [], "early_stop_epoch": None})
            folds[f]["epochs"].append({
                "epoch": int(ep),
                "train": float(tl),
                "val": float(vl),
                "patience": int(p),
            })

        m_stop = re.search(r"\[Early Stop\] Fold (\d+) .* at epoch (\d+)", line)
        if m_stop:
            f, ep = int(m_stop.group(1)), int(m_stop.group(2))
            folds.setdefault(f, {"epochs": [], "early_stop_epoch": None})
            folds[f]["early_stop_epoch"] = ep

    # best는 log에 직접 안 보이므로 print 주기(10 epoch마다)로 본 epochs 중 최소를 사용
    # 더 정확한 best는 model checkpoint를 봐야 하지만 로깅 주기 한계
    for f, info in folds.items():
        if info["epochs"]:
            best_epoch = min(info["epochs"], key=lambda e: e["val"])
            info["best_val"] = best_epoch["val"]
            info["best_epoch_seen"] = best_epoch["epoch"]
            info["last_val_seen"] = info["epochs"][-1]["val"]
        else:
            info["best_val"] = None
            info["best_epoch_seen"] = None
            info["last_val_seen"] = None

    return folds


def main() -> None:
    print("=" * 70)
    print("  Pretrain ON vs OFF - Validation Loss Comparison")
    print("=" * 70)

    on_folds  = parse_fold_results(LOG_ON)
    off_folds = parse_fold_results(LOG_OFF)

    header = f"{'Fold':<6}{'Pretrain ON':<28}{'Pretrain OFF':<28}{'D (ON−OFF)':<15}"
    print(header)
    print("-" * len(header))

    on_vals, off_vals = [], []
    for f in sorted(set(list(on_folds) + list(off_folds))):
        on_info  = on_folds.get(f, {})
        off_info = off_folds.get(f, {})
        on_v  = on_info.get("best_val")
        off_v = off_info.get("best_val")
        if on_v is not None: on_vals.append(on_v)
        if off_v is not None: off_vals.append(off_v)
        delta = (on_v - off_v) if (on_v is not None and off_v is not None) else None
        on_str  = f"{on_v:.4f} @ep{on_info.get('best_epoch_seen','-')}" if on_v else "-"
        off_str = f"{off_v:.4f} @ep{off_info.get('best_epoch_seen','-')}" if off_v else "-"
        delta_str = f"{delta:+.4f}" if delta is not None else "-"
        print(f"{f:<6}{on_str:<28}{off_str:<28}{delta_str:<15}")

    print("-" * len(header))
    if on_vals and off_vals:
        on_mean  = sum(on_vals) / len(on_vals)
        off_mean = sum(off_vals) / len(off_vals)
        improvement_pct = 100.0 * (off_mean - on_mean) / off_mean if off_mean else 0
        print(f"Mean best val_loss  : ON={on_mean:.4f}  OFF={off_mean:.4f}  D={on_mean-off_mean:+.4f}  ({improvement_pct:+.1f}% improvement)")

    # 추가: checkpoint에서 best_val_loss를 직접 확인 (저장돼 있다면)
    print("\n[Checkpoint inspection]")
    for tag, prefix in [("ON ", "RUL_WithPretrain"), ("OFF", "RUL_NoPretrain")]:
        for f in range(1, 5):
            ckpt_path = PROJECT_ROOT / "models" / f"{prefix}_fold{f}.pt"
            if ckpt_path.exists():
                try:
                    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                    keys = [k for k in ck.keys() if "val" in k.lower() or "loss" in k.lower()]
                    size_mb = ckpt_path.stat().st_size / 1024 / 1024
                    print(f"  {tag} fold{f}: {ckpt_path.name} ({size_mb:.1f}MB) loss-keys={keys}")
                except Exception as exc:
                    print(f"  {tag} fold{f}: load failed: {exc}")

    print("\nNote: 로그는 10 epoch 주기로 출력되어 best가 그 사이일 수 있음. checkpoint에는 best_state만 저장됨.")


if __name__ == "__main__":
    main()