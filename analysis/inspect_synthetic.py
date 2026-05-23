"""합성 데이터 (pretrain_data.npz) 통계 점검."""
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
data = np.load(ROOT / "outputs" / "synthetic" / "pretrain_data.npz")

print("=" * 60)
print("Synthetic pretrain_data.npz 통계")
print("=" * 60)
print("Keys                 :", list(data.keys()))
print("X_vib shape          :", data["X_vib"].shape, "dtype:", data["X_vib"].dtype)
print("X_aux shape          :", data["X_aux"].shape)
print("y_rul shape          :", data["y_rul"].shape)
print(f"y_rul range          : {data['y_rul'].min():.1f} ~ {data['y_rul'].max():.1f} sec")
print(f"X_vib global mean    : {data['X_vib'].mean():.6f}")
print(f"X_vib global std     : {data['X_vib'].std():.6f}")
print(f"X_vib min/max        : {data['X_vib'].min():.6f} / {data['X_vib'].max():.6f}")
print()
print("X_aux per-feature [RPM, RMS]")
print(f"  mean: {data['X_aux'].mean(axis=(0,1))}")
print(f"  std : {data['X_aux'].std(axis=(0,1))}")
print(f"  min : {data['X_aux'].min(axis=(0,1))}")
print(f"  max : {data['X_aux'].max(axis=(0,1))}")
print()
print("Fault types:", np.unique(data["fault_types"], return_counts=True))
print(f"RPM range (per-window): {float(data['rpms'].min()):.1f} ~ {float(data['rpms'].max()):.1f}")

# vibration features 구조: (N_windows, window_size, 4_channels, 1026)
# 1026 = 513 freq_mean + 513 freq_std (concat)
xvib = data["X_vib"]
print()
print("Channel-wise stats (한 timestep 평균):")
ch_mean = xvib.mean(axis=(0, 1, 3))  # (4,)
ch_std = xvib.std(axis=(0, 1, 3))
print(f"  channel mean: {ch_mean}")
print(f"  channel std : {ch_std}")

# 평균 스펙트럼 영역 (앞 513) vs std 영역 (뒤 513) 분리
print()
print("Spectrum mean vs std halves (전체 평균):")
print(f"  mean-half avg: {xvib[..., :513].mean():.6f}")
print(f"  std-half avg : {xvib[..., 513:].mean():.6f}")