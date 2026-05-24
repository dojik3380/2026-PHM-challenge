# PHM RUL Training

This repository now uses one training path: original `data/Train` and `data2/Train_No_*` are always trained together with the multi-branch model.

## Train

```powershell
& "C:\Users\kdj33\miniconda3\envs\phm\python.exe" main.py train --epochs 80
```

For a quick cache/smoke run, cap samples per case:

```powershell
& "C:\Users\kdj33\miniconda3\envs\phm\python.exe" main.py train --max-samples 96 --epochs 2
```

The first run builds STFT/feature caches in `data2_features/`; later runs reuse them.
