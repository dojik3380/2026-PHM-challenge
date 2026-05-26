# README_PHYSICAL_CONTEXT.md


# PHM Bearing RUL Challenge — Physical & Dataset Context

This document summarizes the PHM competition dataset, bearing physics, operating conditions, and critical assumptions for future agents/models.

This file intentionally excludes:
- current model architecture details
- temporary experiments
- hyperparameter history

It focuses ONLY on:
- physical system
- degradation context
- dataset structure
- domain constraints
- important engineering assumptions

---

# 1. Problem Definition

Task:
Predict Remaining Useful Life (RUL) of rolling-element bearings from vibration data.

Dataset type:
Run-to-failure degradation experiments.

Evaluation:
Asymmetric A_RUL metric.

Important:
The final evaluation/test environment does NOT provide:
- operation CSV
- true RPM
- torque
- temperature

Inference must therefore rely primarily on vibration.

---

# 2. Bearing Information

Bearing type:
30306 tapered roller bearing

Characteristic fault frequencies at 1000 RPM reference:

| Fault Type | Frequency @1000RPM |
|---|---|
| BPFI (Inner race) | 140 Hz |
| BPFO (Outer race) | 93 Hz |
| BSF (Ball spin) | 78 Hz |
| FTF / Cage | 6.7 Hz |

IMPORTANT:
These frequencies scale linearly with shaft RPM.

Actual frequency:
fault_freq_actual = fault_freq_ref * (rpm / 1000)

Example:
At 700 RPM:
- BPFO ≈ 65 Hz
- BPFI ≈ 98 Hz

At 950 RPM:
- BPFO ≈ 88 Hz
- BPFI ≈ 133 Hz

Therefore:
RPM-aware spectral interpretation is physically important.

---

# 3. Operating Conditions

Machine speed:
Approximately 700–950 RPM.

Important behavior:
RPM changes approximately every 1 hour during operation.

Thus:
- spectral peaks shift over lifetime
- fault harmonics drift
- fixed-frequency features are physically incorrect

---

# 4. Failure / Stop Conditions

Experiment stops when ONE of the following occurs first:

| Condition | Threshold |
|---|---|
| Bearing housing temperature | >= 200 °C |
| Rotational torque | <= -20 Nm |

Important implication:
Failure label is NOT purely vibration-defined.

RUL endpoint may correspond to:
- thermal runaway
- severe friction increase
- lubrication collapse
- mechanical seizure
- torque instability

Therefore:
temperature-related degradation signatures may exist indirectly inside vibration.

---

# 5. Dataset Overview

Total cases:
7 run-to-failure trajectories

## TDMS source

| Case | Lifetime |
|---|---|
| Train1 | 75,251 s |
| Train2 | 67,979 s |
| Train3 | 53,225 s |
| Train4 | 82,613 s |

## data2 source

| Case | Lifetime |
|---|---|
| Train_No_1 | 60,360 s |
| Train_No_2 | 96,720 s |
| Train_No_3 | 68,400 s |

Overall:
- minimum lifetime ≈ 53k s
- maximum lifetime ≈ 97k s
- spread ≈ 1.8×

No extreme lifetime outlier exists.

---

# 6. Signal Information

Sampling rate:
25.6 kHz

Channels:
4 vibration channels

TDMS:
CH1–CH4

data2:
CH03–CH06

Timestep definition:
10 seconds

Samples per timestep:
256,000 samples per channel

Approximate timesteps per case:
500–820

---

# 7. Important Physical Insights

## 7.1 RPM matters physically

Even though RPM branch sometimes hurt ML generalization,
RPM still affects:
- fault frequencies
- harmonic locations
- spectral energy distribution
- envelope peaks

Therefore:
RPM itself is NOT useless.

The issue is:
explicit RPM inputs may become shortcut/case-ID signals.

Better approach:
infer RPM implicitly from vibration spectrum.

---

# 7.2 Bearing degradation is usually late-stage visible

Typical rolling-bearing behavior:
- RMS
- kurtosis
- crest factor
- fault-band energy

often remain relatively stable during early life,
then rise sharply near failure.

Therefore:
early-life windows contain weak observable degradation information.

This is a core PHM difficulty.

---

# 7.3 Failure may correlate with thermal signatures

Because experiment stop conditions include:
- housing temperature >= 200 °C
- torque <= -20 Nm

possible indirect vibration indicators:
- broadband high-frequency energy increase
- friction-induced noise
- sideband broadening
- resonance excitation
- spectral whitening
- impulsiveness growth

Potentially useful:
high-frequency RMS / broadband energy tracking.

---

# 8. RPM Estimation Notes

RPM can be estimated from vibration using:
- low-frequency spectral peaks
- harmonic scoring
- fundamental + harmonic consistency

Current RPM search range:
600–1100 RPM

Current estimator concept:
score(f) =
A(f)
+ 0.5*A(2f)
+ 0.25*A(3f)

where:
f = candidate rotational frequency.

Observed:
RPM estimation from vibration is reasonably accurate on Train1.

Important:
This may still introduce shortcut-learning risk if directly injected as a model branch.

---

# 9. Modeling Implications

Key challenge:
short local vibration windows contain limited information about absolute lifecycle position.

Thus:
- direct RUL regression is difficult
- trajectory learning is more important than static classification
- ranking / progression learning matters heavily

Potentially important:
- monotonic degradation representations
- cumulative damage indicators
- temporal ranking supervision
- long-context aggregation
- probabilistic conservative RUL estimation

---

# 10. Known Constraints

Dataset size:
ONLY 7 run-to-failure cases.

Therefore:
- overfitting risk is extreme
- giant architectures are dangerous
- shortcut learning occurs easily
- robustness matters more than SOTA complexity

Simple, physically grounded methods are preferred.

---

# 11. Practical Recommendations for Future Agents

DO:
- respect RPM-scaled fault frequencies
- use physically meaningful spectral features
- prioritize trajectory consistency
- validate monotonicity
- analyze per-case prediction curves
- think in terms of degradation dynamics

DO NOT:
- assume fixed fault frequencies
- rely heavily on explicit RPM branches
- trust direct end-to-end RUL regression blindly
- over-engineer huge transformer models for 7 cases
- ignore metric asymmetry

---

# 12. CRITICAL AGENT INSTRUCTIONS

> [!WARNING]
> **DO NOT DELETE OR CLEAR THE `data2_features` STFT CACHE WITHOUT EXPLICIT USER PERMISSION!**
> 
> The STFT and handcrafted feature extraction process (including dynamic FFT-based RPM estimation) is extremely computationally expensive. The cache perfectly handles dynamic 1X Shaft estimation. 
> Even if you modify the name of a feature extraction function, do NOT clear the cache unless the underlying mathematical logic fundamentally changes. **ALWAYS reuse the cache** by default or explicitly ask the USER before triggering a rebuild.
