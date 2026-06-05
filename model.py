"""Stage-1 HI prediction model (2-stage PHM pipeline).

Shared encoder:
  - Vibration : STFT (4 × 1026) → 1D CNN (SE) → Projection → BiLSTM → Temporal Attention
  - Handcrafted: 4 × (F*3+1) augmented features → Linear → GRU
                 [rel, cummax_rel, cumdamage_rel, energy_cumdamage] per channel
  - Fusion     : concat → shared MLP → 96-dim representation

Single head:
  - HI head: sigmoid [0, 1]. THIS IS the only output.
             Trained with Huber(δ=0.3, late×5) + HIPairwiseRankingLoss.

Inference sequence (§4):
  1. HI head → per-window HI trajectory.
  2. Stage-2 fits f(t) = a·e^(bt)+c to the trajectory.
  3. T_failure = (1/b)·ln((1-c)/a); RUL = T_failure - t_current.
"""

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from config import (
    AUGMENTED_HANDCRAFTED_DIM,
    DROPOUT,
    HI_LOSS_WEIGHT,
    HI_RANK_LOSS_WEIGHT,
    HUBER_DELTA,
    OVER_EST_PENALTY_SCALE,
    STAGE_CE_WEIGHT,
    STAGE_NUM_CLASSES,
    UNDER_EST_PENALTY_SCALE,
    VIB_HIDDEN,
    VIBRATION_FEATURES_PER_CHANNEL,
)


class SEBlock1D(nn.Module):
    """Squeeze-and-Excitation block: per-frequency attention to suppress noise."""

    def __init__(self, channel: int, reduction: int = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, max(1, channel // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, channel // reduction), channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal attention."""

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class HIPairwiseRankingLoss(nn.Module):
    """HI 단조성 강제: 같은 케이스 내 later window → pred_hi 더 큼.

    true_hi[j] > true_hi[i] + min_gap인 쌍에서
    pred_hi[j] > pred_hi[i] - margin 패널티.
    """

    def __init__(self, margin: float = 0.02, min_gap: float = 0.05):
        super().__init__()
        self.margin = margin
        self.min_gap = min_gap

    def forward(
        self,
        pred_hi:   torch.Tensor,
        target_hi: torch.Tensor,
        case_ids:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pred   = pred_hi.view(-1)
        target = target_hi.view(-1)
        n = pred.size(0)
        if n < 2:
            return pred.sum() * 0.0

        diff_t = target.unsqueeze(0) - target.unsqueeze(1)  # [j]-[i]
        diff_p = pred.unsqueeze(0)   - pred.unsqueeze(1)
        mask = diff_t > self.min_gap

        if case_ids is not None:
            same_case = case_ids.view(1, -1) == case_ids.view(-1, 1)
            mask = mask & same_case

        if not mask.any():
            return pred.sum() * 0.0

        return torch.clamp(self.margin - diff_p[mask], min=0.0).mean()


class TrainingLoss(nn.Module):
    """L = hi*Huber + rank*Rank + λ1*Mono + λ2*Smooth + stage*CE(Stage)"""

    def __init__(
        self,
        hi_weight:      float = 1.0,
        lambda1:        float = 0.5,
    ):
        super().__init__()
        self.hi_loss      = nn.MSELoss(reduction="none")
        self.hi_weight      = hi_weight
        self.lambda1        = lambda1

    def forward(
        self,
        pred_hi:      torch.Tensor,
        target_hi:    torch.Tensor,
        case_ids:     Optional[torch.Tensor] = None,
        elapsed:      Optional[torch.Tensor] = None,
        late_weights: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p_hi = pred_hi.view(-1)
        t_hi = target_hi.view(-1)

        # Exponential Late-stage weighting: target_hi가 클수록 가중치 급증
        late_stage_w = torch.exp(3.0 * t_hi)
        
        if late_weights is not None:
            w = late_weights.view(-1) * late_stage_w
            w = w / (w.mean() + 1e-9)
            l_hi = (self.hi_loss(p_hi, t_hi) * w).mean()
        else:
            w = late_stage_w
            w = w / (w.mean() + 1e-9)
            l_hi = (self.hi_loss(p_hi, t_hi) * w).mean()

        l_mono = torch.tensor(0.0, device=pred_hi.device)

        if case_ids is not None and elapsed is not None:
            c_ids = case_ids.view(-1)
            elap = elapsed.view(-1)

            sort_keys = c_ids.float() * 1000.0 + elap
            sort_idx = torch.argsort(sort_keys)

            p_sorted = p_hi[sort_idx]
            c_sorted = c_ids[sort_idx]

            same_case_mask = (c_sorted[:-1] == c_sorted[1:])

            if same_case_mask.any():
                p_prev = p_sorted[:-1][same_case_mask]
                p_next = p_sorted[1:][same_case_mask]

                l_mono = torch.relu(p_prev - p_next).mean()

        total = (
            self.hi_weight * l_hi
            + self.lambda1 * l_mono
        )

        return total, l_hi, l_mono


def asymmetric_rul_score_np(predictions, targets) -> np.ndarray:
    """Competition A_RUL score, NumPy version for evaluation."""
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    denominator = np.maximum(np.abs(targets), 1000.0)
    er = 100.0 * (targets - predictions) / denominator
    ln_half = np.log(0.5)
    exponent = np.where(
        er <= 0,
        -ln_half * er / OVER_EST_PENALTY_SCALE,
        ln_half * er / UNDER_EST_PENALTY_SCALE,
    )
    return np.exp(exponent)


class HIModel(nn.Module):
    """Single-head HI regression model."""

    def __init__(
        self,
        vibration_channels: int = 4,
        vibration_features: int = VIBRATION_FEATURES_PER_CHANNEL,
        handcrafted_dim: int = AUGMENTED_HANDCRAFTED_DIM,
        vib_hidden: int = VIB_HIDDEN,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.handcrafted_dim = handcrafted_dim

        self.vibration_cnn = nn.Sequential(
            nn.Conv1d(vibration_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            SEBlock1D(32),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            SEBlock1D(64),
            nn.Flatten(),
        )
        dummy = torch.zeros(1, vibration_channels, vibration_features)
        with torch.no_grad():
            cnn_out = self.vibration_cnn(dummy).shape[1]

        self.vibration_projection = nn.Sequential(
            nn.Linear(cnn_out, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.vibration_lstm = nn.LSTM(
            input_size=128, hidden_size=vib_hidden,
            num_layers=2, batch_first=True, bidirectional=True,
        )
        self.pos_encoder = PositionalEncoding(d_model=vib_hidden * 2)
        self.temporal_attention = nn.Sequential(
            nn.Linear(vib_hidden * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        feature_embed = 64
        feat_hidden = 48
        self.feature_projection = nn.Sequential(
            nn.Linear(vibration_channels * handcrafted_dim, feature_embed),
            nn.LayerNorm(feature_embed),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.feature_encoder = nn.GRU(
            input_size=feature_embed, hidden_size=feat_hidden,
            num_layers=1, batch_first=True,
        )

        elapsed_embed_dim = 8
        self.elapsed_embed = nn.Sequential(
            nn.Linear(1, elapsed_embed_dim),
            nn.ReLU(),
        )

        fusion_dim = vib_hidden * 2 + feat_hidden + elapsed_embed_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 96),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.hi_head = nn.Sequential(
            nn.Linear(96, 1),
            nn.Sigmoid(),
        )
        # Stage classification auxiliary head: lifetime quantile 4-class
        # (healthy / incipient / fault / severe) → case fingerprint representation
        # 학계 표준 multi-task: contrastive 대안. 절대 lifetime 위치 학습.
        # First Principles Diet: Removed stage_head and lifetime_head

    def forward(self, x_vib: torch.Tensor, x_feat: torch.Tensor,
                elapsed_frac: Optional[torch.Tensor] = None):
        """Return hi_pred ∈ [0, 1].

        elapsed_frac = time_sec / case_max ∈ [0, 1].
        """
        b, seq, ch, fbin = x_vib.shape

        v = x_vib.reshape(b * seq, ch, fbin)
        v = self.vibration_cnn(v)
        v = self.vibration_projection(v)
        v = v.reshape(b, seq, 128)
        v_out, _ = self.vibration_lstm(v)
        v_out_pe = self.pos_encoder(v_out)
        attn = torch.softmax(self.temporal_attention(v_out_pe), dim=1)
        h_vib = torch.sum(v_out * attn, dim=1)

        f = x_feat.reshape(b, seq, -1)
        f = self.feature_projection(f)
        _, h_feat = self.feature_encoder(f)
        h_feat = h_feat.squeeze(0)

        if elapsed_frac is None:
            elapsed_frac = torch.zeros(b, 1, device=x_vib.device)
        h_elapsed = self.elapsed_embed(elapsed_frac.view(b, 1))

        h = self.fusion(torch.cat([h_vib, h_feat, h_elapsed], dim=1))
        hi = self.hi_head(h)
        return hi


# ── backward-compat alias (evaluate.py / train.py에서 create_model() 사용) ──
def create_model(
    vibration_channels: int = 4,
    vibration_features: int = VIBRATION_FEATURES_PER_CHANNEL,
    handcrafted_dim: int = AUGMENTED_HANDCRAFTED_DIM,
) -> HIModel:
    return HIModel(
        vibration_channels=vibration_channels,
        vibration_features=vibration_features,
        handcrafted_dim=handcrafted_dim,
    )
