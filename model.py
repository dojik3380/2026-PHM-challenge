"""Dual-head RUL + HI regression model.

Shared encoder:
  - Vibration: STFT (4 x 1026) -> 1D CNN (SE) -> Projection -> BiLSTM -> Temporal Attention
  - Handcrafted: 4 x 40 PHM features (after HI augmentation) -> Linear -> GRU
  - Fusion: concat -> shared MLP -> 96-dim representation

Two heads on top:
  - RUL head: linear, unbounded log-space output. Production target -- trained
              with Huber(log) + Asymmetric(real). Matches the competition metric
              which rewards conservative under-prediction.
  - HI  head: sigmoid in [0, 1]. Auxiliary -- trained with MSE on the hybrid HI
              label. Forces the shared encoder to learn degradation trajectory
              features instead of falling into safe-low collapse.

Inference uses the RUL head only. HI head exists solely as a supervision aid.
"""

import math

import numpy as np
import torch
import torch.nn as nn

from config import (
    DROPOUT,
    HANDCRAFTED_DIM,
    HI_FEATURES_ENABLED,
    HI_LOSS_WEIGHT,
    OVER_EST_PENALTY_SCALE,
    RANKING_LOSS_WEIGHT,
    RUL_LOSS_WEIGHT,
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


class AsymmetricRULLoss(nn.Module):
    """A_RUL metric in REAL (seconds) space — used for EVALUATION ONLY, not training.

    Over-prediction (pred > target) is penalized 2.5× harder than under-prediction,
    matching the competition scoring function.
    """

    def __init__(self, over_scale: float = OVER_EST_PENALTY_SCALE,
                 under_scale: float = UNDER_EST_PENALTY_SCALE):
        super().__init__()
        self.over_scale = over_scale
        self.under_scale = under_scale

    def forward(self, predictions_real: torch.Tensor, targets_real: torch.Tensor) -> torch.Tensor:
        targets_real = targets_real.view_as(predictions_real)
        denominator = torch.clamp(targets_real, min=1000.0)
        er = torch.clamp(
            100.0 * (targets_real - predictions_real) / denominator,
            min=-500.0, max=500.0,
        )
        ln_two = 0.69314718
        loss = torch.where(
            er <= 0,
            ln_two * (-er) / self.over_scale,
            ln_two * er / self.under_scale,
        )
        return loss.mean()


class PairwiseRankingLoss(nn.Module):
    """Monotonicity regularizer: pred_rul[i] > pred_rul[j] when true_rul[i] > true_rul[j].

    Only enforces ordering for pairs whose true RUL differs by at least min_gap
    in log space (~22% RUL difference at default 0.2), to avoid noise from
    near-identical timesteps or different-case comparisons.
    """

    def __init__(self, margin: float = 0.05, min_gap: float = 0.2):
        super().__init__()
        self.margin = margin
        self.min_gap = min_gap

    def forward(self, pred_log: torch.Tensor, target_log: torch.Tensor) -> torch.Tensor:
        pred   = pred_log.view(-1)
        target = target_log.view(-1)
        n = pred.size(0)
        if n < 2:
            return pred.sum() * 0.0

        # All (i, j) pairs where target[i] > target[j] + min_gap
        target_diff = target.unsqueeze(1) - target.unsqueeze(0)   # (n, n)
        pred_diff   = pred.unsqueeze(1)   - pred.unsqueeze(0)     # (n, n)
        mask = target_diff > self.min_gap
        if not mask.any():
            return pred.sum() * 0.0

        # Hinge: penalize when pred[i] is not sufficiently larger than pred[j]
        loss = torch.clamp(self.margin - pred_diff[mask], min=0.0)
        return loss.mean()


class TrainingLoss(nn.Module):
    """L = 0.5*MSE(RUL_log) + 0.3*MSE(HI) + 0.2*PairwiseRanking.

    Returns (total, l_rul, l_hi, l_rank) for per-component logging.
    No asymmetric bias — predictions need no post-hoc DENORM correction.
    """

    def __init__(
        self,
        rul_weight:  float = RUL_LOSS_WEIGHT,
        hi_weight:   float = HI_LOSS_WEIGHT,
        rank_weight: float = RANKING_LOSS_WEIGHT,
    ):
        super().__init__()
        self.rul_mse  = nn.MSELoss()
        self.hi_mse   = nn.MSELoss()
        self.ranking  = PairwiseRankingLoss()
        self.rul_weight  = rul_weight
        self.hi_weight   = hi_weight
        self.rank_weight = rank_weight

    def forward(
        self,
        pred_rul_log:    torch.Tensor,
        pred_hi:         torch.Tensor,
        target_rul_log:  torch.Tensor,
        target_hi:       torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        l_rul  = self.rul_mse(pred_rul_log, target_rul_log)
        l_hi   = self.hi_mse(pred_hi, target_hi)
        l_rank = self.ranking(pred_rul_log, target_rul_log)
        total  = self.rul_weight * l_rul + self.hi_weight * l_hi + self.rank_weight * l_rank
        return total, l_rul, l_hi, l_rank


def asymmetric_rul_score_np(predictions, targets) -> np.ndarray:
    """Competition A_RUL score, NumPy version for evaluation."""
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    denominator = np.maximum(np.abs(targets), 1e-6)
    er = 100.0 * (targets - predictions) / denominator
    ln_half = np.log(0.5)
    exponent = np.where(
        er <= 0,
        -ln_half * er / OVER_EST_PENALTY_SCALE,
        ln_half * er / UNDER_EST_PENALTY_SCALE,
    )
    return np.exp(exponent)


class DualHeadModel(nn.Module):
    """Two-branch encoder + RUL head + HI head."""

    def __init__(
        self,
        vibration_channels: int = 4,
        vibration_features: int = VIBRATION_FEATURES_PER_CHANNEL,
        handcrafted_dim: int = HANDCRAFTED_DIM * (3 if HI_FEATURES_ENABLED else 1),
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

        fusion_dim = vib_hidden * 2 + feat_hidden
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 96),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # Two heads on top of the 96-dim shared representation.
        self.rul_head = nn.Linear(96, 1)            # log-space RUL
        self.hi_head = nn.Sequential(               # sigmoid HI in [0, 1]
            nn.Linear(96, 1),
            nn.Sigmoid(),
        )

    def encode(self, x_vib: torch.Tensor, x_feat: torch.Tensor) -> torch.Tensor:
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

        return self.fusion(torch.cat([h_vib, h_feat], dim=1))

    def forward(self, x_vib: torch.Tensor, x_feat: torch.Tensor):
        """Return (rul_log_pred, hi_pred). rul_log is unbounded; hi is in [0, 1]."""
        h = self.encode(x_vib, x_feat)
        return self.rul_head(h), self.hi_head(h)


def create_model(
    vibration_channels: int = 4,
    vibration_features: int = VIBRATION_FEATURES_PER_CHANNEL,
    handcrafted_dim: int = HANDCRAFTED_DIM * (3 if HI_FEATURES_ENABLED else 1),
) -> DualHeadModel:
    return DualHeadModel(
        vibration_channels=vibration_channels,
        vibration_features=vibration_features,
        handcrafted_dim=handcrafted_dim,
    )
