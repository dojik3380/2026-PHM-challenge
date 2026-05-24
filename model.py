"""Health-Indicator regression model.

Two-branch fusion:
  - Vibration branch: STFT (4 x 1026) -> 1D CNN (SE) -> Projection -> BiLSTM -> Temporal Attention
  - Handcrafted branch: 4 x 40 PHM features (after HI augmentation) -> Linear -> GRU
  - Fusion: concat -> MLP -> sigmoid -> HI in [0, 1]

The operation/RPM branch was removed: the RPM ablation showed RPM did not
contribute usable trajectory information.
The output head is a single sigmoid that predicts the Health Indicator. RUL
is recovered downstream in inference.py by curve-fitting the HI trajectory
per case.
"""

import math

import numpy as np
import torch
import torch.nn as nn

from config import (
    DROPOUT,
    HANDCRAFTED_DIM,
    HI_FEATURES_ENABLED,
    OVER_EST_PENALTY_SCALE,
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


def asymmetric_rul_score_np(predictions, targets) -> np.ndarray:
    """Competition A_RUL score, kept here for evaluation only."""
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


class HIModel(nn.Module):
    """Two-branch HI regression: vibration STFT + handcrafted PHM features."""

    def __init__(
        self,
        vibration_channels: int = 4,
        vibration_features: int = VIBRATION_FEATURES_PER_CHANNEL,
        handcrafted_dim: int = HANDCRAFTED_DIM * (4 if HI_FEATURES_ENABLED else 1),
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
            nn.Linear(96, 1),
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

        return torch.cat([h_vib, h_feat], dim=1)

    def forward(self, x_vib: torch.Tensor, x_feat: torch.Tensor) -> torch.Tensor:
        h = self.encode(x_vib, x_feat)
        return torch.sigmoid(self.fusion(h))


def create_model(
    vibration_channels: int = 4,
    vibration_features: int = VIBRATION_FEATURES_PER_CHANNEL,
    handcrafted_dim: int = HANDCRAFTED_DIM * (4 if HI_FEATURES_ENABLED else 1),
) -> HIModel:
    return HIModel(
        vibration_channels=vibration_channels,
        vibration_features=vibration_features,
        handcrafted_dim=handcrafted_dim,
    )
