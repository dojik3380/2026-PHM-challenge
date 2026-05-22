"""STFT + CNN + LSTM RUL 모델 (Physics-based, RPM-auxiliary).

Operation 브랜치를 완전 제거하고, vibration STFT 특징 + RPM/RMS auxiliary
temporal sequence만으로 RUL을 예측하는 구조.

아키텍처:
    Vibration Branch:
        STFT → 1D CNN (SE Block) → Projection → BiLSTM → Temporal Attention
    Auxiliary:
        RPM(t), RMS(t) 마지막 timestep 직접 사용
    Fusion:
        concat [h_vib_attended, h_aux] → MLP → RUL
"""

import math

import numpy as np
import torch
import torch.nn as nn

from config import (
    ASYMMETRIC_WEIGHT,
    AUXILIARY_DIM,
    DROPOUT,
    HUBER_WEIGHT,
    OVER_EST_PENALTY_SCALE,
    UNDER_EST_PENALTY_SCALE,
    VIBRATION_FEATURES_PER_CHANNEL,
)

class SEBlock1D(nn.Module):
    """
    Squeeze-and-Excitation Block for 1D CNN.
    특정 주파수 채널(Fault Harmonic)에 어텐션을 주어 노이즈를 억제합니다.
    """
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, max(1, channel // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, channel // reduction), channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class PositionalEncoding(nn.Module):
    """
    Temporal Attention을 위해 LSTM 출력에 위치 정보(Time-step)를 더해줍니다.
    """
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch_size, seq_len, d_model)
        """
        return x + self.pe[:, :x.size(1), :]



class STFTCNNLSTMRULModel(nn.Module):
    """
    Physics-based RUL 예측 모델.

    진동 branch: STFT 입력 -> CNN(SE Block) -> BiLSTM -> Temporal Attention
    Auxiliary:   RPM(t), RMS(t) 마지막 timestep 직접 사용
    Fusion head: 두 출력을 결합해 RUL 예측
    """

    def __init__(
        self,
        vibration_channels: int = 4,
        auxiliary_dim: int = AUXILIARY_DIM,
        vibration_features: int = VIBRATION_FEATURES_PER_CHANNEL,
        vib_hidden: int = 128,
        dropout: float = DROPOUT,
    ):
        super().__init__()

        self.auxiliary_dim = auxiliary_dim

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
        
        # Calculate flattened size dynamically
        dummy_input = torch.zeros(1, vibration_channels, vibration_features)
        with torch.no_grad():
            cnn_out_size = self.vibration_cnn(dummy_input).shape[1]
            
        self.vibration_projection = nn.Sequential(
            nn.Linear(cnn_out_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.vibration_lstm = nn.LSTM(
            input_size=128,
            hidden_size=vib_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )

        self.pos_encoder = PositionalEncoding(d_model=vib_hidden * 2)

        self.temporal_attention = nn.Sequential(
            nn.Linear(vib_hidden * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.last_attn_weights = None

        # Fusion head: vib_hidden*2 (vibration attention) + auxiliary_dim (RPM, RMS)
        fusion_input_dim = vib_hidden * 2 + auxiliary_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x_vibration: torch.Tensor, x_auxiliary: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_vibration: (batch, seq_len, channels, freq_bins) - STFT magnitude
            x_auxiliary: (batch, seq_len, auxiliary_dim)        - [RPM(t), RMS(t)]

        Returns:
            RUL prediction: (batch, 1)
        """
        # x_vibration: (batch, seq_len, 4, freq_bins)
        batch_size, seq_len, channels, freq_bins = x_vibration.shape

        # CNN은 timestep별 STFT 스펙트럼에서 주파수 패턴을 추출한다.
        vib = x_vibration.reshape(batch_size * seq_len, channels, freq_bins)
        vib = self.vibration_cnn(vib)
        vib = self.vibration_projection(vib)

        # 다시 시퀀스로 복원한 뒤 LSTM으로 degradation 흐름을 학습한다.
        vib = vib.reshape(batch_size, seq_len, 128)
        vib_out, _ = self.vibration_lstm(vib)
        
        # Positional Encoding 적용
        vib_out_pe = self.pos_encoder(vib_out)
        
        # Temporal Attention Pooling (Positional Encoding이 적용된 출력을 기반으로 Attention)
        attn_scores = self.temporal_attention(vib_out_pe)
        attn_weights = torch.softmax(attn_scores, dim=1)
        
        # GPU memory leak 방지를 위해 detach.cpu()로 저장하여 시각화 모듈에서 꺼내 쓸 수 있도록 함
        self.last_attn_weights = attn_weights.detach().cpu()
        
        h_vib_attended = torch.sum(vib_out * attn_weights, dim=1)  # (batch, vib_hidden*2)

        # Auxiliary: 마지막 timestep의 RPM/RMS를 직접 사용
        # (RPM/RMS는 이미 scalar이므로 별도 LSTM 불필요)
        h_aux = x_auxiliary[:, -1, :]  # (batch, auxiliary_dim)

        # Fusion: vibration attention + auxiliary → MLP → RUL
        fused_input = torch.cat([h_vib_attended, h_aux], dim=1)

        return self.fusion(fused_input)


class AsymmetricRULLoss(nn.Module):
    """
    Asymmetric penalty for RUL prediction.
    Overestimation (prediction > target) gets higher penalty.
    """

    def __init__(self, over_scale: float = OVER_EST_PENALTY_SCALE, under_scale: float = UNDER_EST_PENALTY_SCALE):
        super().__init__()
        self.over_scale = over_scale
        self.under_scale = under_scale

    def forward(self, predictions_real: torch.Tensor, targets_real: torch.Tensor) -> torch.Tensor:
        targets_real = targets_real.view_as(predictions_real)
        denominator = torch.clamp(targets_real, min=1e-6)
        
        # 백분율 오차(Percentage Error) 계산: 100 * (실제 - 예측) / 실제
        er = 100.0 * (targets_real - predictions_real) / denominator
        
        # A_RUL 점수 공식의 지수(Exponent)에 음수를 취한 값 (-exponent)
        # 이 식은 완벽한 L1 백분율 오차 형태로 작동하며, 기울기 소실(Vanishing Gradient)이 발생하지 않습니다.
        ln_two = 0.69314718
        loss = torch.where(
            er <= 0,
            ln_two * (-er) / self.over_scale,  # 과대평가 패널티
            ln_two * er / self.under_scale,   # 과소평가 패널티
        )
        return loss.mean()


class CombinedLoss(nn.Module):
    """
    Weighted combination of HuberLoss and AsymmetricRULLoss.
    """

    def __init__(self, huber_weight: float = HUBER_WEIGHT, asymmetric_weight: float = ASYMMETRIC_WEIGHT):
        super().__init__()
        self.huber = nn.HuberLoss()
        self.asymmetric = AsymmetricRULLoss()
        self.huber_weight = huber_weight
        self.asymmetric_weight = asymmetric_weight

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Huber Loss는 학습 안정성을 위해 로그 스케일에서 계산 (MSLE와 유사)
        huber_loss = self.huber(predictions, targets)
        
        # 수치 폭발 방지를 위해 로그 스케일 출력을 안전하게 클리핑 (max RUL ~35000 -> log1p ~10.46)
        predictions_clamped = torch.clamp(predictions, max=11.5)
        
        # Asymmetric Loss는 평가 지표와 동일하게 진짜 스케일(Real Space)의 백분율 오차로 계산
        pred_real = torch.expm1(predictions_clamped)
        target_real = torch.expm1(targets)
        asymmetric_loss = self.asymmetric(pred_real, target_real)
        
        return self.huber_weight * huber_loss + self.asymmetric_weight * asymmetric_loss  #로스 함수 


def asymmetric_rul_score_np(predictions, targets) -> np.ndarray:
    """평가용 A_RUL score 계산."""
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    denominator = np.maximum(np.abs(targets), 1e-6)
    er = 100.0 * (targets - predictions) / denominator
    ln_half = np.log(0.5)
    exponent = np.where(
        er <= 0,
        -ln_half * er / 20.0,
        ln_half * er / 50.0,
    )
    return np.exp(exponent)


def create_model(
    vibration_channels: int = 4,
    auxiliary_dim: int = AUXILIARY_DIM,
    vibration_features: int = VIBRATION_FEATURES_PER_CHANNEL,
) -> STFTCNNLSTMRULModel:
    return STFTCNNLSTMRULModel(
        vibration_channels=vibration_channels,
        auxiliary_dim=auxiliary_dim,
        vibration_features=vibration_features,
        vib_hidden=128,
    )
