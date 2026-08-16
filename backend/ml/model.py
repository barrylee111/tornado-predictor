"""
TornadoTransformer — temporal attention over atmospheric features.

Architecture:
  - Feature embedding + positional encoding
  - Transformer encoder backbone (pretrainable on broad weather forecasting)
  - Tornado classification head (fine-tunable independently)
  - Intensity regression head

Backbone can be frozen for fine-tuning runs then progressively unfrozen.
"""

import torch
import torch.nn as nn
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# Atmospheric features the model ingests at each time step.
FEATURE_NAMES = [
    "cape",           # J/kg — storm fuel
    "cin",            # J/kg — convective cap
    "srh_01km",       # m²/s² — low-level rotation potential
    "srh_03km",       # m²/s² — mid-level rotation potential
    "shear_06km",     # m/s  — storm organization
    "shear_01km",     # m/s  — low-level shear
    "lcl_height",     # m    — cloud base (lower = better for tornadoes)
    "lfc_height",     # m    — level of free convection
    "pwat",           # mm   — precipitable water
    "temp_2m",        # K
    "dewpoint_2m",    # K
    "u_wind_10m",     # m/s
    "v_wind_10m",     # m/s
    "u_wind_500mb",   # m/s
    "v_wind_500mb",   # m/s
    "temp_500mb",     # K
    "pressure_msl",   # hPa
    "lifted_index",   # K    — negative = unstable
    "k_index",        # K
    "total_totals",   # K
]

N_FEATURES = len(FEATURE_NAMES)
FEATURE_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}


@dataclass
class ModelConfig:
    n_features: int = N_FEATURES
    seq_len: int = 24           # hours of history fed to the model
    d_model: int = 256
    n_heads: int = 8
    n_encoder_layers: int = 6
    d_ff: int = 1024
    dropout: float = 0.1
    forecast_hours: int = 2     # predict risk at t+1h and t+2h
    max_seq_len: int = 128
    n_intensity_classes: int = 3  # 0=None, 1=Weak (EF0-2), 2=Significant (EF3+)


INTENSITY_CLASS_NAMES = ["None", "Weak (EF0-2)", "Significant (EF3+)"]
INTENSITY_CLASS_EF_MIDPOINT = [0.0, 1.0, 4.0]  # representative EF for each class


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerBackbone(nn.Module):
    """
    Pretrain this on broad atmospheric forecasting (predict t+6h conditions
    from the past 24h) before fine-tuning the full model on tornado labels.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.input_proj = nn.Linear(cfg.n_features, cfg.d_model)
        self.pos_enc = SinusoidalPositionalEncoding(
            cfg.d_model, cfg.max_seq_len, cfg.dropout
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,  # pre-norm for training stability
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg.n_encoder_layers
        )
        self.norm = nn.LayerNorm(cfg.d_model)

        # Pretraining head: reconstruct next-step atmospheric conditions
        self.pretrain_head = nn.Linear(cfg.d_model, cfg.n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, n_features) → (batch, seq_len, d_model)"""
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        return self.norm(x)

    def pretrain_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns predicted next-step features for pretraining."""
        encoded = self.forward(x)
        return self.pretrain_head(encoded)


class TornadoPredictionHead(nn.Module):
    """
    Fine-tune this head (and optionally the backbone) on SPC tornado labels.
    Outputs per-timestep tornado probability for each forecast hour.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.forecast_hours = cfg.forecast_hours
        self.attn_pool = nn.MultiheadAttention(
            embed_dim=cfg.d_model, num_heads=4, batch_first=True
        )
        self.query = nn.Parameter(torch.randn(1, cfg.forecast_hours, cfg.d_model))
        self.classifier = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.d_model // 2, 1),
        )
        self.intensity_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.d_model // 2, cfg.n_intensity_classes),
            # No activation — raw logits for CrossEntropyLoss
        )

    def forward(self, encoded: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        encoded: (batch, seq_len, d_model)
        Returns:
          tornado_prob:     (batch, forecast_hours) — sigmoid probability
          intensity_logits: (batch, forecast_hours, n_intensity_classes) — CE logits
        """
        batch = encoded.size(0)
        query = self.query.expand(batch, -1, -1)
        context, _ = self.attn_pool(query, encoded, encoded)
        prob = self.classifier(context).squeeze(-1)
        intensity_logits = self.intensity_head(context)
        return {
            "tornado_prob": torch.sigmoid(prob),
            "intensity_logits": intensity_logits,
        }


class TornadoTransformer(nn.Module):
    def __init__(self, cfg: Optional[ModelConfig] = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        self.backbone = TransformerBackbone(self.cfg)
        self.head = TornadoPredictionHead(self.cfg)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = self.backbone(x)
        return self.head(encoded)

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.backbone.pretrain_head.parameters():
            p.requires_grad = True  # keep pretrain head unfrozen for aux loss

    def unfreeze_backbone(self, unfreeze_last_n_layers: int = 2):
        for p in self.backbone.parameters():
            p.requires_grad = False
        layers = list(self.backbone.encoder.layers)
        for layer in layers[-unfreeze_last_n_layers:]:
            for p in layer.parameters():
                p.requires_grad = True

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True

    def save(self, path: Path, metadata: dict | None = None):
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": self.cfg.__dict__,
                "metadata": metadata or {},
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "TornadoTransformer":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = ModelConfig(**ckpt["config"])
        model = cls(cfg)
        model.load_state_dict(ckpt["state_dict"])
        return model
