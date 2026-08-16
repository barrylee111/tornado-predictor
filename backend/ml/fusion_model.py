"""
FusionTornadoModel — combines NEXRAD radar backbone with atmospheric transformer.

Two-tower architecture:
  Tower 1 (radar):       RadarBackbone  → 256-dim storm-scale embedding
  Tower 2 (atmosphere):  TransformerBackbone → 256-dim atmospheric embedding
  Fusion head:           [512-dim] → tornado_prob + intensity_logits

Training phases:
  Phase F1: Both backbones frozen — train fusion head only
  Phase F2: Unfreeze last 2 radar VGG blocks + last 2 atm encoder layers
  Phase F3: Full fine-tune (all layers)
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import numpy as np

from ml.model import TornadoTransformer, ModelConfig, N_FEATURES, INTENSITY_CLASS_NAMES
from ml.radar import RadarBackbone, EMBEDDING_DIM as RADAR_EMB_DIM, load_pretrained_radar_backbone


ATM_EMB_DIM = 256   # d_model from ModelConfig


@dataclass
class FusionConfig:
    atm_model_path:   Path = Path("checkpoints/finetuned_best.pt")
    radar_ckpt_path:  Path = Path("checkpoints/tornet_pretrained.pt")
    n_intensity_classes: int = 3
    fusion_hidden:    int = 512
    dropout:          float = 0.2
    forecast_hours:   int = 2


class FusionTornadoModel(nn.Module):
    """
    Combined radar + atmospheric tornado prediction model.

    Forward inputs:
        atm_features:  [B, 24, N_FEATURES]  — 24h atmospheric sequence
        radar_data:    dict of radar tensors  — see RadarBackbone
                       Pass None to run atmosphere-only fallback.
    """

    def __init__(self, cfg: FusionConfig = FusionConfig(), device: str = "cpu"):
        super().__init__()
        self.cfg = cfg

        # ── Atmospheric tower ─────────────────────────────────────────────────
        atm_model = TornadoTransformer.load(cfg.atm_model_path, device=device)
        self.atm_backbone = atm_model.backbone   # TransformerBackbone

        # ── Radar tower ───────────────────────────────────────────────────────
        self.radar_backbone = load_pretrained_radar_backbone(
            device=device,
            cache_dir=cfg.radar_ckpt_path.parent,
        )

        # ── Fusion head ───────────────────────────────────────────────────────
        combined_dim = RADAR_EMB_DIM + ATM_EMB_DIM   # 256 + 256 = 512
        h = cfg.fusion_hidden
        self.fusion_mlp = nn.Sequential(
            nn.Linear(combined_dim, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(h, h // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.tornado_head = nn.Sequential(
            nn.Linear(h // 2, cfg.forecast_hours),
        )
        self.intensity_head = nn.Sequential(
            nn.Linear(h // 2, cfg.n_intensity_classes),
        )

        # Atm-only head (used when radar unavailable)
        self.atm_only_mlp = nn.Sequential(
            nn.Linear(ATM_EMB_DIM, h // 2),
            nn.LayerNorm(h // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.atm_only_tornado  = nn.Linear(h // 2, cfg.forecast_hours)
        self.atm_only_intensity = nn.Linear(h // 2, cfg.n_intensity_classes)

    def forward(
        self,
        atm_features: torch.Tensor,
        radar_data: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        # ── Atmospheric embedding ─────────────────────────────────────────────
        atm_enc = self.atm_backbone(atm_features)           # [B, 24, 256]
        atm_emb = atm_enc.mean(dim=1)                       # [B, 256]

        if radar_data is None:
            # Atmosphere-only fallback
            h = self.atm_only_mlp(atm_emb)
            return {
                "tornado_prob":     torch.sigmoid(self.atm_only_tornado(h)),
                "intensity_logits": self.atm_only_intensity(h),
                "radar_available":  torch.tensor(False),
            }

        # ── Radar embedding ───────────────────────────────────────────────────
        radar_emb, heatmap = self.radar_backbone(radar_data)   # [B, 256], [B,1,7,15]

        # ── Fusion ────────────────────────────────────────────────────────────
        combined = torch.cat([atm_emb, radar_emb], dim=1)      # [B, 512]
        h = self.fusion_mlp(combined)
        return {
            "tornado_prob":     torch.sigmoid(self.tornado_head(h)),
            "intensity_logits": self.intensity_head(h),
            "radar_heatmap":    heatmap,
            "radar_available":  torch.tensor(True),
        }

    # ── Freeze / unfreeze helpers ─────────────────────────────────────────────
    def freeze_backbones(self):
        self.atm_backbone.freeze_backbone() if hasattr(self.atm_backbone, 'freeze_backbone') \
            else _freeze(self.atm_backbone)
        self.radar_backbone.freeze()

    def unfreeze_last_layers(self):
        """Phase F2: light unfreeze for fine-tuning."""
        _freeze(self.atm_backbone)
        layers = list(self.atm_backbone.encoder.layers)
        for layer in layers[-2:]:
            for p in layer.parameters():
                p.requires_grad = True
        self.radar_backbone.unfreeze_last_n_blocks(n=2)

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True

    def save(self, path: Path, metadata: dict | None = None):
        torch.save({
            "state_dict": self.state_dict(),
            "config": self.cfg.__dict__,
            "metadata": metadata or {},
        }, path)

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "FusionTornadoModel":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg  = FusionConfig(**{k: v for k, v in ckpt["config"].items()
                                if k in FusionConfig.__dataclass_fields__})
        model = cls(cfg, device=device)
        model.load_state_dict(ckpt["state_dict"])
        return model


def _freeze(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False


# ── Paired dataset for fusion training ───────────────────────────────────────
class PairedRadarAtmDataset(torch.utils.data.Dataset):
    """
    Dataset that provides (atm_features, radar_chip_dict, label, ef_scale) tuples.
    Loaded from a pre-built paired cache pickle.
    """

    def __init__(self, samples: list, scaler, coord_tensor: torch.Tensor):
        self.samples = samples
        self.scaler  = scaler
        self.coords  = coord_tensor   # [2, 120, 240] — shared across all samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        atm = s.atm_features.copy()   # [24, N_FEATURES]
        if self.scaler:
            atm = self.scaler.transform(atm)
        atm_t = torch.tensor(atm, dtype=torch.float32)

        # Build radar tensor dict
        from ml.radar import ALL_VARIABLES
        radar: Dict[str, torch.Tensor] = {}
        if hasattr(s, "radar_chip") and s.radar_chip is not None:
            for v in ALL_VARIABLES:
                arr = s.radar_chip.get(v, np.zeros((2, 120, 240), dtype=np.float32))
                radar[v] = torch.tensor(arr, dtype=torch.float32)
            rfm = s.radar_chip.get("range_folded_mask", np.zeros((2, 120, 240), dtype=np.float32))
            radar["range_folded_mask"] = torch.tensor(rfm, dtype=torch.float32)
            radar["coordinates"] = self.coords
        else:
            # No radar — return None marker; collate_fn will handle
            radar = None

        return {
            "atm_features": atm_t,
            "radar_data":   radar,
            "label":        torch.tensor(float(s.label), dtype=torch.float32),
            "ef_scale":     torch.tensor(getattr(s, "ef_scale", 0.0), dtype=torch.float32),
        }
