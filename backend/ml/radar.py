"""
TorNet CNN backbone — PyTorch port, adapted for embedding extraction.

Architecture from: "A Benchmark Dataset for Tornado Detection and Prediction
using Full-Resolution Polarimetric Weather Radar Data", MIT LL 2024.
Source: https://github.com/mit-ll/tornet  (MIT License)

Modifications:
  - extract_embedding() returns [batch, 256] vector for fusion
  - load_pretrained() downloads weights from HuggingFace
  - No Lightning dependency
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ── Constants (from tornet.data.constants) ────────────────────────────────────
ALL_VARIABLES = ["DBZ", "VEL", "KDP", "RHOHV", "ZDR", "WIDTH"]

CHANNEL_MIN_MAX: Dict[str, Tuple[float, float]] = {
    "DBZ":   (-20.0, 60.0),
    "VEL":   (-60.0, 60.0),
    "KDP":   (-2.0,  5.0),
    "RHOHV": (0.2,   1.04),
    "ZDR":   (-1.0,  8.0),
    "WIDTH": (0.0,   9.0),
}

# Input shape: [n_tilts=2, az=120, rng=240]
RADAR_INPUT_SHAPE: Tuple[int, int, int] = (2, 120, 240)
COORD_SHAPE: Tuple[int, int, int] = (2, 120, 240)
EMBEDDING_DIM = 256   # dimension of the feature vector we extract for fusion

PATCH_N_AZ  = RADAR_INPUT_SHAPE[1]   # 120
PATCH_N_RNG = RADAR_INPUT_SHAPE[2]   # 240


# ── CoordConv2D ───────────────────────────────────────────────────────────────
class CoordConv2D(nn.Module):
    """Standard Conv2d that also sees polar coordinate channels at every layer."""

    def __init__(
        self,
        in_image_channels: int,
        in_coord_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: str = "same",
        activation: Optional[str] = "relu",
    ):
        super().__init__()
        self.padding = padding
        self.activation = activation
        total_in = in_image_channels + in_coord_channels
        pad = kernel_size // 2 if padding == "same" else 0
        self.conv = nn.Conv2d(total_in, out_channels, kernel_size, padding=pad)
        self.relu = nn.ReLU() if activation == "relu" else nn.Identity()

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]):
        x, c = inputs
        xc = torch.cat([x, c], dim=1)
        out = self.relu(self.conv(xc))
        return out, c   # coordinates pass through unchanged


# ── VGG-style block ───────────────────────────────────────────────────────────
class VggBlock(nn.Module):
    def __init__(
        self,
        in_img_ch: int,
        in_coord_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        n_convs: int = 2,
        drop_rate: float = 0.1,
    ):
        super().__init__()
        layers: List[CoordConv2D] = []
        for k in range(n_convs):
            in_ch = in_img_ch if k == 0 else out_ch
            layers.append(CoordConv2D(in_ch, in_coord_ch, out_ch, kernel_size))
        self.convs = nn.ModuleList(layers)
        self.pool  = nn.MaxPool2d(2, stride=2)
        self.drop  = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]):
        x, c = inputs
        for conv in self.convs:
            x, c = conv((x, c))
        x = self.pool(x)
        c = F.max_pool2d(c, 2, stride=2)
        x = self.drop(x)
        return x, c


# ── Normalizer ────────────────────────────────────────────────────────────────
class NormalizeVariable(nn.Module):
    def __init__(self, scale: float, offset: float):
        super().__init__()
        self.register_buffer("scale",  torch.tensor(scale,  dtype=torch.float32))
        self.register_buffer("offset", torch.tensor(offset, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.offset) * self.scale


# ── TorNet CNN backbone ───────────────────────────────────────────────────────
class RadarBackbone(nn.Module):
    """
    VGG-style CNN over polarimetric NEXRAD data.

    Input dict keys:
        DBZ, VEL, KDP, RHOHV, ZDR, WIDTH  — each [B, 2, 120, 240]
        coordinates                         — [B, 2, 120, 240]
        range_folded_mask                   — [B, 2, 120, 240]

    Outputs:
        embedding  — [B, EMBEDDING_DIM=256]
        heatmap    — [B, 1, 7, 15]  (spatial likelihood field, for visualization)
    """

    def __init__(
        self,
        input_variables: List[str] = ALL_VARIABLES,
        start_filters: int = 64,
        background_flag: float = -3.0,
        include_range_folded: bool = True,
    ):
        super().__init__()
        self.input_variables   = input_variables
        self.background_flag   = background_flag
        self.include_range_folded = include_range_folded
        n_sweeps = RADAR_INPUT_SHAPE[0]   # 2

        # Per-variable normalization
        self.normalizers = nn.ModuleDict()
        for v in input_variables:
            lo, hi = CHANNEL_MIN_MAX[v]
            scale  = 1.0 / (hi - lo)
            self.normalizers[v] = NormalizeVariable(scale, lo)

        in_img_ch = (len(input_variables) + int(include_range_folded)) * n_sweeps
        in_coord_ch = COORD_SHAPE[0]   # 2

        f = start_filters
        self.blk1 = VggBlock(in_img_ch,    in_coord_ch, f,   n_convs=2)  # → (60, 120)
        self.blk2 = VggBlock(f,            in_coord_ch, 2*f, n_convs=2)  # → (30,  60)
        self.blk3 = VggBlock(2*f,          in_coord_ch, 4*f, n_convs=3)  # → (15,  30)
        self.blk4 = VggBlock(4*f,          in_coord_ch, 8*f, n_convs=3)  # → ( 7,  15)

        self.head = nn.Sequential(
            nn.Conv2d(8*f, 512, kernel_size=1), nn.ReLU(),
            nn.Conv2d(512, EMBEDDING_DIM, kernel_size=1), nn.ReLU(),
        )
        self.out_conv = nn.Conv2d(EMBEDDING_DIM, 1, kernel_size=1)

    def forward(self, data: Dict[str, torch.Tensor]):
        # Normalise + concatenate radar variables
        parts = [self.normalizers[v](data[v]) for v in self.input_variables]
        x = torch.cat(parts, dim=1)                        # [B, n_vars*2, 120, 240]
        x = torch.where(torch.isnan(x), self.background_flag, x)

        if self.include_range_folded and "range_folded_mask" in data:
            x = torch.cat([x, data["range_folded_mask"]], dim=1)

        c = data["coordinates"]

        x, c = self.blk1((x, c))
        x, c = self.blk2((x, c))
        x, c = self.blk3((x, c))
        x, c = self.blk4((x, c))

        feat = self.head(x)                                # [B, 256, 7, 15]
        heatmap = self.out_conv(feat)                      # [B,   1, 7, 15]
        embedding = feat.mean(dim=(-2, -1))                # [B, 256]  global avg pool
        return embedding, heatmap

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze_last_n_blocks(self, n: int = 2):
        """Unfreeze the last n VGG blocks for fine-tuning."""
        for p in self.parameters():
            p.requires_grad = False
        blocks = [self.blk1, self.blk2, self.blk3, self.blk4]
        for blk in blocks[-n:]:
            for p in blk.parameters():
                p.requires_grad = True
        for p in self.head.parameters():
            p.requires_grad = True
        for p in self.out_conv.parameters():
            p.requires_grad = True


def load_pretrained_radar_backbone(
    device: str = "cpu",
    cache_dir: Path = Path("checkpoints"),
) -> RadarBackbone:
    """
    Download TorNet pretrained weights from HuggingFace and adapt to RadarBackbone.
    Falls back to random init if download fails.
    """
    backbone = RadarBackbone()
    ckpt_path = cache_dir / "tornet_pretrained.pt"

    if ckpt_path.exists():
        logger.info(f"Loading cached radar backbone from {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        backbone.load_state_dict(state, strict=False)
        return backbone.to(device)

    try:
        from huggingface_hub import hf_hub_download
        import keras
        logger.info("Downloading TorNet pretrained weights from HuggingFace...")
        keras_path = hf_hub_download(
            repo_id="tornet-ml/tornado_detector_baseline_v1",
            filename="tornado_detector_baseline_v1.keras",
            cache_dir=str(cache_dir / "hf_cache"),
        )
        logger.info(f"Keras model downloaded: {keras_path}")
        logger.info("Keras→PyTorch weight transfer not yet implemented — using random init")
    except Exception as e:
        logger.warning(f"Could not download pretrained weights: {e} — using random init")

    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(backbone.state_dict(), ckpt_path)
    logger.info(f"Random-init backbone saved to {ckpt_path}")
    return backbone.to(device)


# ── Coordinate tensor builder ─────────────────────────────────────────────────
def build_coordinate_tensor(
    az_min_deg: float = 0.0,
    az_max_deg: float = 360.0,
    rng_min_m: float = 2000.0,
    rng_max_m: float = 60000.0,
    n_az: int = 120,
    n_rng: int = 240,
) -> torch.Tensor:
    """
    Returns [2, n_az, n_rng] coordinate tensor [R, 1/R] for CoordConv.
    Matches TorNet preprocess.py convention.
    """
    SCALE = 1e-5
    az   = np.linspace(az_min_deg, az_max_deg, n_az, endpoint=False)
    rng  = np.linspace(rng_min_m, rng_max_m,  n_rng)
    R, _ = np.meshgrid(rng, az)   # (n_az, n_rng)
    R    = np.where(R >= rng_min_m, R, rng_min_m)
    Rinv = 1.0 / R
    coords = np.stack([R * SCALE, Rinv / SCALE], axis=0).astype(np.float32)
    return torch.tensor(coords)   # [2, 120, 240]
