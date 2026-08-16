# NEXRAD Radar Fusion Model

This document describes the radar-fusion upgrade to the tornado predictor: why it exists, how
it's built, how the data pipeline and training work, and how to run it.

## Motivation

The baseline `TornadoTransformer` sees only the **atmospheric environment** (Open-Meteo
features). That's enough to learn *where conditions favor tornadoes*, but it plateaus because it
never sees the **storm itself**. In offline evaluation the atmosphere-only model saturates
(~0.97 AUC on its own split) while missing the storm-scale structure that separates a tornadic
supercell from a non-tornadic one.

Adding **NEXRAD (WSR-88D) radar** gives the model storm-scale observations — reflectivity,
velocity, spectrum width, differential reflectivity, correlation coefficient — which is where the
real, generalizable signal lives. The target is SOTA-range skill (~0.93–0.95 AUC on held-out
years) with calibrated lead time.

## Architecture — two towers + fusion head

```
radar chips  ─▶  RadarBackbone (TorNet CNN, ~8M params)  ─▶  256-dim storm embedding ─┐
                                                                                       ├─▶ fusion head (512→…) ─▶ tornado_prob + intensity_logits
24h atm seq  ─▶  TransformerBackbone (from baseline model)  ─▶  256-dim atm embedding ─┘
```

- **Radar tower** (`backend/ml/radar.py`): a TorNet-style CNN backbone that turns multi-channel
  radar chips into a 256-dim storm embedding. Loadable from a pretrained checkpoint.
- **Atmospheric tower** (`backend/ml/model.py`): the baseline transformer's backbone, reused.
- **Fusion head** (`backend/ml/fusion_model.py`, `FusionTornadoModel`): concatenates the two
  256-dim embeddings and predicts tornado probability + intensity class. ~13.3M params total.
  Passing `radar_data=None` falls back to atmosphere-only inference.

## Data pipeline

- **TorNet** — a curated dataset of NEXRAD radar chips labeled for tornado occurrence
  (2013–2022, ~150 GB). Downloaded and extracted under `backend/data/tornet/` (git-ignored).
- **Pairing** (`run_build_fusion_dataset.py`): joins each TorNet radar chip with the matching
  atmospheric feature vector (ERA5 / Open-Meteo) at that location/time, writing paired samples to
  `backend/data/fusion_cache/paired_samples_<year>.pkl` (git-ignored; tens of GB per year).
- **Live radar** (`backend/ml/nexrad.py`): fetches recent scans directly from the NOAA NEXRAD
  archive on AWS S3 (159 WSR-88D stations registered) for real-time inference.

## Training — 3 phases

`run_fusion_training.py` trains in escalating phases to avoid clobbering the pretrained backbones:

| Phase | What's trainable | Purpose |
|-------|------------------|---------|
| **F1** | fusion head only (both backbones frozen) | learn to combine embeddings cheaply |
| **F2** | + last 2 radar blocks & last 2 atm encoder layers | light adaptation |
| **F3** | all layers | full fine-tune |

## How to run

```bash
# 1) Download + extract TorNet into backend/data/tornet/ (see monitor_tornet_downloads.py)
# 2) Build the paired dataset (radar chip ↔ atmospheric features)
python run_build_fusion_dataset.py --tornet_dir backend/data/tornet --years 2020 2021 2022
# 3) Train the fusion model (phases F1→F2→F3)
python run_fusion_training.py
```

Weights and datasets are **not** committed — see `.gitignore`.

## Files added by this feature

| File | Role |
|------|------|
| `backend/ml/radar.py` | TorNet CNN radar backbone → 256-dim storm embedding |
| `backend/ml/nexrad.py` | Live NEXRAD fetcher (AWS S3, WSR-88D stations) |
| `backend/ml/fusion_model.py` | `FusionTornadoModel` two-tower + fusion head |
| `run_build_fusion_dataset.py` | Pairs TorNet chips with atmospheric features |
| `run_fusion_training.py` | 3-phase fusion training loop |
| `run_fetch_nexrad_recent.py` | Pull recent live radar scans |
| `monitor_tornet_downloads.py` | Download progress monitor + auto-extract |
