"""
Fusion model training: radar backbone + atmospheric transformer.

Requires paired dataset built by run_build_fusion_dataset.py.

Phases:
  F1 (10 epochs): Both backbones frozen — train fusion head only
  F2 (20 epochs): Unfreeze last 2 radar VGG blocks + last 2 atm encoder layers
  F3 (20 epochs): Full fine-tune, low LR

Usage:
  python run_fusion_training.py --fusion_data backend/data/fusion_cache
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from dotenv import load_dotenv
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score

load_dotenv("backend/.env")
sys.path.insert(0, "backend")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

from ml.data import AtmosphericScaler, TornadoDataset
from ml.model import N_FEATURES
from ml.radar import build_coordinate_tensor, ALL_VARIABLES
from ml.fusion_model import FusionTornadoModel, FusionConfig, PairedRadarAtmDataset

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_DIR = Path("backend/checkpoints")
log.info(f"Device: {DEVICE}")

COORD_TENSOR = build_coordinate_tensor()   # [2, 120, 240]


def ef_to_intensity_class(ef: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    cls = torch.zeros_like(ef, dtype=torch.long)
    cls = torch.where((label > 0.5) & (ef < 3),  torch.ones_like(cls),    cls)
    cls = torch.where((label > 0.5) & (ef >= 3), torch.full_like(cls, 2), cls)
    return cls


def radar_collate(batch):
    """Custom collate: handles None radar_data by using atm-only path."""
    atm     = torch.stack([b["atm_features"] for b in batch])
    labels  = torch.stack([b["label"] for b in batch])
    ef      = torch.stack([b["ef_scale"] for b in batch])
    radar_present = [b["radar_data"] is not None for b in batch]
    if all(radar_present):
        radar = {k: torch.stack([b["radar_data"][k] for b in batch])
                 for k in batch[0]["radar_data"]}
    elif not any(radar_present):
        radar = None
    else:
        # Mixed batch — use atm-only for items with missing radar
        radar = None
    return {"atm_features": atm, "radar_data": radar, "label": labels, "ef_scale": ef}


def make_sampler(samples):
    """Balanced sampler: oversample positives to ~40% of batches."""
    weights = [4.0 if s.label == 1 else 1.0 for s in samples]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def evaluate(model: FusionTornadoModel, loader: DataLoader):
    model.eval()
    probs_all, labels_all = [], []
    i_preds, i_true = [], []
    with torch.no_grad():
        for batch in loader:
            atm    = batch["atm_features"].to(DEVICE)
            radar  = {k: v.to(DEVICE) for k, v in batch["radar_data"].items()} \
                     if batch["radar_data"] is not None else None
            y      = batch["label"].to(DEVICE)
            ef     = batch["ef_scale"].to(DEVICE)
            out    = model(atm, radar)
            prob   = out["tornado_prob"].mean(dim=1)
            logits = out["intensity_logits"]
            i_preds.extend(logits.argmax(dim=-1).cpu().tolist())
            i_true.extend(ef_to_intensity_class(ef, y).cpu().tolist())
            probs_all.extend(prob.detach().cpu().tolist())
            labels_all.extend(y.cpu().tolist())
    p  = np.array(probs_all); l  = np.array(labels_all)
    ip = np.array(i_preds);   it = np.array(i_true)
    auc  = roc_auc_score(l, p) if len(set(l)) > 1 else 0.5
    preds_bin = (p > 0.5).astype(int)
    tp = int(((preds_bin==1)&(l==1)).sum()); fp = int(((preds_bin==1)&(l==0)).sum())
    fn = int(((preds_bin==0)&(l==1)).sum())
    pod  = tp / (tp + fn + 1e-8); far  = fp / (tp + fp + 1e-8)
    f1   = 2*tp / (2*tp + fp + fn + 1e-8)
    none_acc = float((ip[it==0]==0).mean()) if (it==0).any() else 0.0
    weak_acc = float((ip[it==1]==1).mean()) if (it==1).any() else 0.0
    sig_acc  = float((ip[it==2]==2).mean()) if (it==2).any() else 0.0
    hmean = 3/(1/(none_acc+1e-8)+1/(weak_acc+1e-8)+1/(sig_acc+1e-8))
    return dict(auc=auc, pod=pod, far=far, f1=f1,
                none=none_acc, weak=weak_acc, sig=sig_acc, hmean=hmean)


def run_phase(
    model, loader, val_loader, epochs, lr,
    phase_name, best_f1, freeze_fn=None, unfreeze_fn=None,
):
    if freeze_fn:   freeze_fn()
    if unfreeze_fn: unfreeze_fn()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"--- {phase_name} ({trainable:,} trainable params) ---")

    bce = nn.BCELoss()
    ce  = nn.CrossEntropyLoss(weight=torch.tensor([0.1, 0.4, 15.0], device=DEVICE))
    opt  = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in loader:
            atm   = batch["atm_features"].to(DEVICE)
            radar = {k: v.to(DEVICE) for k, v in batch["radar_data"].items()} \
                    if batch["radar_data"] is not None else None
            y     = batch["label"].to(DEVICE)
            ef    = batch["ef_scale"].to(DEVICE)
            out   = model(atm, radar)
            prob  = out["tornado_prob"].mean(dim=1)
            logits = out["intensity_logits"]
            i_cls = ef_to_intensity_class(ef, y)
            loss  = bce(prob, y) + 0.3 * ce(logits, i_cls)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        if epoch % 5 == 0 or epoch == 1:
            r = evaluate(model, val_loader)
            if r["f1"] > best_f1:
                best_f1 = r["f1"]
                model.save(CKPT_DIR / "fusion_best.pt", {"phase": phase_name, "f1": best_f1})
            log.info(f"  {phase_name} Epoch {epoch:3d}/{epochs} — "
                     f"AUC:{r['auc']:.4f}  POD:{r['pod']*100:.1f}%  FAR:{r['far']*100:.1f}%  "
                     f"F1:{r['f1']*100:.1f}%  "
                     f"Int:[None:{r['none']*100:.0f}% Weak:{r['weak']*100:.0f}% EF3+:{r['sig']*100:.0f}%]")
    return best_f1


def main(fusion_data_dir: Path):
    # ── Load paired samples ────────────────────────────────────────────────────
    all_samples = []
    for pkl in sorted(fusion_data_dir.glob("paired_samples_*.pkl")):
        with open(pkl, "rb") as f:
            while True:
                try:
                    all_samples.extend(pickle.load(f))
                except EOFError:
                    break
    log.info(f"Loaded {len(all_samples):,} paired samples")

    # Temporal split:
    #   Train  2013-2023 — TorNet radar chips + NEXRAD-fetched 2023
    #   Val    2024      — NEXRAD-fetched, matches our atmospheric val split
    #   Test   2025      — NEXRAD-fetched, matches our atmospheric test split
    train_s = [s for s in all_samples if s.valid_time.year <= 2023]
    val_s   = [s for s in all_samples if s.valid_time.year == 2024]
    test_s  = [s for s in all_samples if s.valid_time.year == 2025]
    log.info(f"Train:{len(train_s):,}  Val:{len(val_s):,}  Test:{len(test_s):,}")

    # Fit scaler on train atm features
    scaler = AtmosphericScaler()
    scaler.fit(np.stack([s.atm_features for s in train_s]))

    train_ds = PairedRadarAtmDataset(train_s, scaler, COORD_TENSOR)
    val_ds   = PairedRadarAtmDataset(val_s,   scaler, COORD_TENSOR)
    test_ds  = PairedRadarAtmDataset(test_s,  scaler, COORD_TENSOR)

    train_loader = DataLoader(train_ds, batch_size=32, sampler=make_sampler(train_s),
                              num_workers=0, collate_fn=radar_collate)
    val_loader   = DataLoader(val_ds,  batch_size=64, shuffle=False,
                              num_workers=0, collate_fn=radar_collate)
    test_loader  = DataLoader(test_ds, batch_size=64, shuffle=False,
                              num_workers=0, collate_fn=radar_collate)

    # ── Build model ────────────────────────────────────────────────────────────
    cfg   = FusionConfig()
    model = FusionTornadoModel(cfg, device=str(DEVICE)).to(DEVICE)

    best_f1 = 0.0

    # Phase F1 — fusion head only
    best_f1 = run_phase(model, train_loader, val_loader, epochs=10, lr=1e-3,
                        phase_name="F1", best_f1=best_f1,
                        freeze_fn=model.freeze_backbones)

    # Phase F2 — light unfreeze
    best_f1 = run_phase(model, train_loader, val_loader, epochs=20, lr=3e-5,
                        phase_name="F2", best_f1=best_f1,
                        unfreeze_fn=model.unfreeze_last_layers)

    # Phase F3 — full fine-tune
    best_f1 = run_phase(model, train_loader, val_loader, epochs=20, lr=1e-5,
                        phase_name="F3", best_f1=best_f1,
                        unfreeze_fn=model.unfreeze_all)

    # ── Final evaluation ───────────────────────────────────────────────────────
    best_model = FusionTornadoModel.load(CKPT_DIR / "fusion_best.pt", device=str(DEVICE)).to(DEVICE)
    log.info("--- Final Evaluation ---")
    for name, loader in [("VAL 2024", val_loader), ("TEST 2025", test_loader)]:
        r = evaluate(best_model, loader)
        log.info(f"{name}: AUC:{r['auc']:.4f}  POD:{r['pod']*100:.1f}%  "
                 f"FAR:{r['far']*100:.1f}%  F1:{r['f1']*100:.1f}%")
        log.info(f"  Intensity — None:{r['none']*100:.1f}%  Weak:{r['weak']*100:.1f}%  "
                 f"EF3+:{r['sig']*100:.1f}%  Harmonic:{r['hmean']*100:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fusion_data", type=Path,
                        default=Path("backend/data/fusion_cache"))
    args = parser.parse_args()
    main(args.fusion_data)
