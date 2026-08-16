"""
Training script for TornadoTransformer.

Two-phase training:
  Phase 1 — Pretrain backbone on atmospheric forecasting (predict t+6h features).
  Phase 2 — Fine-tune on tornado labels.
             Sub-phase 2a: freeze backbone, train head only (fast convergence).
             Sub-phase 2b: unfreeze last N backbone layers + head (full fine-tune).

Usage:
    python -m ml.train --phase pretrain --epochs 50
    python -m ml.train --phase finetune --checkpoint checkpoints/pretrained.pt
    python -m ml.train --phase finetune --checkpoint checkpoints/pretrained.pt --unfreeze-layers 4
"""

import argparse
import asyncio
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from ml.data import AtmosphericScaler, TornadoDataset, build_dataset
from ml.model import ModelConfig, TornadoTransformer, N_FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
CHECKPOINT_DIR = Path("checkpoints")
SCALER_PATH = Path("checkpoints/scaler.pkl")


def make_weighted_sampler(dataset: TornadoDataset) -> WeightedRandomSampler:
    labels = [s.label for s in dataset.samples]
    pos = sum(labels)
    neg = len(labels) - pos
    weight_pos = 1.0 / pos if pos > 0 else 0.0
    weight_neg = 1.0 / neg if neg > 0 else 0.0
    weights = [weight_pos if l == 1 else weight_neg for l in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def pretrain_epoch(model: TornadoTransformer, loader: DataLoader, optimizer, scaler_amp) -> float:
    model.train()
    total_loss = 0.0
    criterion = nn.MSELoss()
    for batch in loader:
        x = batch["features"].to(DEVICE)       # (B, seq_len, N_FEATURES)
        # Predict next step: shift by 1
        inp = x[:, :-1, :]
        target = x[:, 1:, :]
        optimizer.zero_grad()
        with torch.autocast(device_type=DEVICE.type, enabled=scaler_amp is not None):
            pred = model.backbone.pretrain_forward(inp)
            loss = criterion(pred, target)
        if scaler_amp:
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def finetune_epoch(model: TornadoTransformer, loader: DataLoader, optimizer, scaler_amp) -> dict:
    model.train()
    bce = nn.BCELoss()
    mse = nn.MSELoss()
    total_bce = 0.0
    correct = 0
    total = 0
    for batch in loader:
        x = batch["features"].to(DEVICE)
        labels = batch["label"].to(DEVICE)
        optimizer.zero_grad()
        with torch.autocast(device_type=DEVICE.type, enabled=scaler_amp is not None):
            out = model(x)
            # Average probability across forecast hours
            prob = out["tornado_prob"].mean(dim=1)
            loss = bce(prob, labels)
        if scaler_amp:
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total_bce += loss.item()
        preds = (prob > 0.5).float()
        correct += (preds == labels).sum().item()
        total += len(labels)
    return {"loss": total_bce / len(loader), "acc": correct / total}


@torch.no_grad()
def evaluate(model: TornadoTransformer, loader: DataLoader) -> dict:
    model.eval()
    bce = nn.BCELoss()
    total_loss = 0.0
    all_probs, all_labels = [], []
    for batch in loader:
        x = batch["features"].to(DEVICE)
        labels = batch["label"].to(DEVICE)
        out = model(x)
        prob = out["tornado_prob"].mean(dim=1)
        loss = bce(prob, labels)
        total_loss += loss.item()
        all_probs.extend(prob.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds = (all_probs > 0.5).astype(int)
    tp = ((preds == 1) & (all_labels == 1)).sum()
    fp = ((preds == 1) & (all_labels == 0)).sum()
    fn = ((preds == 0) & (all_labels == 1)).sum()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return {
        "val_loss": total_loss / len(loader),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


async def run_pretrain(args):
    logger.info("Building dataset for pretraining...")
    samples = await build_dataset(list(range(2010, 2023)))
    all_feats = np.stack([s.features for s in samples])
    scaler = AtmosphericScaler()
    scaler.fit(all_feats)
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    scaler.save(SCALER_PATH)

    dataset = TornadoDataset(samples, scaler)
    train_ds, val_ds = dataset.split(val_frac=0.1)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=128, num_workers=2)

    model = TornadoTransformer().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler_amp = torch.amp.GradScaler() if DEVICE.type == "cuda" else None

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        loss = pretrain_epoch(model, train_loader, optimizer, scaler_amp)
        scheduler.step()
        logger.info(f"Pretrain epoch {epoch}/{args.epochs} — loss={loss:.4f}")
        if loss < best_loss:
            best_loss = loss
            model.save(
                CHECKPOINT_DIR / "pretrained_best.pt",
                {"epoch": epoch, "pretrain_loss": loss},
            )
    model.save(CHECKPOINT_DIR / "pretrained_final.pt", {"epochs": args.epochs})
    logger.info("Pretraining complete.")


async def run_finetune(args):
    logger.info("Building dataset for fine-tuning...")
    samples = await build_dataset(list(range(2010, 2024)))
    scaler = AtmosphericScaler.load(SCALER_PATH) if SCALER_PATH.exists() else None
    if scaler is None:
        logger.warning("No scaler found — fitting new scaler on fine-tune data")
        all_feats = np.stack([s.features for s in samples])
        scaler = AtmosphericScaler()
        scaler.fit(all_feats)
        CHECKPOINT_DIR.mkdir(exist_ok=True)
        scaler.save(SCALER_PATH)

    dataset = TornadoDataset(samples, scaler)
    train_ds, val_ds = dataset.split(val_frac=0.1)
    sampler = make_weighted_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=64, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=128, num_workers=2)

    if args.checkpoint:
        model = TornadoTransformer.load(Path(args.checkpoint), device=str(DEVICE)).to(DEVICE)
        logger.info(f"Loaded checkpoint from {args.checkpoint}")
    else:
        model = TornadoTransformer().to(DEVICE)
        logger.info("No checkpoint — training from scratch")

    # Phase 2a: head only
    model.freeze_backbone()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
    )
    scaler_amp = torch.amp.GradScaler() if DEVICE.type == "cuda" else None

    logger.info("Phase 2a: training head only (backbone frozen)")
    for epoch in range(1, 11):
        metrics = finetune_epoch(model, train_loader, optimizer, scaler_amp)
        val = evaluate(model, val_loader)
        logger.info(f"Epoch {epoch}/10 — {metrics} | val: {val}")

    # Phase 2b: progressive unfreeze
    unfreeze_n = args.unfreeze_layers
    logger.info(f"Phase 2b: unfreezing last {unfreeze_n} backbone layers")
    model.unfreeze_backbone(unfreeze_last_n_layers=unfreeze_n)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=3e-5, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1 = 0.0
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        metrics = finetune_epoch(model, train_loader, optimizer, scaler_amp)
        scheduler.step()
        val = evaluate(model, val_loader)
        logger.info(f"Finetune epoch {epoch}/{args.epochs} — {metrics} | val: {val}")
        if val["f1"] > best_f1:
            best_f1 = val["f1"]
            model.save(
                CHECKPOINT_DIR / "finetuned_best.pt",
                {"epoch": epoch, **val},
            )

    model.save(CHECKPOINT_DIR / "finetuned_final.pt", {"epochs": args.epochs})
    logger.info(f"Fine-tuning complete. Best F1: {best_f1:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["pretrain", "finetune"], required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--unfreeze-layers", type=int, default=2)
    args = parser.parse_args()

    logger.info(f"Device: {DEVICE}")
    if args.phase == "pretrain":
        asyncio.run(run_pretrain(args))
    else:
        asyncio.run(run_finetune(args))


if __name__ == "__main__":
    main()
