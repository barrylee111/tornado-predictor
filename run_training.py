import sys, pickle, logging
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, WeightedRandomSampler
from dotenv import load_dotenv

load_dotenv('backend/.env')
sys.path.insert(0, 'backend')

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

from ml.model import TornadoTransformer, ModelConfig, N_FEATURES, INTENSITY_CLASS_NAMES
from ml.data import AtmosphericScaler, TornadoDataset

DATA_DIR = Path('backend/data')
CKPT_DIR = Path('backend/checkpoints')
DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log.info(f'Device: {DEVICE}')

# ── 1. Load & split ───────────────────────────────────────────────────────────
log.info('Loading combined cache...')
with open(DATA_DIR / 'cache/samples_2015_2025.pkl', 'rb') as f:
    all_samples = pickle.load(f)

samples      = [s for s in all_samples if s.valid_time.year <= 2023]
val_samples  = [s for s in all_samples if s.valid_time.year == 2024]
test_samples = [s for s in all_samples if s.valid_time.year == 2025]

pos = sum(s.label for s in samples)
log.info(f'Train: {len(samples):,}  pos={pos:,}  years={sorted(set(s.valid_time.year for s in samples))}')
log.info(f'Val:   {len(val_samples):,}  Test: {len(test_samples):,}')

# ── 2. Scaler ─────────────────────────────────────────────────────────────────
log.info('Fitting scaler...')
scaler = AtmosphericScaler()
scaler.fit_transform(np.stack([s.features for s in samples]))
scaler.save(CKPT_DIR / 'scaler.pkl')

# ── 3. Datasets ───────────────────────────────────────────────────────────────
train_ds = TornadoDataset(samples, scaler)
val_ds   = TornadoDataset(val_samples, scaler)
val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0)

def ef_to_intensity_class(ef, label):
    cls = torch.zeros_like(ef, dtype=torch.long)
    cls = torch.where((label > 0.5) & (ef < 3),  torch.ones_like(cls),    cls)
    cls = torch.where((label > 0.5) & (ef >= 3), torch.full_like(cls, 2), cls)
    return cls

# 3-way sampler: oversample EF3+ events 10x relative to their frequency
def make_intensity_sampler(ds, ef3_oversample=10.0):
    intensity_labels = []
    for s in ds.samples:
        if s.label == 0:
            intensity_labels.append(0)
        elif getattr(s, 'ef_scale', 0) >= 3:
            intensity_labels.append(2)
        else:
            intensity_labels.append(1)
    n0 = intensity_labels.count(0)
    n1 = intensity_labels.count(1)
    n2 = intensity_labels.count(2)
    log.info(f'Intensity class counts — None:{n0:,}  Weak:{n1:,}  Significant:{n2:,}')
    w = [1.0/n0 if c==0 else 1.0/n1 if c==1 else ef3_oversample/n2
         for c in intensity_labels]
    return WeightedRandomSampler(w, num_samples=len(w), replacement=True)

balanced_loader = DataLoader(
    train_ds, batch_size=64,
    sampler=make_intensity_sampler(train_ds, ef3_oversample=10.0),
    num_workers=0,
)

# ── 4. Model ──────────────────────────────────────────────────────────────────
cfg = ModelConfig(n_features=N_FEATURES, seq_len=24, d_model=256, n_heads=8,
                  n_encoder_layers=6, d_ff=1024, dropout=0.1, forecast_hours=2,
                  n_intensity_classes=3)
model = TornadoTransformer(cfg).to(DEVICE)
log.info(f'Model params: {sum(p.numel() for p in model.parameters()):,}')

ckpt = torch.load(CKPT_DIR / 'pretrained_best.pt', map_location=DEVICE, weights_only=False)
backbone_state = {k: v for k, v in ckpt['state_dict'].items() if k.startswith('backbone.')}
model.load_state_dict(backbone_state, strict=False)
log.info('Pretrained backbone loaded.')

bce = nn.BCELoss()
intensity_class_weights = torch.tensor([0.10, 0.40, 15.0], device=DEVICE)
ce_loss = nn.CrossEntropyLoss(weight=intensity_class_weights)
INTENSITY_LOSS_WEIGHT = 0.30

def run_epoch(loader, train=True, opt=None):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []
    i_correct, i_total = 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            x = batch['features'].to(DEVICE); y = batch['label'].to(DEVICE)
            ef = batch['ef_scale'].to(DEVICE)
            out    = model(x)
            prob   = out['tornado_prob'].mean(dim=1)
            logits = out['intensity_logits'].mean(dim=1)
            i_cls  = ef_to_intensity_class(ef, y)
            loss   = bce(prob, y) + INTENSITY_LOSS_WEIGHT * ce_loss(logits, i_cls)
            if train and opt:
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            total_loss += loss.item()
            correct    += ((prob > 0.5).float() == y).sum().item()
            total      += len(y)
            all_probs.extend(prob.detach().cpu().tolist())
            all_labels.extend(y.cpu().tolist())
            i_correct  += (logits.argmax(dim=-1) == i_cls).sum().item()
            i_total    += len(i_cls)
    return {'loss': total_loss/len(loader), 'acc': correct/total,
            'probs': np.array(all_probs), 'labels': np.array(all_labels),
            'intensity_acc': i_correct/i_total}

# ── 5. Phase 2a ───────────────────────────────────────────────────────────────
log.info('--- Phase 2a: head only (backbone frozen) ---')
model.freeze_backbone()
opt_head = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
for epoch in range(1, 11):
    tr = run_epoch(balanced_loader, train=True, opt=opt_head)
    vl = run_epoch(val_loader, train=False)
    log.info(f'  2a Epoch {epoch:2d}/10 — train:{tr["loss"]:.4f}  val:{vl["loss"]:.4f}  acc:{vl["acc"]*100:.1f}%')

# ── 6. Phase 2b ───────────────────────────────────────────────────────────────
log.info('--- Phase 2b: progressive unfreeze + 3-class intensity ---')
FINETUNE_EPOCHS = 40
model.unfreeze_backbone(2)
log.info(f'Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
opt_full = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-5, weight_decay=1e-4)
sched    = torch.optim.lr_scheduler.CosineAnnealingLR(opt_full, T_max=FINETUNE_EPOCHS)
best_f1, best_f2 = 0.0, 0.0

for epoch in range(1, FINETUNE_EPOCHS + 1):
    tr = run_epoch(balanced_loader, train=True, opt=opt_full)
    vl = run_epoch(val_loader, train=False)
    sched.step()
    p = vl['probs']; l = vl['labels']
    preds = (p > 0.5).astype(int)
    tp = ((preds==1)&(l==1)).sum(); fp = ((preds==1)&(l==0)).sum(); fn = ((preds==0)&(l==1)).sum()
    prec = tp/(tp+fp+1e-8); rec = tp/(tp+fn+1e-8)
    f1 = 2*prec*rec/(prec+rec+1e-8)
    f2 = 5*prec*rec/(4*prec+rec+1e-8)
    if f1 > best_f1:
        best_f1 = f1
        model.save(CKPT_DIR / 'finetuned_best.pt', {'epoch': epoch, 'f1': f1})
    if f2 > best_f2:
        best_f2 = f2
        model.save(CKPT_DIR / 'finetuned_f2_best.pt', {'epoch': epoch, 'f2': f2})
    if epoch % 5 == 0 or epoch == 1:
        log.info(f'  2b Epoch {epoch:3d}/{FINETUNE_EPOCHS} — loss:{vl["loss"]:.4f}  '
                 f'F1:{f1:.3f}  F2:{f2:.3f}  P:{prec:.3f}  R:{rec:.3f}  '
                 f'IntAcc:{vl["intensity_acc"]*100:.1f}%')

log.info(f'Phase 2b done. Best F1:{best_f1:.4f}  F2:{best_f2:.4f}')

# ── 7. Phase 2c — intensity specialist ───────────────────────────────────────
log.info('--- Phase 2c: intensity specialist (positives only, equal Weak/EF3+) ---')

# Load best Phase 2b checkpoint as starting point
model2c = TornadoTransformer.load(CKPT_DIR / 'finetuned_best.pt', device=str(DEVICE)).to(DEVICE)

# Freeze everything except intensity head
for p in model2c.parameters():
    p.requires_grad = False
for p in model2c.head.intensity_head.parameters():
    p.requires_grad = True
log.info(f'Phase 2c trainable: {sum(p.numel() for p in model2c.parameters() if p.requires_grad):,}')

# Positives-only loader with equal Weak / EF3+ sampling
pos_samples = [s for s in train_ds.samples if s.label == 1]
pos_ds      = TornadoDataset(pos_samples, scaler)
pos_labels  = [2 if getattr(s, 'ef_scale', 0) >= 3 else 1 for s in pos_samples]
n_weak = pos_labels.count(1); n_sig = pos_labels.count(2)
log.info(f'Phase 2c positives — Weak:{n_weak:,}  Significant:{n_sig:,}')
pos_w       = [1.0/n_weak if l==1 else 1.0/n_sig for l in pos_labels]
pos_sampler = WeightedRandomSampler(pos_w, num_samples=len(pos_w), replacement=True)
pos_loader  = DataLoader(pos_ds, batch_size=32, sampler=pos_sampler, num_workers=0)

ce_2c    = nn.CrossEntropyLoss()   # equal sampling handles balance — no extra weighting needed
opt_2c   = torch.optim.AdamW(filter(lambda p: p.requires_grad, model2c.parameters()), lr=1e-4)
sched_2c = torch.optim.lr_scheduler.CosineAnnealingLR(opt_2c, T_max=20)
best_sig_acc = 0.0

for epoch in range(1, 21):
    # Train
    model2c.train()
    for batch in pos_loader:
        x  = batch['features'].to(DEVICE)
        y  = batch['label'].to(DEVICE)
        ef = batch['ef_scale'].to(DEVICE)
        logits = model2c(x)['intensity_logits'].mean(dim=1)
        i_cls  = ef_to_intensity_class(ef, y)
        loss   = ce_2c(logits, i_cls)
        opt_2c.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model2c.parameters(), 1.0)
        opt_2c.step()
    sched_2c.step()

    # Val — measure EF3+ accuracy specifically
    model2c.eval()
    i_preds, i_true = [], []
    with torch.no_grad():
        for batch in val_loader:
            x = batch['features'].to(DEVICE); y = batch['label'].to(DEVICE)
            ef = batch['ef_scale'].to(DEVICE)
            logits = model2c(x)['intensity_logits'].mean(dim=1)
            i_preds.extend(logits.argmax(dim=-1).cpu().tolist())
            i_true.extend(ef_to_intensity_class(ef, y).cpu().tolist())
    ip = np.array(i_preds); it = np.array(i_true)
    sig_acc  = (ip[it==2]==2).mean() if (it==2).any() else 0.0
    weak_acc = (ip[it==1]==1).mean() if (it==1).any() else 0.0
    none_acc = (ip[it==0]==0).mean() if (it==0).any() else 0.0

    if sig_acc > best_sig_acc:
        best_sig_acc = sig_acc
        # Save only the intensity head weights
        torch.save(model2c.head.intensity_head.state_dict(),
                   CKPT_DIR / 'intensity_head_best.pt')

    if epoch % 5 == 0 or epoch == 1:
        log.info(f'  2c Epoch {epoch:2d}/20 — None:{none_acc*100:.1f}%  '
                 f'Weak:{weak_acc*100:.1f}%  Significant:{sig_acc*100:.1f}%')

log.info(f'Phase 2c done. Best EF3+ acc: {best_sig_acc*100:.1f}%')

# ── 8. Merge intensity head into best binary checkpoint ───────────────────────
log.info('Merging Phase 2c intensity head into Phase 2b checkpoint...')
merged = TornadoTransformer.load(CKPT_DIR / 'finetuned_best.pt', device=str(DEVICE)).to(DEVICE)
merged.head.intensity_head.load_state_dict(
    torch.load(CKPT_DIR / 'intensity_head_best.pt', map_location=DEVICE, weights_only=True)
)
merged.save(CKPT_DIR / 'finetuned_best.pt',
            {'phase': '2b+2c', 'f1': best_f1, 'ef3_acc': best_sig_acc})
log.info('Merged checkpoint saved to finetuned_best.pt')
log.info('Done.')
