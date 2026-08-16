"""
Phase 2c v3 — intensity specialist with focal loss + aggressive EF3+ sampling.

vs v2 (1:2:1 harmonic sampler):
  - Focal loss (γ=2): down-weights easy examples, forces attention to hard EF3+ cases
  - 20/40/40 batch allocation (None/Weak/EF3+): EF3+ gets 40% of every batch
  - 30 epochs, cosine LR 1e-4 → 1e-6
  - Confusion matrix in final eval for diagnosis
"""

import sys, pickle, logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score
from dotenv import load_dotenv

load_dotenv('backend/.env')
sys.path.insert(0, 'backend')

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

from ml.model import TornadoTransformer, N_FEATURES
from ml.data import AtmosphericScaler, TornadoDataset

DATA_DIR = Path('backend/data')
CKPT_DIR = Path('backend/checkpoints')
DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log.info(f'Device: {DEVICE}')

# ── 1. Load & split ───────────────────────────────────────────────────────────
log.info('Loading cache...')
with open(DATA_DIR / 'cache/samples_2015_2025.pkl', 'rb') as f:
    all_samples = pickle.load(f)

samples      = [s for s in all_samples if s.valid_time.year <= 2023]
val_samples  = [s for s in all_samples if s.valid_time.year == 2024]
test_samples = [s for s in all_samples if s.valid_time.year == 2025]

scaler = AtmosphericScaler.load(CKPT_DIR / 'scaler.pkl')

train_ds = TornadoDataset(samples, scaler)
val_ds   = TornadoDataset(val_samples, scaler)
test_ds  = TornadoDataset(test_samples, scaler)


def ef_to_intensity_class(ef: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    cls = torch.zeros_like(ef, dtype=torch.long)
    cls = torch.where((label > 0.5) & (ef < 3),  torch.ones_like(cls),    cls)
    cls = torch.where((label > 0.5) & (ef >= 3), torch.full_like(cls, 2), cls)
    return cls


# ── 2. Sampler with explicit batch fractions ──────────────────────────────────
def make_sampler(ds, frac_none: float, frac_weak: float, frac_sig: float):
    labels = []
    for s in ds.samples:
        if s.label == 0:
            labels.append(0)
        elif getattr(s, 'ef_scale', 0) >= 3:
            labels.append(2)
        else:
            labels.append(1)
    n0 = labels.count(0); n1 = labels.count(1); n2 = labels.count(2)
    log.info(f'Train — None:{n0:,}  Weak:{n1:,}  EF3+:{n2:,}')
    log.info(f'Batch slots — None:{frac_none*100:.0f}%  Weak:{frac_weak*100:.0f}%  EF3+:{frac_sig*100:.0f}%')
    w = [frac_none/n0 if c == 0 else frac_weak/n1 if c == 1 else frac_sig/n2
         for c in labels]
    return WeightedRandomSampler(w, num_samples=len(w), replacement=True)


train_loader = DataLoader(
    train_ds, batch_size=64,
    sampler=make_sampler(train_ds, frac_none=0.20, frac_weak=0.40, frac_sig=0.40),
    num_workers=0,
)
val_loader  = DataLoader(val_ds,  batch_size=128, shuffle=False, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)


# ── 3. Focal loss ─────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma * ce).mean()


focal = FocalLoss(gamma=2.0)


# ── 4. Load Phase 2b checkpoint, freeze backbone ──────────────────────────────
log.info('Loading Phase 2b checkpoint...')
model = TornadoTransformer.load(CKPT_DIR / 'finetuned_best.pt', device=str(DEVICE)).to(DEVICE)

for p in model.parameters():
    p.requires_grad = False
for p in model.head.intensity_head.parameters():
    p.requires_grad = True
log.info(f'Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

opt   = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-4
)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30, eta_min=1e-6)
EPOCHS = 30


# ── 5. Eval helper ────────────────────────────────────────────────────────────
def evaluate(m: TornadoTransformer, loader: DataLoader):
    m.eval()
    i_preds, i_true, probs_all, labels_all = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            x  = batch['features'].to(DEVICE)
            y  = batch['label'].to(DEVICE)
            ef = batch['ef_scale'].to(DEVICE)
            out    = m(x)
            prob   = out['tornado_prob'].mean(dim=1)
            logits = out['intensity_logits'].mean(dim=1)
            i_preds.extend(logits.argmax(dim=-1).cpu().tolist())
            i_true.extend(ef_to_intensity_class(ef, y).cpu().tolist())
            probs_all.extend(prob.detach().cpu().tolist())
            labels_all.extend(y.cpu().tolist())

    ip = np.array(i_preds); it = np.array(i_true)
    p  = np.array(probs_all); l  = np.array(labels_all)

    none_acc = float((ip[it == 0] == 0).mean()) if (it == 0).any() else 0.0
    weak_acc = float((ip[it == 1] == 1).mean()) if (it == 1).any() else 0.0
    sig_acc  = float((ip[it == 2] == 2).mean()) if (it == 2).any() else 0.0
    hmean = 3.0 / (1.0/(none_acc + 1e-8) + 1.0/(weak_acc + 1e-8) + 1.0/(sig_acc + 1e-8))

    auc   = roc_auc_score(l, p) if len(set(l)) > 1 else 0.5
    preds_bin = (p > 0.5).astype(int)
    tp = int(((preds_bin == 1) & (l == 1)).sum())
    fp = int(((preds_bin == 1) & (l == 0)).sum())
    fn = int(((preds_bin == 0) & (l == 1)).sum())
    pod = tp / (tp + fn + 1e-8)
    far = fp / (tp + fp + 1e-8)
    f1  = 2*tp / (2*tp + fp + fn + 1e-8)

    conf = np.zeros((3, 3), dtype=int)
    for t, pd in zip(it.tolist(), ip.tolist()):
        conf[t, pd] += 1

    return {
        'none': none_acc, 'weak': weak_acc, 'sig': sig_acc, 'hmean': hmean,
        'auc': auc, 'pod': pod, 'far': far, 'f1': f1, 'conf': conf,
    }


# ── 6. Training loop ──────────────────────────────────────────────────────────
log.info('--- Phase 2c v3: focal loss + 20/40/40 sampling ---')
best_hmean = 0.0

for epoch in range(1, EPOCHS + 1):
    model.train()
    for batch in train_loader:
        x  = batch['features'].to(DEVICE)
        y  = batch['label'].to(DEVICE)
        ef = batch['ef_scale'].to(DEVICE)
        logits = model(x)['intensity_logits'].mean(dim=1)
        i_cls  = ef_to_intensity_class(ef, y)
        loss   = focal(logits, i_cls)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    sched.step()

    r = evaluate(model, val_loader)
    if r['hmean'] > best_hmean:
        best_hmean = r['hmean']
        torch.save(model.head.intensity_head.state_dict(),
                   CKPT_DIR / 'intensity_head_v3.pt')
        best_epoch = epoch

    if epoch % 5 == 0 or epoch == 1:
        log.info(f'  Epoch {epoch:2d}/{EPOCHS} — None:{r["none"]*100:.1f}%  '
                 f'Weak:{r["weak"]*100:.1f}%  EF3+:{r["sig"]*100:.1f}%  '
                 f'Harmonic:{r["hmean"]*100:.1f}%')

log.info(f'Best harmonic: {best_hmean*100:.1f}% at epoch {best_epoch}')

# ── 7. Merge + final eval ─────────────────────────────────────────────────────
log.info('Merging best intensity head into finetuned_best.pt...')
merged = TornadoTransformer.load(CKPT_DIR / 'finetuned_best.pt', device=str(DEVICE)).to(DEVICE)
merged.head.intensity_head.load_state_dict(
    torch.load(CKPT_DIR / 'intensity_head_v3.pt', map_location=DEVICE, weights_only=True)
)
merged.save(CKPT_DIR / 'finetuned_best.pt',
            {'phase': '2b+2c_v3', 'harmonic': best_hmean})
log.info('Merged and saved.')

log.info('--- Final Evaluation ---')
for name, loader in [('VAL 2024', val_loader), ('TEST 2025', test_loader)]:
    r = evaluate(merged, loader)
    log.info(f'{name}: AUC:{r["auc"]:.4f}  POD:{r["pod"]*100:.1f}%  '
             f'FAR:{r["far"]*100:.1f}%  F1:{r["f1"]*100:.1f}%')
    log.info(f'  Intensity — None:{r["none"]*100:.1f}%  Weak:{r["weak"]*100:.1f}%  '
             f'EF3+:{r["sig"]*100:.1f}%  Harmonic:{r["hmean"]*100:.1f}%')
    c = r['conf']
    log.info(f'  Confusion (row=true, col=pred) [None / Weak / EF3+]:')
    log.info(f'    None → {c[0].tolist()}')
    log.info(f'    Weak → {c[1].tolist()}')
    log.info(f'    EF3+ → {c[2].tolist()}')
