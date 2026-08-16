import sys, pickle, asyncio, logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

sys.path.insert(0, 'backend')
from dotenv import load_dotenv
load_dotenv('backend/.env')
from ml.data import build_dataset

DATA_DIR   = Path('backend/data')
CACHE_DIR  = DATA_DIR / 'cache'
COMBINED   = CACHE_DIR / 'samples_2015_2025.pkl'
CKPT_2025  = CACHE_DIR / 'checkpoint_2025_2025.pkl'
CACHE_2025 = CACHE_DIR / 'samples_2025_2025.pkl'

# ── 1. Extract existing 2025 samples from combined cache ──────────────────────
log.info('Loading combined cache...')
with open(COMBINED, 'rb') as f:
    all_samples = pickle.load(f)

samples_2025 = [s for s in all_samples if s.valid_time.year == 2025]
pos = sum(s.label for s in samples_2025)
neg = len(samples_2025) - pos
log.info(f'Existing 2025: {len(samples_2025)} total  pos={pos}  neg={neg}  need={pos*3 - neg} more negatives')

# ── 2. Build done_keys so build_dataset skips positive re-fetching ────────────
done_keys = set()
for s in samples_2025:
    if s.label == 1:
        key = f"pos_{s.lat:.3f}_{s.lon:.3f}_{s.valid_time.isoformat()}"
        done_keys.add(key)
log.info(f'Seeding checkpoint with {len(samples_2025)} samples and {len(done_keys)} done_keys')

with open(CKPT_2025, 'wb') as f:
    pickle.dump({'samples': samples_2025, 'done_keys': done_keys}, f)

# ── 3. Run build_dataset — only fetches missing negatives ─────────────────────
log.info('Fetching missing negatives via build_dataset...')
samples_new = asyncio.run(build_dataset(
    years=[2025],
    seq_len=24,
    neg_ratio=3.0,
    cache_dir=CACHE_DIR,
))

new_pos = sum(s.label for s in samples_new)
new_neg = len(samples_new) - new_pos
log.info(f'2025 rebuilt: {len(samples_new)} total  pos={new_pos}  neg={new_neg}  ratio=1:{new_neg//new_pos}')

# ── 4. Rebuild combined cache: replace 2025 slice with new samples ────────────
log.info('Rebuilding combined cache...')
non_2025 = [s for s in all_samples if s.valid_time.year != 2025]
combined_new = sorted(non_2025 + samples_new, key=lambda s: s.valid_time)

year_counts = {}
for s in combined_new:
    yr = s.valid_time.year
    year_counts[yr] = year_counts.get(yr, 0) + 1
log.info(f'Combined cache: {len(combined_new):,} total  years={sorted(year_counts)}')
for yr in sorted(year_counts):
    log.info(f'  {yr}: {year_counts[yr]:,} samples')

with open(COMBINED, 'wb') as f:
    pickle.dump(combined_new, f)
log.info(f'Combined cache saved: {COMBINED}  ({COMBINED.stat().st_size/1e6:.1f} MB)')
