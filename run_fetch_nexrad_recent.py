"""
Fetch NEXRAD radar chips for 2023-2025 SPC events directly from AWS.

For each event in our existing sample cache (2023/2024/2025), this script:
  1. Finds the nearest WSR-88D station
  2. Downloads the closest Level-II file from s3://unidata-nexrad-level2/
  3. Extracts a centered polar radar chip [2, 120, 240] per variable
  4. Saves a paired_samples_YYYY.pkl matching the format expected by run_fusion_training.py

Negatives get a null radar chip (zeros) since there's no event to center on —
the model handles this via the atm-only fallback path.

Run: python run_fetch_nexrad_recent.py --years 2023 2024 2025
"""

import argparse
import asyncio
import logging
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv("backend/.env")
sys.path.insert(0, "backend")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

from ml.nexrad import fetch_radar_chip_for_event, find_nearest_station
from ml.radar import ALL_VARIABLES, PATCH_N_AZ, PATCH_N_RNG

DATA_DIR  = Path("backend/data")
OUT_DIR   = Path("backend/data/fusion_cache")
NEXRAD_CACHE = Path("backend/data/nexrad_cache")
CONCURRENT = 4   # NEXRAD downloads — be conservative, large files


@dataclass
class PairedSample:
    atm_features: np.ndarray
    radar_chip:   dict | None
    label:        int
    ef_scale:     float
    lat:          float
    lon:          float
    valid_time:   object
    tornet_category: str


def null_radar_chip() -> dict:
    """Zero-filled chip for negative samples with no storm center."""
    chip = {v: np.zeros((2, PATCH_N_AZ, PATCH_N_RNG), dtype=np.float32)
            for v in ALL_VARIABLES}
    chip["range_folded_mask"] = np.zeros((2, PATCH_N_AZ, PATCH_N_RNG), dtype=np.float32)
    return chip


async def process_year(year: int):
    out_path = OUT_DIR / f"paired_samples_{year}.pkl"
    if out_path.exists():
        log.info(f"{year}: paired_samples already built, skipping")
        return

    cache_path = DATA_DIR / "cache/samples_2015_2025.pkl"
    if not cache_path.exists():
        log.error(f"Sample cache not found: {cache_path}")
        return

    with open(cache_path, "rb") as f:
        all_samples = pickle.load(f)

    year_samples = [s for s in all_samples if s.valid_time.year == year]
    log.info(f"{year}: {len(year_samples):,} samples "
             f"(pos={sum(s.label for s in year_samples):,})")

    sem = asyncio.Semaphore(CONCURRENT)
    paired: list[PairedSample] = []
    done = 0

    async def process(s):
        nonlocal done
        done += 1
        if done % 200 == 0:
            log.info(f"  {year}: {done}/{len(year_samples)} processed "
                     f"({sum(p.radar_chip is not None and p.label==1 for p in paired)} radar chips)")

        if s.label == 1:
            # Positive: fetch real radar chip centered on event
            nearest = find_nearest_station(s.lat, s.lon)
            if nearest is None:
                log.debug(f"  No NEXRAD within range for ({s.lat:.2f},{s.lon:.2f})")
                chip = null_radar_chip()
            else:
                async with sem:
                    chip_tensors = await fetch_radar_chip_for_event(
                        s.lat, s.lon, s.valid_time,
                        cache_dir=NEXRAD_CACHE / str(year),
                    )
                if chip_tensors is not None:
                    # Convert tensors back to numpy for storage
                    chip = {k: v.squeeze(0).numpy() for k, v in chip_tensors.items()
                            if k != "coordinates"}
                else:
                    chip = null_radar_chip()
        else:
            # Negative: no event center, use null chip
            chip = null_radar_chip()

        paired.append(PairedSample(
            atm_features=s.features,
            radar_chip=chip,
            label=s.label,
            ef_scale=getattr(s, "ef_scale", 0.0),
            lat=s.lat,
            lon=s.lon,
            valid_time=s.valid_time,
            tornet_category="TOR" if s.label == 1 else "NUL",
        ))

    await asyncio.gather(*[process(s) for s in year_samples])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(paired, f)

    real_chips = sum(1 for p in paired if p.label == 1 and p.radar_chip is not None
                     and not np.all(p.radar_chip.get("DBZ", np.zeros(1)) == 0))
    log.info(f"{year}: saved {len(paired):,} samples "
             f"({real_chips:,} with real radar chips) → {out_path}")


async def main(years: list[int]):
    for year in years:
        await process_year(year)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, nargs="+", default=[2023, 2024, 2025])
    args = parser.parse_args()
    asyncio.run(main(args.years))
