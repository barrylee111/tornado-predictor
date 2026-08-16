"""
Build fusion training dataset by pairing TorNet radar chips with ERA5 features.

Steps:
  1. Read TorNet catalog.csv (points to NetCDF radar chips, 2013-2022)
  2. For each sample: parse lat/lon/time from catalog metadata
  3. Fetch 24h ERA5 atmospheric features from Open-Meteo (reuse existing pipeline)
  4. Attach radar chip arrays from NetCDF
  5. Save as paired_samples_YYYY.pkl files

Usage:
  python run_build_fusion_dataset.py --tornet_dir /path/to/tornet_data --years 2020 2021 2022

TorNet dataset download:
  Per-year files (~8-12 GB each) from Zenodo:
  https://zenodo.org/records/10566484  (2013-2022, ~100 GB total)
"""

import argparse
import asyncio
import logging
import pickle
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv("backend/.env")
sys.path.insert(0, "backend")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

from ml.data import fetch_era5_features, _fill_nan_linear, _build_feature_matrix
from ml.radar import ALL_VARIABLES, PATCH_N_AZ, PATCH_N_RNG

CONCURRENT = 8   # ERA5 fetch concurrency


@dataclass
class PairedSample:
    atm_features: np.ndarray   # [24, N_FEATURES]
    radar_chip:   dict         # {DBZ: [2,120,240], VEL: ..., ..., range_folded_mask, coords}
    label:        int          # 1=tornado, 0=non-tornado
    ef_scale:     float        # 0.0 for negatives
    lat:          float
    lon:          float
    valid_time:   datetime
    tornet_category: str       # TOR / NUL / WRN


def read_tornet_chip(nc_path: str) -> dict | None:
    """Read radar chip from TorNet NetCDF file."""
    try:
        ds = nc.Dataset(nc_path)
        chip = {}
        for v in ALL_VARIABLES:
            if v in ds.variables:
                # shape: [time, az, rng, tilt] → take last time frame → [tilt, az, rng]
                arr = np.array(ds.variables[v][:], dtype=np.float32)
                if arr.ndim == 4:
                    arr = arr[-1]        # last time frame [az, rng, tilt]
                    arr = arr.transpose(2, 0, 1)  # → [tilt, az, rng]
                elif arr.ndim == 3:
                    arr = arr.transpose(2, 0, 1)  # [az, rng, tilt] → [tilt, az, rng]
                # Pad/trim to [2, 120, 240]
                n_tilts = min(arr.shape[0], 2)
                out = np.full((2, PATCH_N_AZ, PATCH_N_RNG), np.nan, dtype=np.float32)
                out[:n_tilts, :arr.shape[1], :arr.shape[2]] = arr[:n_tilts,
                                                                  :PATCH_N_AZ,
                                                                  :PATCH_N_RNG]
                chip[v] = out
            else:
                chip[v] = np.full((2, PATCH_N_AZ, PATCH_N_RNG), np.nan, dtype=np.float32)
        # Range-folded mask
        if "range_folded_mask" in ds.variables:
            rfm = np.array(ds.variables["range_folded_mask"][:], dtype=np.float32)
            if rfm.ndim == 4:
                rfm = rfm[-1].transpose(2, 0, 1)
            elif rfm.ndim == 3:
                rfm = rfm.transpose(2, 0, 1)
            out_rfm = np.zeros((2, PATCH_N_AZ, PATCH_N_RNG), dtype=np.float32)
            n = min(rfm.shape[0], 2)
            out_rfm[:n, :rfm.shape[1], :rfm.shape[2]] = rfm[:n, :PATCH_N_AZ, :PATCH_N_RNG]
            chip["range_folded_mask"] = out_rfm
        else:
            chip["range_folded_mask"] = np.zeros((2, PATCH_N_AZ, PATCH_N_RNG), dtype=np.float32)
        ds.close()
        return chip
    except Exception as e:
        log.warning(f"Failed to read {nc_path}: {e}")
        return None


def parse_catalog_metadata(row: pd.Series, tornet_dir: Path) -> tuple | None:
    """Extract (lat, lon, event_time, ef_number, category) from catalog row."""
    nc_path = tornet_dir / row["filename"]
    if not nc_path.exists():
        return None
    try:
        lat = float(row["lat"])
        lon = float(row["lon"])
        ef  = float(row.get("ef_number", -1))
        cat = str(row.get("category", "NUL"))
        event_time = pd.to_datetime(row.get("start_time", row.get("time"))).to_pydatetime()
        if event_time.tzinfo is None:
            event_time = event_time.replace(tzinfo=timezone.utc)
        if np.isnan(lat) or np.isnan(lon):
            return None
        return lat, lon, event_time, max(ef, 0.0), cat
    except Exception as e:
        log.debug(f"Metadata parse error for row {row.get('filename', '?')}: {e}")
        return None


BATCH_SIZE = 2000   # rows per chunk — keeps peak memory ~2.6 GB per batch


async def build_year(year: int, tornet_dir: Path, out_dir: Path, client):
    import httpx
    out_path = out_dir / f"paired_samples_{year}.pkl"
    if out_path.exists():
        log.info(f"{year}: already built, skipping")
        return

    catalog_path = tornet_dir / "catalog.csv"
    if not catalog_path.exists():
        log.error(f"catalog.csv not found in {tornet_dir}")
        return

    catalog = pd.read_csv(catalog_path, parse_dates=["start_time"])
    catalog = catalog[catalog["start_time"].dt.year == year]
    rows = list(catalog.iterrows())
    log.info(f"{year}: {len(rows)} catalog entries")

    out_dir.mkdir(parents=True, exist_ok=True)
    total_saved = total_pos = 0

    for batch_start in range(0, len(rows), BATCH_SIZE):
        batch_rows = rows[batch_start: batch_start + BATCH_SIZE]
        samples: list[PairedSample] = []
        sem = asyncio.Semaphore(CONCURRENT)
        done_in_batch = 0

        async def process_row(row, _idx=batch_start):
            nonlocal done_in_batch
            meta = parse_catalog_metadata(row, tornet_dir)
            if meta is None:
                return
            lat, lon, event_time, ef, cat = meta

            start = event_time - timedelta(hours=24)
            async with sem:
                feats = await fetch_era5_features(lat, lon, start, event_time, client)
            done_in_batch += 1
            global_done = _idx + done_in_batch
            if global_done % 500 == 0:
                log.info(f"  {year}: processed {global_done}/{len(rows)}")

            if feats is None or len(feats) < 6:
                return
            _fill_nan_linear(feats)
            atm_24h = feats[-24:]

            nc_path = str(tornet_dir / row["filename"])
            chip = read_tornet_chip(nc_path)
            if chip is None:
                return

            label = 1 if cat == "TOR" else 0
            samples.append(PairedSample(
                atm_features=atm_24h,
                radar_chip=chip,
                label=label,
                ef_scale=ef if label == 1 else 0.0,
                lat=lat, lon=lon,
                valid_time=event_time,
                tornet_category=cat,
            ))

        tasks = [process_row(row) for _, row in batch_rows]
        await asyncio.gather(*tasks)

        # Append batch to pkl (multiple pickle objects in one file)
        with open(out_path, "ab") as f:
            pickle.dump(samples, f)
        total_saved += len(samples)
        total_pos   += sum(s.label for s in samples)
        log.info(f"  {year}: batch {batch_start//BATCH_SIZE + 1} — "
                 f"{len(samples)} samples written (running total: {total_saved})")

    log.info(f"{year}: saved {total_saved} paired samples (pos={total_pos}) → {out_path}")


async def main(tornet_dir: Path, years: list[int], out_dir: Path):
    import httpx
    async with httpx.AsyncClient() as client:
        for year in years:
            await build_year(year, tornet_dir, out_dir, client)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tornet_dir", type=Path, required=True,
                        help="Directory containing TorNet NetCDF files + catalog.csv")
    parser.add_argument("--years", type=int, nargs="+", default=list(range(2020, 2023)),
                        help="Years to process (default: 2020-2022)")
    parser.add_argument("--out_dir", type=Path, default=Path("backend/data/fusion_cache"))
    args = parser.parse_args()
    asyncio.run(main(args.tornet_dir, args.years, args.out_dir))
