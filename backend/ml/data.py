"""
Data pipeline for TornadoTransformer.

Sources:
  - SPC storm reports: https://www.spc.noaa.gov/wcm/#data  (tornado labels)
  - Open-Meteo historical ERA5: hourly atmospheric reanalysis (features)

The pipeline pairs each tornado event (lat/lon/time from SPC) with 24h of
ERA5 atmospheric data leading up to it, labeled positive. Negative samples
are drawn from the same geographic region on non-event days.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

import httpx
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import pickle

from ml.model import FEATURE_NAMES, N_FEATURES

logger = logging.getLogger(__name__)

import os

SPC_TORNADO_URL = "https://www.spc.noaa.gov/wcm/data/{year}_torn.csv"

# Open-Meteo customer API — paid tier, no rate limits, full variable set.
_API_KEY = os.environ.get("OPEN_METEO_API_KEY", "")
# ERA5 reanalysis archive: covers 1940-present, works for TorNet 2013+ data
OPEN_METEO_HIST_URL = "https://customer-archive-api.open-meteo.com/v1/archive"

OPEN_METEO_VARS = [
    "cape",
    "convective_inhibition",
    "surface_pressure",
    "temperature_2m",
    "dewpoint_2m",
    "windspeed_10m",
    "winddirection_10m",
    "wind_speed_500hPa",
    "wind_direction_500hPa",
    "temperature_500hPa",
    "precipitation",
    "lifted_index",
]


@dataclass
class Sample:
    features: np.ndarray  # (seq_len, N_FEATURES) float32
    label: int            # 1 = tornado within next 2h, 0 = no tornado
    forecast_hour: int    # which hour ahead the label refers to (1 or 2)
    lat: float
    lon: float
    valid_time: datetime
    ef_scale: float = 0.0  # EF scale 0-5 for positives, 0 for negatives


class AtmosphericScaler:
    """Fit once on training set, persist for inference."""

    def __init__(self):
        self._scaler = StandardScaler()
        self._fitted = False

    def fit(self, data: np.ndarray):
        flat = data.reshape(-1, N_FEATURES)
        self._scaler.fit(flat)
        self._fitted = True

    def transform(self, data: np.ndarray) -> np.ndarray:
        shape = data.shape
        return self._scaler.transform(data.reshape(-1, N_FEATURES)).reshape(shape)

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        self.fit(data)
        return self.transform(data)

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump(self._scaler, f)

    @classmethod
    def load(cls, path: Path) -> "AtmosphericScaler":
        obj = cls()
        with open(path, "rb") as f:
            obj._scaler = pickle.load(f)
        obj._fitted = True
        return obj


HEADERS = {"User-Agent": "TornadoWatch/0.1 (research project; contact: research@example.com)"}


async def fetch_spc_tornado_reports(year: int, client: httpx.AsyncClient) -> pd.DataFrame:
    url = SPC_TORNADO_URL.format(year=year)
    r = await client.get(url, timeout=30, headers=HEADERS, follow_redirects=True)
    r.raise_for_status()
    from io import StringIO
    df = pd.read_csv(StringIO(r.text), encoding="latin-1")
    # SPC columns: yr, mo, dy, time, tz, st, stf, stn, mag, inj, fat, loss,
    #              closs, slat, slon, elat, elon, len, wid, ns, sn, sg, f1..f4
    df = df.rename(columns={"slat": "lat", "slon": "lon", "mag": "ef_scale"})
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["ef_scale"] = pd.to_numeric(df.get("ef_scale", df.get("f", 0)), errors="coerce").fillna(0)
    df = df.dropna(subset=["lat", "lon"])
    df = df[(df["lat"] != 0) & (df["lon"] != 0)]
    # SPC time column is HH:MM:SS — combine with date column directly
    df["datetime"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["time"].astype(str),
        errors="coerce", utc=True
    )
    return df[["datetime", "lat", "lon", "ef_scale"]].dropna()


async def fetch_era5_features(
    lat: float,
    lon: float,
    start: datetime,
    end: datetime,
    client: httpx.AsyncClient,
) -> np.ndarray | None:
    """
    Fetch historical atmospheric data from Open-Meteo customer API.
    Returns (hours, N_FEATURES) float32 array covering [start, end], or None on failure.
    """
    params = {
        "latitude": round(lat, 2),
        "longitude": round(lon, 2),
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "hourly": ",".join(OPEN_METEO_VARS),
        "timezone": "UTC",
        "apikey": _API_KEY,
    }
    data = None
    for attempt, delay in enumerate([0] + ARCHIVE_RETRY_DELAYS):
        if delay:
            logger.warning(f"Open-Meteo error — waiting {delay}s (retry {attempt}/{len(ARCHIVE_RETRY_DELAYS)})")
            await asyncio.sleep(delay)
        try:
            r = await client.get(OPEN_METEO_HIST_URL, params=params, timeout=30)
            if r.status_code == 429:
                continue
            r.raise_for_status()
            data = r.json()
            await asyncio.sleep(ARCHIVE_REQUEST_INTERVAL)
            break
        except Exception as e:
            logger.warning(f"Open-Meteo fetch failed for ({lat},{lon}): {e}")
            return None
    if data is None:
        logger.error(f"Open-Meteo fetch exhausted retries for ({lat},{lon})")
        return None

    hourly = data.get("hourly", {})
    times = pd.to_datetime(hourly.get("time", []), utc=True)
    mask = (times >= start) & (times <= end)

    raw = {}
    for var in OPEN_METEO_VARS:
        vals = np.array(hourly.get(var, []), dtype=np.float32)
        raw[var] = vals[mask] if len(vals) == len(times) else np.full(mask.sum(), np.nan)

    features = _build_feature_matrix(raw)
    _fill_nan_linear(features)
    return features


def _build_feature_matrix(raw: dict) -> np.ndarray:
    n = len(raw.get("cape", []))
    mat = np.full((n, N_FEATURES), np.nan, dtype=np.float32)

    def get(key, default=np.nan):
        return raw.get(key, np.full(n, default))

    cape  = get("cape")
    mat[:, 0] = np.where(cape > 0, cape, 0)
    mat[:, 1] = np.abs(get("convective_inhibition"))
    ws10  = get("windspeed_10m")
    wd10  = np.deg2rad(get("winddirection_10m"))
    ws500 = get("wind_speed_500hPa")
    wd500 = np.deg2rad(get("wind_direction_500hPa"))
    u10   = -ws10  * np.sin(wd10)
    v10   = -ws10  * np.cos(wd10)
    u500  = -ws500 * np.sin(wd500)
    v500  = -ws500 * np.cos(wd500)
    shear_mag = np.sqrt((u500 - u10)**2 + (v500 - v10)**2)
    mat[:, 2]  = shear_mag * np.abs(u10)
    mat[:, 3]  = shear_mag * np.abs(v10)
    mat[:, 4]  = shear_mag
    mat[:, 5]  = np.sqrt(u10**2 + v10**2)
    mat[:, 6]  = np.full(n, np.nan)
    mat[:, 7]  = np.full(n, np.nan)
    mat[:, 8]  = np.full(n, 20.0)
    mat[:, 9]  = get("temperature_2m")
    mat[:, 10] = get("dewpoint_2m")
    mat[:, 11] = u10
    mat[:, 12] = v10
    mat[:, 13] = u500
    mat[:, 14] = v500
    mat[:, 15] = get("temperature_500hPa")
    mat[:, 16] = get("surface_pressure")
    mat[:, 17] = get("lifted_index")
    mat[:, 18] = np.full(n, np.nan)
    mat[:, 19] = np.full(n, np.nan)

    return mat


def _fill_nan_linear(mat: np.ndarray):
    for col in range(mat.shape[1]):
        series = mat[:, col]
        nans = np.isnan(series)
        if nans.all():
            mat[:, col] = 0.0
        elif nans.any():
            idx = np.arange(len(series))
            mat[:, col] = np.interp(idx, idx[~nans], series[~nans])


class TornadoDataset(Dataset):
    def __init__(self, samples: list[Sample], scaler: AtmosphericScaler | None = None):
        self.samples = samples
        self.scaler = scaler

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s = self.samples[idx]
        feat = s.features.copy()
        if self.scaler:
            feat = self.scaler.transform(feat)
        return {
            "features": torch.tensor(feat, dtype=torch.float32),
            "label": torch.tensor(s.label, dtype=torch.float32),
            "ef_scale": torch.tensor(getattr(s, 'ef_scale', 0.0), dtype=torch.float32),
            "forecast_hour": torch.tensor(s.forecast_hour, dtype=torch.long),
        }

    def split(self, val_frac: float = 0.1) -> tuple["TornadoDataset", "TornadoDataset"]:
        n = len(self.samples)
        n_val = max(1, int(n * val_frac))
        # Time-ordered split — no future leakage
        return (
            TornadoDataset(self.samples[:-n_val], self.scaler),
            TornadoDataset(self.samples[-n_val:], self.scaler),
        )


CONCURRENT_REQUESTS = 25        # grid prediction endpoint (forecast API — tolerant)
ARCHIVE_CONCURRENT = 5         # NASA POWER — no daily quota, handles concurrency well
ARCHIVE_REQUEST_INTERVAL = 0.2  # seconds between requests
ARCHIVE_RETRY_DELAYS = [15, 45, 90]  # seconds to wait on consecutive errors


def _load_checkpoint(path: Path) -> tuple[list[Sample], set[str]]:
    if not path.exists():
        return [], set()
    with open(path, "rb") as f:
        data = pickle.load(f)
    samples = data.get("samples", [])
    done_keys = data.get("done_keys", set())
    logger.info(f"Resumed from checkpoint: {len(samples)} samples, {len(done_keys)} completed fetches")
    return samples, done_keys


def _save_checkpoint(path: Path, samples: list[Sample], done_keys: set[str]):
    with open(path, "wb") as f:
        pickle.dump({"samples": samples, "done_keys": done_keys}, f)


async def build_dataset(
    years: list[int],
    seq_len: int = 24,
    neg_ratio: float = 3.0,
    cache_dir: Path = Path("data/cache"),
) -> list[Sample]:
    """
    Download SPC reports + ERA5 features and build labeled samples.
    Checkpoints after every year so progress survives interruption.
    Final cache written on completion; checkpoint deleted.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"samples_{min(years)}_{max(years)}.pkl"
    checkpoint_path = cache_dir / f"checkpoint_{min(years)}_{max(years)}.pkl"

    if cache_path.exists():
        logger.info(f"Loading cached dataset from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    samples, done_keys = _load_checkpoint(checkpoint_path)
    sem = asyncio.Semaphore(ARCHIVE_CONCURRENT)

    async def fetch_positive(row, client: httpx.AsyncClient) -> list[Sample]:
        event_time: datetime = row["datetime"]
        lat, lon = float(row["lat"]), float(row["lon"])
        key = f"pos_{lat:.3f}_{lon:.3f}_{event_time.isoformat()}"
        if key in done_keys:
            return []
        start = event_time - timedelta(hours=seq_len)
        async with sem:
            feats = await fetch_era5_features(lat, lon, start, event_time, client)
        done_keys.add(key)
        if feats is None or len(feats) < seq_len:
            return []
        feats = feats[-seq_len:]
        ef = float(row.get("ef_scale", 0) or 0)
        return [Sample(feats, 1, fh, lat, lon, event_time, ef) for fh in [1, 2]]

    async def fetch_negative(lat: float, lon: float, neg_time: datetime, client: httpx.AsyncClient) -> list[Sample]:
        key = f"neg_{lat:.3f}_{lon:.3f}_{neg_time.isoformat()}"
        if key in done_keys:
            return []
        start = neg_time - timedelta(hours=seq_len)
        async with sem:
            feats = await fetch_era5_features(lat, lon, start, neg_time, client)
        done_keys.add(key)
        if feats is None or len(feats) < seq_len:
            return []
        feats = feats[-seq_len:]
        return [Sample(feats, 0, fh, lat, lon, neg_time) for fh in [1, 2]]

    async with httpx.AsyncClient() as client:
        for year in years:
            logger.info(f"Processing year {year}")
            try:
                reports = await fetch_spc_tornado_reports(year, client)
            except Exception as e:
                logger.error(f"Failed to fetch SPC reports for {year}: {e}")
                continue

            pos_tasks = [fetch_positive(row, client) for _, row in reports.iterrows()]
            pos_results = await asyncio.gather(*pos_tasks)
            for result in pos_results:
                samples.extend(result)

            # Build negative sample coordinates for the year.
            # Sample from ALL of CONUS (not just Tornado Alley) so the model sees
            # truly tornado-sparse regions (Pacific NW, Rockies, New England).
            # Exclude any date/location within 1 degree of a real event to avoid
            # mislabeling near-misses as 0.
            pos_count = len([s for s in samples if s.label == 1])
            neg_target = int(pos_count * neg_ratio) - len([s for s in samples if s.label == 0])
            rng = np.random.default_rng(year)

            event_keys: set[tuple] = set()
            for _, row in reports.iterrows():
                t = row["datetime"]
                for dlat in [-1, 0, 1]:
                    for dlon in [-1, 0, 1]:
                        event_keys.add((
                            round(float(row["lat"])) + dlat,
                            round(float(row["lon"])) + dlon,
                            t.strftime("%Y-%m-%d"),
                        ))

            neg_coords = []
            attempts = 0
            while len(neg_coords) < max(0, neg_target) and attempts < neg_target * 5:
                attempts += 1
                lat_neg = float(rng.uniform(25, 50))
                lon_neg = float(rng.uniform(-125, -65))
                days_offset = int(rng.integers(0, 365))
                neg_time = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(
                    days=days_offset, hours=int(rng.integers(12, 22))
                )
                key = (round(lat_neg), round(lon_neg), neg_time.strftime("%Y-%m-%d"))
                if key in event_keys:
                    continue
                neg_coords.append((lat_neg, lon_neg, neg_time))

            neg_tasks = [fetch_negative(la, lo, t, client) for la, lo, t in neg_coords]
            neg_results = await asyncio.gather(*neg_tasks)
            for result in neg_results:
                samples.extend(result)

            logger.info(f"Year {year} complete — {len(samples)} total samples so far")
            _save_checkpoint(checkpoint_path, samples, done_keys)
            logger.info(f"Checkpoint saved after year {year}")

    samples.sort(key=lambda s: s.valid_time)
    with open(cache_path, "wb") as f:
        pickle.dump(samples, f)
    checkpoint_path.unlink(missing_ok=True)
    logger.info(f"Dataset built: {len(samples)} samples, cached to {cache_path}")
    return samples
