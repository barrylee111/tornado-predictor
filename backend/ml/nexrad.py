"""
NEXRAD Level-II fetching and preprocessing for real-time tornado inference.

Pipeline for a given (lat, lon, time):
  1. Find nearest WSR-88D station within MAX_RADAR_RANGE_KM
  2. Discover the closest Level-II file on AWS S3
  3. Read with pyart, extract DBZ/VEL/KDP/RHOHV/ZDR/WIDTH at 2 lowest tilts
  4. Build centered radar chip [2, 120, 240] per variable
  5. Return dict matching RadarBackbone input format

AWS bucket: s3://unidata-nexrad-level2/YYYY/MM/DD/KXXX/
"""

from __future__ import annotations
import asyncio
import logging
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

MAX_RADAR_RANGE_KM = 230.0   # WSR-88D max unambiguous range
PATCH_N_AZ  = 120            # azimuth gates in output chip
PATCH_N_RNG = 240            # range gates in output chip (240 × 250m = 60km)
RANGE_GATE_M = 250.0         # nominal NEXRAD super-res range spacing
NEXRAD_BUCKET = "unidata-nexrad-level2"

# Target elevation tilts (degrees) — use 2 lowest available
TARGET_TILTS_DEG = (0.5, 1.45)


# ── WSR-88D station registry ──────────────────────────────────────────────────
# Format: ICAO_ID → (lat, lon, elevation_m)
# Source: NOAA/NWS NEXRAD network (CONUS + territories)
WSR88D_STATIONS: Dict[str, Tuple[float, float, float]] = {
    "KABR": (45.4558, -98.4132, 397), "KABX": (35.1497, -106.8239, 1789),
    "KAKQ": (36.9839, -77.0075,  34), "KAMA": (35.2333, -101.7092, 1093),
    "KAMX": (25.6111, -80.4128,   4), "KAPX": (44.9072, -84.7197, 446),
    "KARX": (43.8228, -91.1911, 390), "KATX": (48.1945,-122.4958, 150),
    "KBBX": (39.4957,-121.6317, 173), "KBGM": (42.1997, -75.9847, 490),
    "KBHX": (40.4986,-124.2919, 734), "KBIS": (46.7708,-100.7603, 505),
    "KBLX": (45.8537,-108.6068, 1097),"KBMX": (33.1722, -86.7697, 197),
    "KBOX": (41.9558, -71.1369,  31), "KBRO": (25.9161, -97.4189,   7),
    "KBUF": (42.9489, -78.7369, 216), "KBYX": (24.5975, -81.7033,   4),
    "KCAE": (33.9487, -81.1183,  70), "KCBW": (46.0392, -67.8067, 229),
    "KCBX": (43.4911,-116.2356, 933), "KCCX": (40.9228, -78.0039, 733),
    "KCLE": (41.4131, -81.8597, 247), "KCLX": (32.6553, -81.0422,  99),
    "KCRP": (27.7839, -97.5111,  14), "KCXX": (44.5111, -73.1661, 97),
    "KCYS": (41.1519,-104.8061, 1867),"KDAX": (38.5011,-121.6778,   9),
    "KDDC": (37.7608, -99.9689, 790), "KDFX": (29.2731,-100.2803, 344),
    "KDGX": (32.2797, -89.9844, 149), "KDIX": (39.9469, -74.4108,  45),
    "KDLH": (46.8369, -92.2097, 435), "KDMX": (41.7311, -93.7228, 299),
    "KDOX": (38.8256, -75.4400,  15), "KDTX": (42.6997, -83.4717, 324),
    "KDVN": (41.6117, -90.5808, 230), "KDYX": (32.5381, -99.2542, 462),
    "KEAX": (38.8103, -94.2644, 303), "KEMX": (31.8936,-110.6303, 1586),
    "KENX": (42.5864, -74.0639, 561), "KEOX": (31.4603, -85.4594, 131),
    "KEPZ": (31.8731,-106.6981, 1251),"KESX": (35.7011,-114.8914, 1484),
    "KEVX": (30.5644, -85.9217,  49), "KEWX": (29.7039, -98.0281, 193),
    "KEYX": (35.0979,-117.5608, 840), "KFCX": (37.0242, -80.2744, 874),
    "KFDR": (34.3622, -98.9764, 386), "KFDX": (34.6344,-103.6294, 1417),
    "KFFC": (33.3633, -84.5658, 262), "KFSD": (43.5878, -96.7292, 436),
    "KFSX": (34.5744,-111.1983, 2261),"KFTG": (39.7867,-104.5458, 1675),
    "KFWS": (32.5728, -97.3031, 208), "KGGW": (48.2064,-106.6250, 696),
    "KGJX": (39.0622,-108.2139, 3044),"KGLD": (39.3669,-101.7003, 1113),
    "KGRB": (44.4986, -88.1111, 208), "KGRK": (30.7217, -97.3828, 166),
    "KGRR": (42.8939, -85.5447, 236), "KGSP": (34.8833, -82.2197, 286),
    "KGTF": (47.4597,-111.3856, 1116),"KGWX": (33.8967, -88.3294, 145),
    "KGYX": (43.8914, -70.2561, 128), "KHDX": (33.0767,-106.1228, 1286),
    "KHGX": (29.4719, -95.0792,   9), "KHNX": (36.3144,-119.6319,  74),
    "KHPX": (36.7369, -87.2850, 164), "KHTX": (34.9306, -86.0839, 536),
    "KICT": (37.6544, -97.4428, 407), "KICX": (37.5908,-112.8628, 3284),
    "KILN": (39.4203, -83.8217, 320), "KILX": (40.1506, -89.3367, 177),
    "KIND": (39.7075, -86.2803, 241), "KINX": (36.1750, -95.5644, 204),
    "KIWX": (41.3589, -85.7000, 292), "KJAN": (32.3178, -90.0797,  90),
    "KJAX": (30.4847, -81.7019,   9), "KJGX": (32.6753, -83.3511, 159),
    "KJKL": (37.5908, -83.3131, 416), "KLBB": (33.6542,-101.8142, 990),
    "KLCH": (30.1253, -93.2158,   6), "KLIX": (30.3367, -89.8253,   7),
    "KLNX": (41.9578,-100.5761, 916), "KLOT": (41.6044, -88.0847, 202),
    "KLRX": (40.7397,-116.8028, 2056),"KLSX": (38.6989, -90.6828, 185),
    "KLTX": (33.9889, -78.4292,   9), "KLVX": (37.9753, -85.9439, 219),
    "KLWX": (38.9753, -77.4778,  83), "KLZK": (34.8364, -92.2619, 173),
    "KMAF": (31.9433,-102.1894, 875), "KMAX": (42.0811,-122.7175, 2290),
    "KMBX": (48.3928,-100.8647, 455), "KMHX": (34.7761, -76.8764,   9),
    "KMKX": (42.9678, -88.5506, 293), "KMLB": (28.1133, -80.6544,  11),
    "KMOB": (30.6797, -88.2397,  63), "KMPX": (44.8489, -93.5653, 288),
    "KMQT": (46.5311, -87.5483, 430), "KMRX": (36.1686, -83.4017, 400),
    "KMSX": (47.0411,-113.9861, 2393),"KMTX": (41.2628,-112.4478, 1969),
    "KMUX": (37.1553,-121.8983, 1057),"KMVX": (47.5278, -97.3253, 300),
    "KMXX": (32.5367, -85.7897, 126), "KNKX": (32.9189,-117.0419, 293),
    "KNQA": (35.3447, -89.8733, 104), "KOAX": (41.3203, -96.3664, 350),
    "KOHX": (36.2472, -86.5625, 176), "KOKX": (40.8656, -72.8639,  26),
    "KOTX": (47.6806,-117.6267, 726), "KPAH": (37.0686, -88.7719, 119),
    "KPBZ": (40.5317, -80.2181, 361), "KPDT": (45.6906,-118.8531, 462),
    "KPOE": (31.1553, -92.9761,  98), "KPUX": (38.4597,-104.1814, 1624),
    "KRAX": (35.6656, -78.4900, 106), "KRGX": (39.7542,-119.4611, 2530),
    "KRIW": (43.0661,-108.4772, 1697),"KRLX": (38.3111, -81.7228, 326),
    "KRMX": (43.4678, -75.4578, 515), "KRTX": (45.7150,-122.9650, 479),
    "KSFX": (43.1056,-112.6861, 1364),"KSGF": (37.2353, -93.4006, 390),
    "KSHV": (32.4508, -93.8411,  83), "KSIX": (34.1736,-116.1661, 1878),
    "KSLC": (40.7197,-111.8356, 1288),"KSOX": (33.8178,-117.6358, 946),
    "KSRX": (35.2906, -94.3619, 194), "KTBW": (27.7056, -82.4017,  15),
    "KTFX": (47.4597,-111.3856, 1116),"KTLH": (30.3975, -84.3289,  19),
    "KTLX": (35.3331, -97.2778, 370), "KTOP": (38.9972, -95.6644, 317),
    "KTWX": (38.9969, -96.2322, 417), "KUDX": (44.1253,-102.8297, 920),
    "KUEX": (40.3208, -98.4419, 602), "KVAX": (30.8903, -83.0019,  52),
    "KVEL": (40.9256,-112.0606, 1944),"KVNX": (36.7406, -98.1283, 373),
    "KVTX": (34.4117,-119.1794, 831), "KVWX": (38.2603, -87.7247, 144),
    "KYUX": (32.4953,-114.6558,  53), "PABC": (60.7928,-161.8761,  47),
    "PACG": (56.8528,-135.5258,  68), "PAEC": (64.5114,-165.2950,  20),
    "PAHG": (60.6119,-151.3508,  61), "PAIH": (59.4619,-146.3011,   6),
    "PAKC": (58.6794,-156.6294,  18), "PAPD": (65.0361,-147.5014, 790),
    "PHKI": (21.8939,-159.5522,  68), "PHKM": (20.1256,-155.7781, 1167),
    "PHMO": (21.1328,-157.1797, 419), "PHWA": (19.0950,-155.5689,  417),
    "RKJK": (35.9242, 126.6222, 23),  "RKSG": (36.9503, 127.0197,  52),
    "TJUA": (18.1156, -66.0781, 865),
}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (math.sin(d_lat/2)**2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lon/2)**2)
    return R * 2 * math.asin(math.sqrt(a))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """True north bearing from (lat1,lon1) to (lat2,lon2) in [0, 360)."""
    d_lon = math.radians(lon2 - lon1)
    lat1r = math.radians(lat1); lat2r = math.radians(lat2)
    x = math.sin(d_lon) * math.cos(lat2r)
    y = math.cos(lat1r)*math.sin(lat2r) - math.sin(lat1r)*math.cos(lat2r)*math.cos(d_lon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def find_nearest_station(lat: float, lon: float) -> Optional[Tuple[str, float]]:
    """Return (station_id, distance_km) for the nearest WSR-88D within range."""
    best_id, best_dist = None, float("inf")
    for sid, (slat, slon, _) in WSR88D_STATIONS.items():
        d = haversine_km(lat, lon, slat, slon)
        if d < best_dist and d <= MAX_RADAR_RANGE_KM:
            best_id, best_dist = sid, d
    return (best_id, best_dist) if best_id else None


# ── AWS S3 file discovery ─────────────────────────────────────────────────────
def _s3_prefix(station: str, dt: datetime) -> str:
    return f"{dt.year}/{dt.month:02d}/{dt.day:02d}/{station}/"


async def find_nexrad_file(
    station: str,
    target_time: datetime,
    window_minutes: int = 10,
) -> Optional[str]:
    """
    Return S3 key of the NEXRAD Level-II file closest to target_time.
    Uses anonymous S3 access (public bucket).
    """
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config
        s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
        prefix = _s3_prefix(station, target_time)
        resp = s3.list_objects_v2(Bucket=NEXRAD_BUCKET, Prefix=prefix)
        if "Contents" not in resp:
            return None
        keys = [obj["Key"] for obj in resp["Contents"] if obj["Key"].endswith(".ar2v")]
        if not keys:
            return None
        # Parse timestamp from filename: KXXX_YYYYMMDD_HHMMSS_VNN.ar2v
        def parse_dt(key: str) -> Optional[datetime]:
            fname = key.split("/")[-1]
            parts = fname.split("_")
            if len(parts) < 3:
                return None
            try:
                return datetime.strptime(parts[1] + parts[2], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
            except ValueError:
                return None
        best_key, best_delta = None, timedelta(minutes=window_minutes + 1)
        for key in keys:
            dt = parse_dt(key)
            if dt is None:
                continue
            delta = abs(dt - target_time)
            if delta < best_delta:
                best_delta, best_key = delta, key
        return best_key
    except Exception as e:
        logger.error(f"S3 lookup failed for {station} at {target_time}: {e}")
        return None


# ── NEXRAD file reader ────────────────────────────────────────────────────────
RADAR_VAR_MAP = {
    "DBZ":   ["reflectivity", "DBZ", "REF"],
    "VEL":   ["velocity", "VEL", "RAD_VEL"],
    "ZDR":   ["differential_reflectivity", "ZDR"],
    "RHOHV": ["cross_correlation_ratio", "RHO", "RHOHV"],
    "KDP":   ["specific_differential_phase", "KDP"],
    "WIDTH": ["spectrum_width", "SW"],
}


def _read_field(radar, field_candidates: list, tilt_idx: int) -> Optional[np.ndarray]:
    for name in field_candidates:
        if name in radar.fields:
            return radar.get_field(tilt_idx, name).filled(np.nan)
    return None


def _find_tilt(radar, target_deg: float) -> int:
    """Return tilt index with elevation closest to target_deg."""
    elevs = radar.fixed_angle["data"]
    return int(np.argmin(np.abs(elevs - target_deg)))


def _centered_patch(data: np.ndarray, center_az: int, n_az: int = PATCH_N_AZ) -> np.ndarray:
    """Extract n_az azimuths centered on center_az (wraps around 360°)."""
    n_total = data.shape[0]
    half = n_az // 2
    indices = np.arange(center_az - half, center_az + half) % n_total
    return data[indices]


def extract_radar_chip(
    nexrad_file: str,
    storm_lat: float,
    storm_lon: float,
    station_lat: float,
    station_lon: float,
    n_az: int = PATCH_N_AZ,
    n_rng: int = PATCH_N_RNG,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Read a NEXRAD Level-II file and extract a centered radar chip.
    Returns dict with keys DBZ, VEL, etc., each [2, n_az, n_rng], plus
    'range_folded_mask' [2, n_az, n_rng] and 'coordinates' (built separately).
    """
    try:
        import pyart
        radar = pyart.io.read(nexrad_file)
    except Exception as e:
        logger.error(f"Failed to read NEXRAD file {nexrad_file}: {e}")
        return None

    # Range and azimuth to storm center
    dist_m   = haversine_km(station_lat, station_lon, storm_lat, storm_lon) * 1000
    az_storm = bearing_deg(station_lat, station_lon, storm_lat, storm_lon)

    result: Dict[str, np.ndarray] = {}
    for target_tilt, tilt_deg in enumerate(TARGET_TILTS_DEG):
        tilt_idx = _find_tilt(radar, tilt_deg)
        n_rays   = radar.nrays // radar.nsweeps

        # Map azimuth degrees → ray index
        az_data   = radar.get_azimuth(tilt_idx)
        center_ray = int(np.argmin(np.abs(az_data - az_storm)))

        for var, candidates in RADAR_VAR_MAP.items():
            field = _read_field(radar, candidates, tilt_idx)
            if field is None:
                field = np.full((n_rays, radar.ngates), np.nan, dtype=np.float32)
            # Slice range gates
            patch_az = _centered_patch(field, center_ray, n_az)[:, :n_rng]
            # Pad/trim to exact shape
            if patch_az.shape[1] < n_rng:
                pad = np.full((n_az, n_rng - patch_az.shape[1]), np.nan, dtype=np.float32)
                patch_az = np.concatenate([patch_az, pad], axis=1)
            if var not in result:
                result[var] = np.zeros((2, n_az, n_rng), dtype=np.float32)
            result[var][target_tilt] = patch_az.astype(np.float32)

    # Range-folded mask (zeros — real implementation would check VEL quality flags)
    result["range_folded_mask"] = np.zeros((2, n_az, n_rng), dtype=np.float32)
    return result


def chip_to_tensors(
    chip: Dict[str, np.ndarray],
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """Convert numpy chip dict to batched tensors [1, 2, 120, 240]."""
    from ml.radar import build_coordinate_tensor, ALL_VARIABLES
    out: Dict[str, torch.Tensor] = {}
    for v in ALL_VARIABLES:
        arr = chip.get(v, np.zeros((2, PATCH_N_AZ, PATCH_N_RNG), dtype=np.float32))
        out[v] = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).to(device)
    rfm = chip.get("range_folded_mask", np.zeros((2, PATCH_N_AZ, PATCH_N_RNG), dtype=np.float32))
    out["range_folded_mask"] = torch.tensor(rfm, dtype=torch.float32).unsqueeze(0).to(device)
    out["coordinates"] = build_coordinate_tensor().unsqueeze(0).to(device)
    return out


# ── High-level async fetcher ──────────────────────────────────────────────────
async def fetch_radar_chip_for_event(
    lat: float,
    lon: float,
    event_time: datetime,
    cache_dir: Path = Path("data/nexrad_cache"),
) -> Optional[Dict[str, torch.Tensor]]:
    """
    Full pipeline: locate nearest NEXRAD, download file, extract chip.
    Returns tensor dict ready for RadarBackbone, or None on failure.
    """
    nearest = find_nearest_station(lat, lon)
    if nearest is None:
        logger.warning(f"No NEXRAD station within {MAX_RADAR_RANGE_KM}km of ({lat:.2f},{lon:.2f})")
        return None
    station_id, dist_km = nearest
    slat, slon, _ = WSR88D_STATIONS[station_id]
    logger.info(f"Nearest station: {station_id} at {dist_km:.1f}km")

    s3_key = await find_nexrad_file(station_id, event_time)
    if s3_key is None:
        logger.warning(f"No NEXRAD file found for {station_id} near {event_time}")
        return None

    cache_dir.mkdir(parents=True, exist_ok=True)
    local_path = cache_dir / Path(s3_key).name
    if not local_path.exists():
        try:
            import boto3
            from botocore import UNSIGNED
            from botocore.config import Config
            s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
            logger.info(f"Downloading {s3_key} …")
            s3.download_file(NEXRAD_BUCKET, s3_key, str(local_path))
        except Exception as e:
            logger.error(f"S3 download failed: {e}")
            return None

    chip = extract_radar_chip(str(local_path), lat, lon, slat, slon)
    if chip is None:
        return None
    return chip_to_tensors(chip)
