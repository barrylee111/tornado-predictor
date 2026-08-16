"""Inference pipeline — fetch live weather, run model, return structured prediction."""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
import torch

from ml.data import AtmosphericScaler, fetch_era5_features, _fill_nan_linear
from ml.model import TornadoTransformer, ModelConfig

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path("checkpoints/finetuned_best.pt")
DEFAULT_SCALER_PATH = Path("checkpoints/scaler.pkl")

import os
OPEN_METEO_FORECAST_URL = "https://customer-api.open-meteo.com/v1/forecast"
_API_KEY = os.environ.get("OPEN_METEO_API_KEY", "")

FORECAST_HOURLY_VARS = [
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
class TornadoRisk:
    lat: float
    lon: float
    valid_time: datetime
    prob_1h: float        # tornado probability in next 1 hour
    prob_2h: float        # tornado probability in next 2 hours
    risk_level: str       # LOW / ELEVATED / HIGH / EXTREME
    ef_estimate: float    # estimated EF scale if tornado occurs
    atmospheric_summary: dict[str, float]
    model_version: str


def _risk_level(prob: float) -> str:
    if prob < 0.05:
        return "LOW"
    if prob < 0.20:
        return "ELEVATED"
    if prob < 0.50:
        return "HIGH"
    return "EXTREME"


async def fetch_live_features(lat: float, lon: float, client: httpx.AsyncClient) -> np.ndarray | None:
    """Fetch the past 24h of atmospheric data from Open-Meteo forecast API."""
    params = {
        "latitude": round(lat, 2),
        "longitude": round(lon, 2),
        "hourly": ",".join(FORECAST_HOURLY_VARS),
        "past_days": 1,
        "forecast_days": 1,
        "timezone": "UTC",
        "apikey": _API_KEY,
    }
    try:
        r = await client.get(OPEN_METEO_FORECAST_URL, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        logger.error(f"Failed to fetch live weather for ({lat},{lon}): {e}")
        return None

    hourly = data.get("hourly", {})
    import pandas as pd
    times = pd.to_datetime(hourly.get("time", []), utc=True)
    now = datetime.now(tz=timezone.utc)
    past_mask = times <= now
    if past_mask.sum() < 6:
        return None

    from ml.data import OPEN_METEO_VARS, _build_feature_matrix, _fill_nan_linear
    raw = {}
    for var in FORECAST_HOURLY_VARS:
        vals = np.array(hourly.get(var, []), dtype=np.float32)
        raw[var] = vals[past_mask] if len(vals) == len(times) else np.full(past_mask.sum(), np.nan)
    for var in OPEN_METEO_VARS:
        if var not in raw:
            raw[var] = np.zeros(past_mask.sum(), dtype=np.float32)

    feats = _build_feature_matrix(raw)
    _fill_nan_linear(feats)
    return feats[-24:]


class TornadoPredictor:
    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        scaler_path: Path = DEFAULT_SCALER_PATH,
        device: str | None = None,
    ):
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
        else:
            self.device = torch.device(device)

        self.model: TornadoTransformer | None = None
        self.scaler: AtmosphericScaler | None = None
        self._model_version = "untrained"

        if model_path.exists():
            self.model = TornadoTransformer.load(model_path, str(self.device)).to(self.device)
            self.model.eval()
            self._model_version = model_path.stem
            logger.info(f"Loaded model from {model_path} on {self.device}")
        else:
            logger.warning(f"No model checkpoint at {model_path} — using statistical fallback")

        if scaler_path.exists():
            self.scaler = AtmosphericScaler.load(scaler_path)

    async def predict(self, lat: float, lon: float) -> TornadoRisk:
        async with httpx.AsyncClient() as client:
            features = await fetch_live_features(lat, lon, client)

        if features is None or len(features) < 2:
            return self._fallback_risk(lat, lon)

        if self.scaler:
            features = self.scaler.transform(features)

        atm_summary = self._atmospheric_summary(features)

        if self.model is None:
            prob_1h, prob_2h, ef_est = self._statistical_predict(features)
        else:
            prob_1h, prob_2h, ef_est = self._model_predict(features)

        risk = max(prob_1h, prob_2h)
        return TornadoRisk(
            lat=lat,
            lon=lon,
            valid_time=datetime.now(tz=timezone.utc),
            prob_1h=round(prob_1h, 4),
            prob_2h=round(prob_2h, 4),
            risk_level=_risk_level(risk),
            ef_estimate=round(ef_est, 1),
            atmospheric_summary=atm_summary,
            model_version=self._model_version,
        )

    def _model_predict(self, features: np.ndarray) -> tuple[float, float, float]:
        from ml.model import INTENSITY_CLASS_EF_MIDPOINT
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(x)
        probs = out["tornado_prob"].squeeze(0).cpu().tolist()
        p1 = probs[0] if len(probs) > 0 else 0.0
        p2 = probs[1] if len(probs) > 1 else p1
        # intensity_logits: (1, forecast_hours, n_classes) → average over hours → argmax
        logits = out["intensity_logits"].squeeze(0)         # (forecast_hours, n_classes)
        avg_probs = torch.softmax(logits.mean(dim=0), dim=0)
        intensity_cls = int(avg_probs.argmax().item())
        ef_est = INTENSITY_CLASS_EF_MIDPOINT[intensity_cls]
        return p1, p2, ef_est

    def _statistical_predict(self, features: np.ndarray) -> tuple[float, float, float]:
        """
        Physics-informed fallback when no trained model is available.
        Based on known tornado-favorable thresholds from meteorological literature.
        """
        from ml.model import FEATURE_INDEX
        latest = features[-1]

        def get(name: str, default: float = 0.0) -> float:
            idx = FEATURE_INDEX.get(name)
            if idx is None:
                return default
            v = float(latest[idx])
            return v if np.isfinite(v) else default

        cape = get("cape")
        srh_01 = get("srh_01km")
        srh_03 = get("srh_03km")
        shear = get("shear_06km")
        lifted = get("lifted_index")
        cin = get("cin")

        # STP-like index (simplified)
        cape_term = min(cape / 1500.0, 2.0)
        srh_term = min(srh_03 / 150.0, 2.0)
        shear_term = min(shear / 18.0, 2.0)
        lifted_term = max(0.0, -lifted / 6.0)
        cin_penalty = max(0.0, 1.0 - cin / 100.0)

        stp = cape_term * srh_term * shear_term * lifted_term * cin_penalty
        prob = 1.0 / (1.0 + np.exp(-2.0 * (stp - 1.0)))  # logistic
        prob = float(np.clip(prob, 0.0, 1.0))
        return prob * 0.9, prob * 0.75, stp * 0.8

    def _atmospheric_summary(self, features: np.ndarray) -> dict[str, float]:
        from ml.model import FEATURE_NAMES
        latest = features[-1]
        return {
            name: round(float(v), 2) if np.isfinite(v) else 0.0
            for name, v in zip(FEATURE_NAMES, latest)
        }

    def _fallback_risk(self, lat: float, lon: float) -> TornadoRisk:
        return TornadoRisk(
            lat=lat, lon=lon,
            valid_time=datetime.now(tz=timezone.utc),
            prob_1h=0.0, prob_2h=0.0,
            risk_level="LOW",
            ef_estimate=0.0,
            atmospheric_summary={},
            model_version="fallback",
        )
