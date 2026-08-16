"""Fetch current atmospheric conditions for display alongside predictions."""

import logging
from datetime import datetime, timezone
from typing import Any

import httpx

logger = logging.getLogger(__name__)

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


async def get_current_conditions(lat: float, lon: float) -> dict[str, Any]:
    params = {
        "latitude": round(lat, 2),
        "longitude": round(lon, 2),
        "current": [
            "temperature_2m",
            "dewpoint_2m",
            "windspeed_10m",
            "winddirection_10m",
            "surface_pressure",
            "cape",
            "precipitation",
            "weathercode",
        ],
        "hourly": ["cape", "lifted_index", "convective_inhibition"],
        "past_hours": 1,
        "forecast_hours": 3,
        "timezone": "UTC",
    }
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(OPEN_METEO_URL, params=params)
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        logger.error(f"Weather fetch failed for ({lat},{lon}): {e}")
        return {"error": str(e), "lat": lat, "lon": lon}

    current = data.get("current", {})
    hourly = data.get("hourly", {})

    latest_cape = 0.0
    latest_li = 0.0
    if hourly.get("cape"):
        latest_cape = float(hourly["cape"][-1] or 0)
    if hourly.get("lifted_index"):
        latest_li = float(hourly["lifted_index"][-1] or 0)

    return {
        "lat": lat,
        "lon": lon,
        "fetched_at": datetime.now(tz=timezone.utc).isoformat(),
        "temperature_c": current.get("temperature_2m"),
        "dewpoint_c": current.get("dewpoint_2m"),
        "windspeed_ms": (current.get("windspeed_10m") or 0) / 3.6,
        "wind_direction_deg": current.get("winddirection_10m"),
        "pressure_hpa": current.get("surface_pressure"),
        "cape_jkg": latest_cape,
        "lifted_index": latest_li,
        "precipitation_mm": current.get("precipitation"),
        "weather_code": current.get("weathercode"),
    }
