import asyncio
import base64
import logging
from datetime import datetime, timezone

from litestar import Router, get, post
from litestar.exceptions import HTTPException

from app.models.schemas import (
    GridCell,
    GridPredictionRequest,
    GridPredictionResponse,
    PredictionRequest,
    TTSRequest,
    TTSResponse,
    TornadoRiskResponse,
)
from app.services.prediction_service import (
    generate_narrative,
    run_prediction,
    synthesize_speech,
)

logger = logging.getLogger(__name__)


@get("/predict")
async def predict(lat: float, lon: float) -> TornadoRiskResponse:
    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        raise HTTPException(status_code=400, detail="Invalid coordinates")

    risk = await run_prediction(lat, lon)
    narrative = generate_narrative(risk)

    return TornadoRiskResponse(
        lat=risk.lat,
        lon=risk.lon,
        valid_time=risk.valid_time,
        prob_1h=risk.prob_1h,
        prob_2h=risk.prob_2h,
        risk_level=risk.risk_level,
        ef_estimate=risk.ef_estimate,
        atmospheric_summary=risk.atmospheric_summary,
        model_version=risk.model_version,
        narrative=narrative,
    )


@post("/predict/grid")
async def predict_grid(data: GridPredictionRequest) -> GridPredictionResponse:
    import numpy as np

    lats = np.arange(data.lat_min, data.lat_max, data.resolution)
    lons = np.arange(data.lon_min, data.lon_max, data.resolution)
    coords = [(float(la), float(lo)) for la in lats for lo in lons]

    # Batch predictions with limited concurrency to avoid hammering Open-Meteo
    sem = asyncio.Semaphore(25)

    async def _predict_one(lat: float, lon: float) -> GridCell | None:
        async with sem:
            try:
                risk = await run_prediction(lat, lon)
                return GridCell(
                    lat=lat,
                    lon=lon,
                    prob_2h=risk.prob_2h,
                    risk_level=risk.risk_level,
                )
            except Exception as e:
                logger.warning(f"Grid predict failed ({lat},{lon}): {e}")
                return None

    results = await asyncio.gather(*[_predict_one(la, lo) for la, lo in coords])
    cells = [r for r in results if r is not None]

    return GridPredictionResponse(
        cells=cells,
        valid_time=datetime.now(tz=timezone.utc),
        model_version="current",
    )


@post("/tts")
async def text_to_speech(data: TTSRequest) -> TTSResponse:
    audio_bytes = await synthesize_speech(data.text, data.voice_id)
    if audio_bytes:
        audio_b64 = base64.b64encode(audio_bytes).decode()
        return TTSResponse(
            audio_url=f"data:audio/mpeg;base64,{audio_b64}",
            text=data.text,
            provider="elevenlabs",
        )
    # No ElevenLabs key — tell the frontend to use Web Speech API
    return TTSResponse(text=data.text, provider="browser")


prediction_router = Router(
    path="/api",
    route_handlers=[predict, predict_grid, text_to_speech],
)
