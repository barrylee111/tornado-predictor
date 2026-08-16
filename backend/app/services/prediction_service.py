"""
Wraps TornadoPredictor for use in the Litestar app.
Generates Bill Paxton–style narratives from risk objects.
"""

import asyncio
import logging
import os
from pathlib import Path

import httpx

from ml.inference import TornadoPredictor, TornadoRisk

logger = logging.getLogger(__name__)

_predictor: TornadoPredictor | None = None


def get_predictor() -> TornadoPredictor:
    global _predictor
    if _predictor is None:
        _predictor = TornadoPredictor(
            model_path=Path(os.getenv("MODEL_PATH", "checkpoints/finetuned_best.pt")),
            scaler_path=Path(os.getenv("SCALER_PATH", "checkpoints/scaler.pkl")),
        )
    return _predictor


def generate_narrative(risk: TornadoRisk) -> str:
    """
    Bill Paxton as Bill Harding from Twister — calm urgency, plain language,
    no panic but no sugarcoating. Talks like a guy who's seen a lot of storms.
    """
    atm = risk.atmospheric_summary
    cape = atm.get("cape", 0)
    srh = atm.get("srh_03km", 0)
    shear = atm.get("shear_06km", 0)

    p1_pct = int(risk.prob_1h * 100)
    p2_pct = int(risk.prob_2h * 100)

    if risk.risk_level == "LOW":
        return (
            f"Alright, we're sitting at a {p1_pct}% chance over the next hour, "
            f"{p2_pct}% over two. Atmosphere's not doing much — CAPE's at {cape:.0f}, "
            f"shear's low. I'd keep an eye on the radar but you're not chasing anything today."
        )
    elif risk.risk_level == "ELEVATED":
        return (
            f"Okay, we've got something building here. {p1_pct}% chance in the next hour, "
            f"climbing to {p2_pct}% by two hours out. CAPE is {cape:.0f} joules, "
            f"we're seeing {srh:.0f} meters squared per second of helicity — "
            f"that's enough to spin something up. Stay aware."
        )
    elif risk.risk_level == "HIGH":
        return (
            f"Listen up — this is serious. {p1_pct}% probability in the next hour, "
            f"{p2_pct}% at two hours. CAPE is {cape:.0f}, helicity {srh:.0f}, "
            f"shear at {shear:.1f} meters per second. "
            f"If a tornado does drop, we're estimating EF{risk.ef_estimate:.0f} territory. "
            f"Take shelter now. Don't wait on this one."
        )
    else:  # EXTREME
        return (
            f"This is as bad as it gets. {p1_pct}% in the next hour, "
            f"{p2_pct}% at two hours. CAPE is off the charts at {cape:.0f}, "
            f"helicity {srh:.0f} — the atmosphere is primed. "
            f"We're looking at a potential EF{risk.ef_estimate:.0f}. "
            f"If you are in this area, get underground now. I mean it."
        )


async def run_prediction(lat: float, lon: float) -> TornadoRisk:
    predictor = get_predictor()
    risk = await predictor.predict(lat, lon)
    return risk


ELEVENLABS_TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"


async def synthesize_speech(text: str, voice_id: str) -> bytes | None:
    """Call ElevenLabs API. Returns audio bytes or None if unavailable."""
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        return None

    url = ELEVENLABS_TTS_URL.format(voice_id=voice_id)
    payload = {
        "text": text,
        "model_id": "eleven_turbo_v2_5",
        "voice_settings": {
            "stability": 0.45,
            "similarity_boost": 0.80,
            "style": 0.35,
            "use_speaker_boost": True,
        },
    }
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.post(
                url,
                json=payload,
                headers={
                    "xi-api-key": api_key,
                    "Content-Type": "application/json",
                    "Accept": "audio/mpeg",
                },
            )
            r.raise_for_status()
            return r.content
    except Exception as e:
        logger.error(f"ElevenLabs TTS failed: {e}")
        return None
