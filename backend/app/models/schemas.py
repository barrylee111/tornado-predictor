from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)


class AtmosphericConditions(BaseModel):
    cape: float
    cin: float
    srh_01km: float
    srh_03km: float
    shear_06km: float
    lifted_index: float
    k_index: float


class TornadoRiskResponse(BaseModel):
    lat: float
    lon: float
    valid_time: datetime
    prob_1h: float = Field(..., description="Tornado probability in next 1 hour (0–1)")
    prob_2h: float = Field(..., description="Tornado probability in next 2 hours (0–1)")
    risk_level: Literal["LOW", "ELEVATED", "HIGH", "EXTREME"]
    ef_estimate: float = Field(..., description="Estimated EF scale if tornado occurs (0–5)")
    atmospheric_summary: dict[str, float]
    model_version: str
    narrative: str = Field(..., description="Bill Paxton–style voice narrative")


class GridPredictionRequest(BaseModel):
    lat_min: float = Field(default=25.0)
    lat_max: float = Field(default=50.0)
    lon_min: float = Field(default=-105.0)
    lon_max: float = Field(default=-75.0)
    resolution: float = Field(default=2.0, description="Grid spacing in degrees")


class GridCell(BaseModel):
    lat: float
    lon: float
    prob_2h: float
    risk_level: Literal["LOW", "ELEVATED", "HIGH", "EXTREME"]


class GridPredictionResponse(BaseModel):
    cells: list[GridCell]
    valid_time: datetime
    model_version: str


class TTSRequest(BaseModel):
    text: str = Field(..., max_length=2000)
    voice_id: str = Field(
        default="pNInz6obpgDQGcFmaJgB",  # ElevenLabs "Adam" — warm, authoritative
        description="ElevenLabs voice ID"
    )


class TTSResponse(BaseModel):
    audio_url: str | None = None
    text: str
    provider: Literal["elevenlabs", "browser"]
