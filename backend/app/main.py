import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from litestar import Litestar, get
from litestar.config.cors import CORSConfig
from litestar.logging import LoggingConfig
from litestar.openapi import OpenAPIConfig
from litestar.openapi.spec import Tag

from app.api.prediction import prediction_router
from app.api.weather import weather_router

load_dotenv()

logging_config = LoggingConfig(
    root={"level": "INFO", "handlers": ["console"]},
    formatters={"standard": {"format": "%(asctime)s %(levelname)s %(name)s: %(message)s"}},
)

cors_config = CORSConfig(
    allow_origins=[os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@get("/health", tags=["meta"])
async def health() -> dict:
    return {"status": "ok"}


app = Litestar(
    route_handlers=[health, prediction_router, weather_router],
    cors_config=cors_config,
    logging_config=logging_config,
    openapi_config=OpenAPIConfig(
        title="Tornado Predictor API",
        version="0.1.0",
        tags=[
            Tag(name="prediction", description="Tornado risk predictions"),
            Tag(name="weather", description="Current atmospheric conditions"),
        ],
    ),
)
