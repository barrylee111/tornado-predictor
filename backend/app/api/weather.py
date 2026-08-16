from litestar import Router, get
from litestar.exceptions import HTTPException

from app.services.weather_service import get_current_conditions


@get("/weather")
async def current_weather(lat: float, lon: float) -> dict:
    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        raise HTTPException(status_code=400, detail="Invalid coordinates")
    return await get_current_conditions(lat, lon)


weather_router = Router(path="/api", route_handlers=[current_weather])
