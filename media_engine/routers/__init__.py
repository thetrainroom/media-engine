"""API routers for Media Engine."""

from media_engine.routers.batch import router as batch_router
from media_engine.routers.health import router as health_router
from media_engine.routers.models import router as models_router
from media_engine.routers.settings import router as settings_router
from media_engine.routers.utils import router as utils_router

__all__ = [
    "batch_router",
    "health_router",
    "models_router",
    "settings_router",
    "utils_router",
]
