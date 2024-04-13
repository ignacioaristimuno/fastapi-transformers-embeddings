from fastapi import FastAPI

from src.api.routers.health import health_router
from src.api.routers.encoding import encoding_router


def init_routers(app: FastAPI) -> None:
    """Function for initializing the routers of the API"""

    app.include_router(health_router)
    app.include_router(encoding_router, prefix="/encoding")
