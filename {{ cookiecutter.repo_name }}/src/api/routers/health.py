from fastapi import APIRouter

from src.settings import custom_logger


logger = custom_logger("API Health")

health_router = APIRouter()


@health_router.get("/health")
def ping():
    """Endpoint for checking the health of the API server"""

    return {"status": "ok"}
