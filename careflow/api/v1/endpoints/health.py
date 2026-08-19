"""Health Check Endpoint for history-service."""

from fastapi import APIRouter
from careflow.core.config import settings

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("", summary="Service Health Check")
async def health_check():
    """Health check endpoint returning service status, version, and environment."""
    return {
        "status": "ok",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION if hasattr(settings, "APP_VERSION") else "1.0.0",
        "environment": settings.APP_ENV,
    }
