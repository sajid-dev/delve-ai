"""Admin endpoints for operational analytics."""

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger

from ..models.dashboard import DashboardData
from ..services.chat_service import ChatService, get_chat_service

router = APIRouter(prefix="/admin", tags=["Admin"])


@router.get("/dashboard", response_model=DashboardData)
async def dashboard_endpoint(
    service: ChatService = Depends(get_chat_service),
) -> DashboardData:
    """Return analytics for all user conversations."""
    try:
        return service.get_dashboard_data()
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Failed to build dashboard data")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to build dashboard data",
        ) from exc
