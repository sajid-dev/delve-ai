"""Admin endpoints for operational analytics."""

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger

from ..models.conversation import Conversation
from ..models.dashboard import DashboardData
from ..services.chat_service import ChatService, get_chat_service

router = APIRouter(prefix="/admin", tags=["Admin"])


@router.get("/dashboard", response_model=DashboardData)
async def dashboard_endpoint(
    service: ChatService = Depends(get_chat_service),
) -> DashboardData:
    """Return analytics for all user sessions."""
    try:
        return service.get_dashboard_data()
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Failed to build dashboard data")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to build dashboard data",
        ) from exc


@router.delete("/conversations")
async def delete_all_sessions_endpoint(
    service: ChatService = Depends(get_chat_service),
) -> dict[str, str]:
    """Delete every stored session across all users."""
    try:
        service.delete_all_sessions_global()
        return {"status": "ok"}
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Failed to delete all sessions")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete sessions",
        ) from exc


@router.delete("/conversations/{user_id}")
async def delete_user_sessions_endpoint(
    user_id: str,
    service: ChatService = Depends(get_chat_service),
) -> dict[str, str]:
    """Delete every stored session for a specific user."""
    try:
        service.delete_all_sessions(user_id)
        return {"status": "ok"}
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Failed to delete sessions for user %s", user_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete sessions",
        ) from exc


@router.get("/conversations/{user_id}", response_model=list[Conversation])
async def list_user_sessions_endpoint(
    user_id: str,
    service: ChatService = Depends(get_chat_service),
) -> list[Conversation]:
    """Return all sessions for the specified user."""
    try:
        return service.list_sessions(user_id)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Failed to list sessions for user %s", user_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list sessions",
        ) from exc
