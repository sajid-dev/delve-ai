"""API controller for chat operations.

Defines the routes for interacting with the ChatService.  All
endpoints defined here are registered in ``main.py``.
"""

"""Controllers for chat endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger

from ..models.chat_request import ChatRequest
from ..models.chat_response import ChatResponse
from ..services.chat_service import ChatService, get_chat_service
from ..utils.error_handler import ChatError

router = APIRouter(prefix="", tags=["Chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    """Accept a user prompt and return an assistant response.

    A 500 error is returned if the underlying service raises a
    ChatError.
    """
    try:
        logger.info("Received chat request: %r", request.message)
        answer = service.chat(request.message)
        logger.info("Answer generated successfully")
        return ChatResponse(answer=answer)
    except ChatError as exc:
        logger.error("ChatError: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        # Catch any unexpected exceptions and return an internal server error
        logger.exception("Unhandled exception during chat processing")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        ) from exc