"""API controller for chat operations.

Defines the routes for interacting with the ChatService.  All
endpoints defined here are registered in ``gi.py``.
"""

"""Controllers for chat endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger

from ..models.chat_request import ChatRequest
from ..models.chat_response import ChatResponse
from ..models.conversation import Conversation
from ..services.chat_service import ChatService, get_chat_service
from ..utils.error_handler import ChatError

router = APIRouter(prefix="", tags=["Chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    """Accept a chat request and return the assistant's response.

    The request may include optional ``user_id`` and ``session_id``
    fields.  When omitted, new identifiers are generated automatically.
    In addition to the answer text, the response contains the user
    and session identifiers so the client can continue the
    dialogue in context.
    """
    try:
        logger.info("Received chat request: {}", request)
        response = service.chat(request)
        logger.info("Answer generated successfully")
        return response
    except ChatError as exc:
        logger.error("ChatError: {}", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Unhandled exception during chat processing")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        ) from exc


@router.get("/conversations", response_model=list[Conversation])
async def list_conversations_endpoint(
    user_id: str,
    service: ChatService = Depends(get_chat_service),
) -> list[Conversation]:
    """List all conversations for a user.

    The ``user_id`` query parameter is required.  Returns a list of
    conversation metadata objects.  An empty list is returned if the
    user has no conversations.
    """
    try:
        logger.info("Listing conversations for user: {}", user_id)
        return service.list_conversations(user_id)
    except Exception as exc:
        logger.exception("Failed to list conversations")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list conversations",
        ) from exc


@router.get("/conversations/{conversation_id}", response_model=Conversation)
async def get_conversation_endpoint(
    conversation_id: str,
    user_id: str,
    service: ChatService = Depends(get_chat_service),
) -> Conversation:
    """Retrieve a single conversation for a user."""
    try:
        conv = service.get_conversation(user_id, conversation_id)
        if conv is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
        return conv
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to get conversation")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get conversation",
        ) from exc


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation_endpoint(
    conversation_id: str,
    user_id: str,
    service: ChatService = Depends(get_chat_service),
) -> None:
    """Delete a specific conversation for a user."""
    try:
        service.delete_conversation(user_id, conversation_id)
        return None
    except Exception as exc:
        logger.exception("Failed to delete conversation")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete conversation",
        ) from exc


@router.delete("/conversations", status_code=status.HTTP_204_NO_CONTENT)
async def delete_all_conversations_endpoint(
    user_id: str,
    service: ChatService = Depends(get_chat_service),
) -> None:
    """Delete all conversations for a user."""
    try:
        service.delete_all_conversations(user_id)
        return None
    except Exception as exc:
        logger.exception("Failed to delete all conversations")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete conversations",
        ) from exc
