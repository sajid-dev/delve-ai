"""Error handling utilities and custom exceptions."""

from __future__ import annotations

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger


class ChatError(Exception):
    """Exception raised when a chat operation fails."""

    pass


async def http_exception_handler(request: Request, exc: ChatError) -> JSONResponse:
    """Convert a ChatError into an HTTP 500 response."""
    logger.error("ChatError occurred: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )