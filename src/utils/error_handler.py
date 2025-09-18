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
    logger.error("ChatError occurred: {}", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )

# ---------------------------------------------------------------------------
# Decorators for synchronous service/controller methods

from functools import wraps
from typing import Any, Callable, Dict


def handle_llm_error(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to handle errors arising from LLM operations.

    If a :class:`ChatError` is raised by the wrapped function, a
    dictionary with an error message and type is returned.  Any other
    unexpected exceptions are logged and converted into a generic error
    response.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        try:
            return func(*args, **kwargs)
        except ChatError as exc:
            # Log the error and return a structured response
            logger.error("LLM error: {}", exc)
            return {"success": False, "error": str(exc), "error_type": "llm"}
        except Exception as exc:
            logger.exception("Unexpected error in {}", func.__name__)
            return {
                "success": False,
                "error": "An unexpected error occurred during LLM processing.",
                "error_type": "unexpected",
            }

    return wrapper


def handle_memory_error(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to handle errors arising from memory operations.

    Captures exceptions and returns a standardised error dictionary.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            logger.exception("Memory error in {}", func.__name__)
            return {
                "success": False,
                "error": "A memory error occurred.",
                "error_type": "memory",
            }

    return wrapper
