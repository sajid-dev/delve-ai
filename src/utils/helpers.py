"""General helper functions used across the application."""

from __future__ import annotations

from typing import Any, Callable, Coroutine, TypeVar

from loguru import logger

T = TypeVar("T")


def log_execution(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to log the execution of a function."""

    def wrapper(*args: Any, **kwargs: Any) -> T:
        logger.debug("Entering %s", func.__name__)
        result = func(*args, **kwargs)
        logger.debug("Exiting %s", func.__name__)
        return result

    return wrapper