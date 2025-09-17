"""Logging utilities for the application.

This module configures Loguru to emit structured logs to a file with
rotation and retention policies.  It exposes a ``configure_logger``
function that can be called during application startup.
"""

from __future__ import annotations

import os
from loguru import logger


def configure_logger(
    log_dir: str | None = None,
    log_file: str | None = None,
    level: str = "INFO",
) -> None:
    """Set up log sinks and formatting.

    Parameters
    ----------
    log_dir: str | None
        Directory to write logs to.  If provided this will override
        the directory part of ``log_file``.  If both are ``None``, a
        ``logs`` directory within the current working directory will be used.
    log_file: str | None
        Full path of the log file.  If provided it overrides the
        combination of ``log_dir`` and default file name.  This is
        useful when the log file location is set via environment
        variables.
    level: str
        Logging level (e.g. ``INFO``, ``DEBUG``).
    """
    # Determine log file path
    if log_file:
        path = log_file
        directory = os.path.dirname(path)
    else:
        directory = log_dir or os.path.join(os.getcwd(), "logs")
        path = os.path.join(directory, "app.log")
    os.makedirs(directory, exist_ok=True)

    # Remove default handlers to avoid duplicate logging
    logger.remove()

    # Add rotating file handler
    logger.add(
        path,
        rotation="10 MB",
        retention="10 days",
        compression="zip",
        enqueue=True,
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )