"""Logging configuration using Loguru.

The log configuration ensures that structured logs are written
to ``logs/app.log`` with rotation and retention.  It removes
the default stderr handler installed by Loguru so that logs
are not duplicated.
"""

from __future__ import annotations

import os
from typing import Optional

from loguru import logger


def configure_logging(log_dir: Optional[str] = None) -> None:
    """Configure Loguru to write rotating logs.

    Parameters
    ----------
    log_dir: Optional[str]
        Directory in which to store log files.  When ``None`` a default
        ``logs`` directory within the project root is used.

    The function ensures the directory exists, removes the default
    handler and adds a new file sink with rotation and retention.  It
    is idempotent; subsequent calls will not duplicate sinks.
    """
    directory = log_dir or os.path.join(os.getcwd(), "logs")
    os.makedirs(directory, exist_ok=True)
    log_file = os.path.join(directory, "app.log")

    # Remove existing handlers to avoid duplicate logs
    logger.remove()

    # Add rotating file handler
    logger.add(
        log_file,
        rotation="10 MB",    # rotate log after it reaches 10 MB
        retention="10 days",  # keep logs for 10 days
        compression="zip",     # compress old logs
        enqueue=True,
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )