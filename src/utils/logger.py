"""Logging utilities for the application.

This module configures Loguru to emit structured logs to both the console
and an optional file.  It also bridges the standard Python ``logging``
module to Loguru so that messages from third‑party libraries are
captured consistently.  A global ``app_logger`` is provided for
convenience.
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path
from typing import Optional

from loguru import logger

from ..config.app_config import get_app_config


def setup_logging() -> "loguru.Logger":
    """Configure Loguru logging for the application.

    This function initialises the global Loguru logger with sensible
    defaults.  It removes the default Loguru handler, ensures the log
    directory exists if a log file is configured, and adds sinks for
    console and file outputs.  It also redirects the standard Python
    ``logging`` module to Loguru via a custom handler.

    Returns
    -------
    loguru.Logger
        The configured Loguru logger instance.
    """
    # Load application configuration
    app_config = get_app_config()

    # Remove default handler to prevent duplicate logs
    logger.remove()

    # Ensure the logs directory exists if a log file is configured
    if app_config.log_file:
        log_path = Path(app_config.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

    # Define a rich log format with colours and context
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    # Add console sink with colourised output
    logger.add(
        sys.stdout,
        level=app_config.log_level,
        format=log_format,
        colorize=True,
        backtrace=True,
        diagnose=app_config.app_debug,
    )

    # Add file sink with rotation and retention if configured
    if app_config.log_file:
        logger.add(
            app_config.log_file,
            level=app_config.log_level,
            format=log_format,
            rotation="10 MB",  # Rotate after 10MB
            retention="30 days",  # Keep logs for 30 days
            compression="zip",
            backtrace=True,
            diagnose=app_config.app_debug,
        )

    # Redirect the standard logging module to Loguru
    class LoguruHandler(logging.Handler):
        """Handler to forward standard logging records to Loguru."""

        def emit(self, record: logging.LogRecord) -> None:
            try:
                # Fetch the corresponding Loguru level if it exists
                level = logger.level(record.levelname).name
            except (KeyError, ValueError):
                level = record.levelno

            # Find the caller from where the logging call was made
            frame, depth = logging.currentframe(), 2
            while frame and frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1

            logger.opt(depth=depth, exception=record.exc_info).log(
                level, record.getMessage()
            )

    # Configure root logging to use our Loguru handler
    logging.basicConfig(handlers=[LoguruHandler()], level=logging.WARNING)

    # Emit some startup diagnostics
    logger.info("Logging configured successfully")
    logger.debug(f"App environment: {app_config.app_env}")
    logger.debug(f"Log level: {app_config.log_level}")

    return logger


# Provide a global logger instance for convenience
app_logger = logger