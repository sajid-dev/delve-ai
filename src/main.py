"""FastAPI application entry point.

This module initialises the FastAPI app, configures logging and
registers API routes.  The `uvicorn` ASGI server can point to
``src.main:app`` to serve the application.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from .utils.logger import setup_logging
# Import settings from the src.config package.  The configuration files
# have been moved under the `src/config` directory, so we import
# Settings and helpers relative to this package.
# Load application configuration for logging.  The application and LLM
# settings have been separated into dedicated config modules.
from .config.app_config import app_config  # noqa: F401
from .controllers.chat_controller import router as chat_router
from .controllers.admin_controller import router as admin_router
from .utils.error_handler import ChatError, http_exception_handler


def create_app() -> FastAPI:
    """Create and configure a FastAPI application."""
    # Configure structured logging using application settings
    # Calling setup_logging() initialises Loguru with console and file sinks.
    setup_logging()

    app = FastAPI(title="LLM Chat App", version="0.1.0")

    # Enable CORS for all origins; adjust in production as needed
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register exception handler for ChatError
    app.add_exception_handler(ChatError, http_exception_handler)

    # Include chat routes
    app.include_router(chat_router)
    # Include admin analytics routes
    app.include_router(admin_router)

    @app.get("/health", tags=["Health"])
    async def health() -> dict[str, str]:
        """Simple health check endpoint."""
        logger.debug("Health check invoked")
        return {"status": "ok"}

    return app


# Create an application instance for ASGI servers
app = create_app()
