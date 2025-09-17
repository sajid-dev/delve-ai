"""Simple HTTP client utilities using httpx.

Although the application currently does not make outbound HTTP
requests, this module provides a ready interface for future
integrations with external services.
"""

from __future__ import annotations

import httpx
from typing import Any, Dict


async def post(url: str, json: Dict[str, Any]) -> httpx.Response:
    """Perform an asynchronous HTTP POST request."""
    async with httpx.AsyncClient() as client:
        return await client.post(url, json=json)