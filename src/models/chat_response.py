"""Response model for the chat API."""

from typing import Any

from pydantic import BaseModel, Field


class ChatResponse(BaseModel):
    """Represents the assistant's reply to a chat request.

    The response includes identifiers for the user and session so
    the client can maintain context across messages.  Assistant output
    is delivered entirely through the ``data`` payload which contains a
    ``type`` discriminator and a render-ready ``payload`` structure.
    """

    user_id: str
    session_id: str = Field(
        ...,
        description="Identifier for the session tied to this response.",
    )
    data: Any = Field(
        ...,
        description="Structured payload (text, table rows, chart spec, list items, etc.).",
    )
