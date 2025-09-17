"""Response model for the chat API."""

from pydantic import BaseModel


class ChatResponse(BaseModel):
    """Represents the assistant's reply."""

    answer: str