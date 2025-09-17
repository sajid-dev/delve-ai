"""Request model for the chat API."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Represents a request payload for a chat message.

    The message must be a non-empty string.  A maximum length is
    enforced to prevent excessively long prompts from overloading the
    LLM service.  Pydantic will automatically validate incoming
    requests based on these constraints and return a 422 response if
    they are violated.
    """

    message: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="The user's message content."
    )