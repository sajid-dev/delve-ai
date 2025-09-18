"""Request model for the chat API."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Represents a request payload for a chat message.

    This model supports multi‑user, multi‑conversation scenarios.  The
    ``message`` field contains the user's prompt.  Optional ``user_id``
    and ``conversation_id`` fields allow clients to specify an existing
    user and conversation; if omitted a new user or conversation will
    be created automatically.  The message must be a non‑empty string
    and is limited to 500 characters to prevent excessively long
    prompts from overloading the LLM service.  Pydantic will
    automatically validate incoming requests based on these
    constraints and return a 422 response if they are violated.
    """

    message: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="The user's message content."
    )
    user_id: str | None = Field(
        default=None,
        description="Optional identifier for the user.  If omitted a new user ID will be generated."
    )
    conversation_id: str | None = Field(
        default=None,
        description="Optional identifier for the conversation.  If omitted a new conversation will be started."
    )
