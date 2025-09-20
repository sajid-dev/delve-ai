"""Response model for the chat API."""

from pydantic import BaseModel, Field

from .enums import MessageContentType


class ChatResponse(BaseModel):
    """Represents the assistant's reply to a chat request.

    The response includes the generated answer text as well as the
    identifiers for the user and conversation.  These identifiers
    enable the client to maintain context across multiple messages.
    """

    user_id: str
    session_id: str = Field(
        ...,
        description="Identifier for the session tied to this response.",
    )
    answer: str
    content_type: MessageContentType = Field(
        default=MessageContentType.TEXT,
        description="Classification of the assistant's answer content.",
    )
