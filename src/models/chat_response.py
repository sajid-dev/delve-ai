"""Response model for the chat API."""

from pydantic import BaseModel


class ChatResponse(BaseModel):
    """Represents the assistant's reply to a chat request.

    The response includes the generated answer text as well as the
    identifiers for the user and conversation.  These identifiers
    enable the client to maintain context across multiple messages.
    """

    user_id: str
    conversation_id: str
    answer: str