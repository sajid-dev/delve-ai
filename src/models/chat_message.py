"""Models representing chat messages and related structures."""

from pydantic import BaseModel
from .enums import MessageRole


class ChatMessage(BaseModel):
    """Represents a single message in a conversation with a role and content."""

    role: MessageRole
    content: str