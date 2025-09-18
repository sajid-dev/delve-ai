"""Models representing chat messages and related structures."""

from pydantic import BaseModel
from .enums import MessageRole


class ChatMessage(BaseModel):
    """Represents a single message in a conversation.

    In addition to the role and content, each message is timestamped.
    The ``timestamp`` field records when the message was created in ISO
    8601 format (UTC).  This allows for ordering and auditing of
    conversations over time.  Clients are not required to supply a
    timestamp; if omitted, one will be generated automatically by
    services when the message is persisted.
    """

    role: MessageRole
    content: str
    timestamp: str | None = None