"""Model representing a full conversation."""

from __future__ import annotations

from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field

from .chat_message import ChatMessage


class Conversation(BaseModel):
    """Represents a conversation between a user and the assistant.

    A conversation has a unique identifier scoped to a user.  It also
    records metadata such as creation and last updated times, an
    optional user-provided title, and a count of messages exchanged.
    The ``messages`` field contains the chronological sequence of
    messages in the conversation.  When serialised, timestamps are
    formatted as ISO 8601 strings in UTC.
    """

    conversation_id: str = Field(..., description="Unique identifier for the conversation.")
    user_id: str = Field(..., description="Identifier of the user who owns this conversation.")
    title: Optional[str] = Field(
        default=None,
        description="Optional title assigned to the conversation.  Can be derived from the first prompt."
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.utcnow(),
        description="Timestamp when the conversation was created (UTC)."
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.utcnow(),
        description="Timestamp when the conversation was last updated (UTC)."
    )
    message_count: int = Field(
        default=0,
        description="Number of messages exchanged in the conversation."
    )
    messages: List[ChatMessage] = Field(
        default_factory=list,
        description="Chronological list of messages in the conversation."
    )

    def add_message(self, message: ChatMessage) -> None:
        """Append a new message and update metadata.

        When a message is added, ``updated_at`` is refreshed and the
        ``message_count`` is incremented.  This method should be
        called by services when persisting interactions.
        """
        self.messages.append(message)
        self.message_count += 1
        self.updated_at = datetime.utcnow()