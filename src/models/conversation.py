"""Model representing a full conversation."""

from typing import List
from pydantic import BaseModel

from .chat_message import ChatMessage


class Conversation(BaseModel):
    """A conversation is composed of a sequence of chat messages."""

    messages: List[ChatMessage]