"""Data structure to hold a user's memory across conversations."""

from typing import Dict

from pydantic import BaseModel, Field

from .conversation import Conversation


class UserMemory(BaseModel):
    """Stores conversations keyed by user ID."""

    conversations: Dict[str, Conversation] = Field(default_factory=dict)
