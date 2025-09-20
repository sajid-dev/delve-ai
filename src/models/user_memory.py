"""Data structure to hold a user's memory across sessions."""

from typing import Dict

from pydantic import BaseModel, Field

from .conversation import Conversation


class UserMemory(BaseModel):
    """Stores sessions keyed by user ID."""

    sessions: Dict[str, Conversation] = Field(default_factory=dict)
