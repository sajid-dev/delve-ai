"""Manage memory instances per user.

If the application needs to maintain separate conversation histories
for different users, this manager can be used to allocate and
retrieve ChatMemory objects keyed by a user identifier.  For now we
provide a basic implementation.
"""

from __future__ import annotations

from typing import Dict

from ..config.llm_config import LlmConfig
from .chat_memory import ChatMemory


class UserMemoryManager:
    """Handles mapping of user IDs to ChatMemory instances."""

    def __init__(self, llm_config: LlmConfig) -> None:
        self.llm_config = llm_config
        self._memories: Dict[str, ChatMemory] = {}

    def get_memory(self, user_id: str) -> ChatMemory:
        """Return a ChatMemory instance for the given user."""
        if user_id not in self._memories:
            self._memories[user_id] = ChatMemory(self.llm_config)
        return self._memories[user_id]