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
    """Manage per‑user and per‑conversation memory instances and metadata.

    The manager maintains separate memories for each conversation
    belonging to a user.  Conversation metadata (such as message
    counts and timestamps) is stored alongside the memory and can be
    queried or updated.  New conversations are automatically created
    on demand with unique identifiers.
    """

    def __init__(self, llm_config: LlmConfig) -> None:
        self.llm_config = llm_config
        # Mapping of user_id -> conversation_id -> ChatMemory
        self._memories: Dict[str, Dict[str, ChatMemory]] = {}
        # Mapping of user_id -> conversation_id -> Conversation metadata
        from ..models.conversation import Conversation

        self._conversations: Dict[str, Dict[str, Conversation]] = {}

    def get_memory(self, user_id: str, conversation_id: str) -> ChatMemory:
        """Retrieve or create a ChatMemory for a user's conversation.

        When a memory for the given ``user_id`` and ``conversation_id`` does
        not exist, a new one is created.  The memory will persist its
        data in a conversation‑specific directory (``chroma_db/<user>/<conversation>``)
        so that separate histories are isolated on disk.
        """
        if user_id not in self._memories:
            self._memories[user_id] = {}
        if conversation_id not in self._memories[user_id]:
            # Determine the persistence directory for this conversation
            persist_dir = f"chroma_db/{user_id}/{conversation_id}"
            self._memories[user_id][conversation_id] = ChatMemory(
                llm_config=self.llm_config,
                persist_directory=persist_dir,
            )
        return self._memories[user_id][conversation_id]

    def create_conversation(self, user_id: str, conversation_id: str, title: str | None = None) -> None:
        """Initialise metadata for a new conversation.

        A new :class:`Conversation` record is created and stored.  This
        should be called when a conversation is first started.
        """
        from ..models.conversation import Conversation

        if user_id not in self._conversations:
            self._conversations[user_id] = {}
        if conversation_id not in self._conversations[user_id]:
            self._conversations[user_id][conversation_id] = Conversation(
                conversation_id=conversation_id,
                user_id=user_id,
                title=title,
            )

    def add_message(self, user_id: str, conversation_id: str, message: "ChatMessage") -> None:
        """Append a message to the conversation metadata.

        This updates the stored :class:`Conversation` object with the
        new message.  If the conversation does not yet exist, it is
        created implicitly.
        """
        from ..models.conversation import Conversation
        from ..models.chat_message import ChatMessage

        if user_id not in self._conversations:
            self._conversations[user_id] = {}
        if conversation_id not in self._conversations[user_id]:
            # create conversation without title if missing
            self._conversations[user_id][conversation_id] = Conversation(
                conversation_id=conversation_id,
                user_id=user_id,
            )
        conv = self._conversations[user_id][conversation_id]
        conv.add_message(message)  # type: ignore[arg-type]

    def list_conversations(self, user_id: str) -> list["Conversation"]:
        """Return a list of conversations for a user.

        If the user has no conversations, an empty list is returned.
        """
        if user_id not in self._conversations:
            return []
        return list(self._conversations[user_id].values())

    def list_all_conversations(self) -> dict[str, list["Conversation"]]:
        """Return conversations grouped by user identifier."""
        from ..models.conversation import Conversation

        return {
            user_id: list(conversations.values())
            for user_id, conversations in self._conversations.items()
        }

    def get_conversation(self, user_id: str, conversation_id: str) -> "Conversation" | None:
        """Return metadata for a specific conversation or None if missing."""
        return self._conversations.get(user_id, {}).get(conversation_id)

    def delete_conversation(self, user_id: str, conversation_id: str) -> None:
        """Delete a conversation and its associated memory.

        Removes both the in‑memory metadata and the persisted vector store on
        disk (if present).  Errors during disk removal are ignored since
        they do not affect in-memory state.
        """
        # Remove memory instance
        if user_id in self._memories and conversation_id in self._memories[user_id]:
            del self._memories[user_id][conversation_id]
        # Remove metadata
        if user_id in self._conversations and conversation_id in self._conversations[user_id]:
            del self._conversations[user_id][conversation_id]
        # Remove persisted vectors from disk if they exist
        import os
        import shutil
        persist_dir = f"chroma_db/{user_id}/{conversation_id}"
        try:
            if os.path.isdir(persist_dir):
                shutil.rmtree(persist_dir)
        except Exception:
            # If deletion fails we log but do not raise
            pass

    def delete_all_conversations(self, user_id: str) -> None:
        """Clear all conversations and memories for a user."""
        # Delete each conversation directory
        for conv_id in list(self._memories.get(user_id, {}).keys()):
            self.delete_conversation(user_id, conv_id)
        # Clean up root user directory if empty
        import os
        import shutil
        user_dir = f"chroma_db/{user_id}"
        try:
            if os.path.isdir(user_dir) and not os.listdir(user_dir):
                shutil.rmtree(user_dir)
        except Exception:
            pass
