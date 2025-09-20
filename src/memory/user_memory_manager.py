"""Manage memory instances per user.

If the application needs to maintain separate session histories
for different users, this manager can be used to allocate and
retrieve ChatMemory objects keyed by a user identifier.  For now we
provide a basic implementation.
"""

from __future__ import annotations

from typing import Dict

from ..config.llm_config import LlmConfig
from .chat_memory import ChatMemory


class UserMemoryManager:
    """Manage per‑user and per‑session memory instances and metadata.

    The manager maintains separate memories for each session
    belonging to a user.  Conversation metadata (such as message
    counts and timestamps) is stored alongside the memory and can be
    queried or updated.  New sessions are automatically created
    on demand with unique identifiers.
    """

    def __init__(self, llm_config: LlmConfig) -> None:
        self.llm_config = llm_config
        # Mapping of user_id -> session_id -> ChatMemory
        self._memories: Dict[str, Dict[str, ChatMemory]] = {}
        # Mapping of user_id -> session_id -> Session metadata
        from ..models.conversation import Conversation

        self._sessions: Dict[str, Dict[str, Conversation]] = {}

    def get_memory(self, user_id: str, session_id: str) -> ChatMemory:
        """Retrieve or create a ChatMemory for a user's session.

        When a memory for the given ``user_id`` and ``session_id`` does
        not exist, a new one is created.  The memory will persist its
        data in a session‑specific directory (``chroma_db/<user>/<session>``)
        so that separate histories are isolated on disk.
        """
        if user_id not in self._memories:
            self._memories[user_id] = {}
        if session_id not in self._memories[user_id]:
            # Determine the persistence directory for this session
            persist_dir = f"chroma_db/{user_id}/{session_id}"
            self._memories[user_id][session_id] = ChatMemory(
                llm_config=self.llm_config,
                persist_directory=persist_dir,
            )
        return self._memories[user_id][session_id]

    def create_session(self, user_id: str, session_id: str, title: str | None = None) -> None:
        """Initialise metadata for a new session.

        A new :class:`Conversation` record is created and stored.  This
        should be called when a session is first started.
        """
        from ..models.conversation import Conversation

        if user_id not in self._sessions:
            self._sessions[user_id] = {}
        if session_id not in self._sessions[user_id]:
            self._sessions[user_id][session_id] = Conversation(
                session_id=session_id,
                user_id=user_id,
                title=title,
            )

    def add_message(self, user_id: str, session_id: str, message: "ChatMessage") -> None:
        """Append a message to the session metadata.

        This updates the stored :class:`Conversation` object with the
        new message.  If the session does not yet exist, it is
        created implicitly.
        """
        from ..models.conversation import Conversation
        from ..models.chat_message import ChatMessage

        if user_id not in self._sessions:
            self._sessions[user_id] = {}
        if session_id not in self._sessions[user_id]:
            # create session without title if missing
            self._sessions[user_id][session_id] = Conversation(
                session_id=session_id,
                user_id=user_id,
            )
        conv = self._sessions[user_id][session_id]
        conv.add_message(message)  # type: ignore[arg-type]

    def list_sessions(self, user_id: str) -> list["Conversation"]:
        """Return a list of sessions for a user.

        If the user has no sessions, an empty list is returned.
        """
        if user_id not in self._sessions:
            return []
        return list(self._sessions[user_id].values())

    def list_all_sessions(self) -> dict[str, list["Conversation"]]:
        """Return sessions grouped by user identifier."""
        from ..models.conversation import Conversation

        return {user_id: list(sessions.values()) for user_id, sessions in self._sessions.items()}

    def get_session(self, user_id: str, session_id: str) -> "Conversation" | None:
        """Return metadata for a specific session or None if missing."""
        return self._sessions.get(user_id, {}).get(session_id)

    def delete_session(self, user_id: str, session_id: str) -> None:
        """Delete a session and its associated memory.

        Removes both the in‑memory metadata and the persisted vector store on
        disk (if present).  Errors during disk removal are ignored since
        they do not affect in-memory state.
        """
        # Remove memory instance
        if user_id in self._memories and session_id in self._memories[user_id]:
            del self._memories[user_id][session_id]
        # Remove metadata
        if user_id in self._sessions and session_id in self._sessions[user_id]:
            del self._sessions[user_id][session_id]
        # Remove persisted vectors from disk if they exist
        import os
        import shutil
        persist_dir = f"chroma_db/{user_id}/{session_id}"
        try:
            if os.path.isdir(persist_dir):
                shutil.rmtree(persist_dir)
        except Exception:
            # If deletion fails we log but do not raise
            pass

    def delete_all_sessions(self, user_id: str) -> None:
        """Clear all sessions and memories for a user."""
        # Delete each session directory
        for session_id in list(self._memories.get(user_id, {}).keys()):
            self.delete_session(user_id, session_id)
        # Clean up root user directory if empty
        import os
        import shutil
        user_dir = f"chroma_db/{user_id}"
        try:
            if os.path.isdir(user_dir) and not os.listdir(user_dir):
                shutil.rmtree(user_dir)
        except Exception:
            pass
