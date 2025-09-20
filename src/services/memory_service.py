"""Service for managing session memory.

This service wraps the underlying memory implementation provided by
LangChain and ChromaDB.  It offers a simple API to persist and
retrieve context for sessions.  A ``UserMemoryManager`` can be
introduced to scope memory per user if required.
"""

from __future__ import annotations

from loguru import logger

from ..config.llm_config import LlmConfig, get_llm_config
from ..memory.chat_memory import ChatMemory
from ..memory.user_memory_manager import UserMemoryManager
from ..models.conversation import Conversation
from ..models.chat_message import ChatMessage
from ..models.enums import MessageContentType, MessageRole


class MemoryService:
    """High‑level interface over per‑user, per-session memory.

    This service delegates to :class:`UserMemoryManager` to provide
    isolated memories and session metadata for each user and
    session.  It exposes methods to obtain a memory, persist
    interactions and manage sessions.
    """

    def __init__(self, llm_config: LlmConfig | None = None) -> None:
        # Load LLM configuration if none provided
        self.llm_config = llm_config or get_llm_config()
        # Initialise the user memory manager
        self._manager = UserMemoryManager(llm_config=self.llm_config)

    def get_memory(self, user_id: str, session_id: str) -> ChatMemory:
        """Return a ChatMemory scoped to a user's session."""
        return self._manager.get_memory(user_id, session_id)

    def save_interaction(
        self,
        user_id: str,
        session_id: str,
        question: str,
        answer: str,
        *,
        question_type: MessageContentType = MessageContentType.TEXT,
        answer_type: MessageContentType = MessageContentType.TEXT,
        question_structured: object | None = None,
        answer_structured: object | None = None,
    ) -> None:
        """Persist a question/answer pair into a user's session memory.

        A corresponding chat message entry is added to the session
        metadata for both the user and assistant.  Timestamps are
        generated automatically.
        """
        logger.debug(
            "Saving interaction to memory: user={} session={} Q={!r} A={!r}",
            user_id,
            session_id,
            question,
            answer,
        )
        try:
            memory = self.get_memory(user_id, session_id)
            # Save to vector store (only the assistant side is persisted since
            # questions and answers are passed separately below via embeddings)
            memory.save_interaction(question, answer)
            # Update session metadata with explicit chat messages
            from datetime import datetime
            # Create ChatMessage objects with timestamps
            user_msg = ChatMessage(
                role=MessageRole.USER,
                content=question,
                content_type=question_type,
                timestamp=datetime.utcnow().isoformat(),
                structured_data=question_structured,
            )
            assistant_msg = ChatMessage(
                role=MessageRole.ASSISTANT,
                content=answer,
                content_type=answer_type,
                timestamp=datetime.utcnow().isoformat(),
                structured_data=answer_structured,
            )
            # Create session record if needed
            self._manager.create_session(user_id, session_id)
            # Append messages to session metadata
            self._manager.add_message(user_id, session_id, user_msg)
            self._manager.add_message(user_id, session_id, assistant_msg)
        except Exception as exc:
            logger.exception("Failed to save interaction to memory")
            from ..utils.error_handler import ChatError
            raise ChatError("Failed to save interaction") from exc

    def list_sessions(self, user_id: str) -> list[Conversation]:
        """Return a list of all sessions for a user."""
        return self._manager.list_sessions(user_id)

    def list_all_sessions(self) -> dict[str, list[Conversation]]:
        """Return all sessions grouped by user identifier."""
        return self._manager.list_all_sessions()

    def get_session(self, user_id: str, session_id: str) -> Conversation | None:
        """Return a single session metadata record."""
        return self._manager.get_session(user_id, session_id)

    def delete_session(self, user_id: str, session_id: str) -> None:
        """Delete a session and its memory."""
        self._manager.delete_session(user_id, session_id)

    def delete_all_sessions(self, user_id: str) -> None:
        """Delete all sessions for a user."""
        self._manager.delete_all_sessions(user_id)

    def delete_everything(self) -> None:
        """Delete all sessions for all users."""
        logger.info("Deleting all sessions across all users")
        all_users = list(self._manager.list_all_sessions().keys())
        for user_id in all_users:
            self.delete_all_sessions(user_id)
