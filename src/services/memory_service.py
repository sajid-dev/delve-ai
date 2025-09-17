"""Service for managing conversational memory.

This service wraps the underlying memory implementation provided by
LangChain and ChromaDB.  It offers a simple API to persist and
retrieve context for conversations.  A ``UserMemoryManager`` can be
introduced to scope memory per user if required.
"""

from __future__ import annotations

from loguru import logger

from ..config.llm_config import LlmConfig, get_llm_config
from ..memory.chat_memory import ChatMemory


class MemoryService:
    """A high‑level interface over a chat memory implementation."""

    def __init__(self, llm_config: LlmConfig | None = None) -> None:
        # Load LLM configuration if none provided
        self.llm_config = llm_config or get_llm_config()
        # For now we maintain a single ChatMemory instance.  A future
        # improvement could be to delegate to a UserMemoryManager to provide
        # per‑user memory isolation.
        self.chat_memory = ChatMemory(llm_config=self.llm_config)

    def save_interaction(self, question: str, answer: str) -> None:
        """Persist a question/answer pair into the memory.

        Any exception raised by the underlying memory implementation is
        caught and re-raised as a ChatError.  This protects callers from
        unexpected errors.
        """
        logger.debug("Saving interaction to memory: Q=%r A=%r", question, answer)
        try:
            self.chat_memory.save_interaction(question, answer)
        except Exception as exc:
            logger.exception("Failed to save interaction to memory")
            from ..utils.error_handler import ChatError

            raise ChatError("Failed to save interaction") from exc

    def get_memory(self) -> ChatMemory:
        """Expose the underlying chat memory object."""
        return self.chat_memory