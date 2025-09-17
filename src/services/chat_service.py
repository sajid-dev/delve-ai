"""Orchestration service combining memory and LLM interactions.

The ChatService receives user prompts, retrieves context from the
memory, asks the LLM for a response and then persists the question
and answer back into the memory.  It centralises error handling so
controllers can remain thin.
"""

from __future__ import annotations

from functools import lru_cache
from loguru import logger

from ..config.llm_config import LlmConfig, get_llm_config
from ..config.app_config import AppConfig, get_app_config
from ..models.chat_response import ChatResponse
from ..models.chat_request import ChatRequest
from ..utils.error_handler import ChatError
from .memory_service import MemoryService
from .llm_service import LLMService


class ChatService:
    """Coordinates memory retrieval and LLM generation.

    The service composes a memory service and an LLM service.  It uses
    dedicated configuration objects for the application and LLM so that
    environment variables are loaded in a modular fashion.
    """

    def __init__(self, llm_config: LlmConfig | None = None, app_config: AppConfig | None = None) -> None:
        # Load configurations if not provided
        self.llm_config = llm_config or get_llm_config()
        self.app_config = app_config or get_app_config()
        # Initialise memory and LLM services using the LLM configuration
        self.memory_service = MemoryService(llm_config=self.llm_config)
        self.llm_service = LLMService(llm_config=self.llm_config)

    def chat(self, prompt: str) -> str:
        """Generate a reply to the user prompt.

        This method retrieves prior conversation context via the memory
        service, invokes the LLM for a new answer and persists the
        interaction back into memory.

        Parameters
        ----------
        prompt: str
            The user's message to send to the LLM.

        Returns
        -------
        str
            The assistant's reply.

        Raises
        ------
        ChatError
            If an exception occurs while querying the LLM.
        """
        logger.info("Processing chat for prompt: %s", prompt)
        try:
            # Retrieve underlying memory
            chat_memory = self.memory_service.get_memory()
            # Generate answer using LLM and current memory
            answer = self.llm_service.generate(prompt, memory=chat_memory)
            # Save the interaction for future context
            self.memory_service.save_interaction(prompt, answer)
            return answer
        except Exception as exc:
            logger.exception("LLM processing failed")
            # Wrap the error in our custom exception
            raise ChatError("LLM processing failed") from exc


@lru_cache()
def get_chat_service() -> ChatService:
    """Dependency injector for ChatService instances.

    FastAPI will call this function to obtain a singleton
    ChatService.  The lru_cache decorator ensures only one
    instance exists.
    """
    return ChatService()