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
from ..models.conversation import Conversation
import uuid
from ..models.dashboard import ConversationStats, DashboardData, UserConversationStats
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

    def chat(self, chat_request: ChatRequest) -> ChatResponse:
        """Generate a reply to a chat request.

        This method handles multi‑user, multi‑conversation contexts.  It
        generates new user and conversation identifiers when they are not
        provided, retrieves or initialises the appropriate memory, asks
        the LLM for a response, persists the interaction and updates
        conversation metadata.  A :class:`ChatResponse` containing the
        answer and identifiers is returned.

        Parameters
        ----------
        chat_request: ChatRequest
            The incoming chat request containing the message and optional
            user and conversation identifiers.

        Returns
        -------
        ChatResponse
            A response with the assistant's answer and identifiers.

        Raises
        ------
        ChatError
            If any exception occurs during processing.
        """
        logger.info("Processing chat for request: {}", chat_request)
        try:
            # Determine or generate user and conversation identifiers
            user_id = chat_request.user_id or str(uuid.uuid4())
            conversation_id = chat_request.conversation_id or str(uuid.uuid4())

            # Retrieve memory for this user and conversation
            chat_memory = self.memory_service.get_memory(user_id, conversation_id)
            # Generate an answer using the LLM and current memory
            answer_text = self.llm_service.generate(
                chat_request.message,
                memory=chat_memory,
                user_id=user_id,
            )
            # Persist the interaction (both question and answer) and update metadata
            self.memory_service.save_interaction(
                user_id=user_id,
                conversation_id=conversation_id,
                question=chat_request.message,
                answer=answer_text,
            )
            # Set conversation title on first message if missing
            conversation = self.memory_service.get_conversation(user_id, conversation_id)
            if conversation and conversation.title is None and conversation.message_count == 2:
                # Use the user's first message as title, truncated to 60 chars
                title = chat_request.message.strip()
                conversation.title = title[:60]
            # Return response
            return ChatResponse(user_id=user_id, conversation_id=conversation_id, answer=answer_text)
        except Exception as exc:
            logger.exception("LLM processing failed")
            raise ChatError("LLM processing failed") from exc

    # ------------------------------------------------------------------
    # Conversation management API

    def list_conversations(self, user_id: str) -> list[Conversation]:
        """Return all conversations for a user."""
        return self.memory_service.list_conversations(user_id)

    def get_conversation(self, user_id: str, conversation_id: str) -> Conversation | None:
        """Return a specific conversation for a user if it exists."""
        return self.memory_service.get_conversation(user_id, conversation_id)

    def delete_conversation(self, user_id: str, conversation_id: str) -> None:
        """Delete a conversation for a user."""
        self.memory_service.delete_conversation(user_id, conversation_id)

    def delete_all_conversations(self, user_id: str) -> None:
        """Delete all conversations for a user."""
        self.memory_service.delete_all_conversations(user_id)

    # ------------------------------------------------------------------
    # Health and service info

    def get_dashboard_data(self) -> DashboardData:
        """Return analytics for all users and conversations."""
        conversations_by_user = self.memory_service.list_all_conversations()
        users: list[UserConversationStats] = []
        total_tokens = 0
        total_conversations = 0

        for user_id, conversations in conversations_by_user.items():
            conversation_stats: list[ConversationStats] = []
            user_token_sum = 0

            for conversation in conversations:
                tokens_used = self.llm_service.count_tokens(conversation.messages)
                user_token_sum += tokens_used
                total_conversations += 1
                conversation_stats.append(
                    ConversationStats(
                        conversation_id=conversation.conversation_id,
                        title=conversation.title,
                        message_count=conversation.message_count,
                        created_at=conversation.created_at,
                        updated_at=conversation.updated_at,
                        tokens_used=tokens_used,
                    )
                )

            total_tokens += user_token_sum
            users.append(
                UserConversationStats(
                    user_id=user_id,
                    conversation_count=len(conversations),
                    total_tokens=user_token_sum,
                    conversations=conversation_stats,
                )
            )

        return DashboardData(
            total_users=len(conversations_by_user),
            total_conversations=total_conversations,
            total_tokens=total_tokens,
            users=users,
        )

    def health_check(self) -> dict[str, str]:
        """Return a simple health status for the chat service.

        This method can be used by higher-level controllers to verify
        that the service and its dependencies are reachable.  It
        currently returns a static status but could be extended to
        perform checks against the LLM and memory backends.
        """
        try:
            # We could add more sophisticated checks here (e.g. ping the LLM)
            return {"status": "ok"}
        except Exception:
            return {"status": "unhealthy"}

    def get_service_info(self) -> dict[str, object]:
        """Return basic information about the chat service.

        Provides configuration details such as the LLM model name and
        tuning parameters.  This can be useful for UI components to
        display the current backend configuration.
        """
        return {
            "model": self.llm_config.model,
            "temperature": self.llm_config.temperature,
            "max_tokens": self.llm_config.max_tokens,
            "timeout": self.llm_config.timeout,
        }


@lru_cache()
def get_chat_service() -> ChatService:
    """Dependency injector for ChatService instances.

    FastAPI will call this function to obtain a singleton
    ChatService.  The lru_cache decorator ensures only one
    instance exists.
    """
    return ChatService()
