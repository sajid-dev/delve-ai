"""Orchestration service combining memory and LLM interactions.

The ChatService receives user prompts, retrieves context from the
memory, asks the LLM for a response and then persists the question
and answer back into the memory.  It centralises error handling so
controllers can remain thin.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from functools import lru_cache
from loguru import logger

from ..config.llm_config import LlmConfig, get_llm_config
from ..config.app_config import AppConfig, get_app_config
from ..models.chat_response import ChatResponse
from ..models.chat_request import ChatRequest
from ..models.conversation import Conversation
from ..models.enums import MessageContentType, MessageRole
import uuid
from ..models.dashboard import ConversationStats, DashboardData, UserConversationStats
from ..utils.error_handler import ChatError
from ..utils.structured_output import analyse_content, build_response_payload
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

        This method handles multi-user, multi-session contexts.  It
        generates new user and session identifiers when they are not
        provided, retrieves or initialises the appropriate memory, asks
        the LLM for a response, persists the interaction and updates
        session metadata.  A :class:`ChatResponse` containing the
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
            # Determine or generate user and session identifiers
            user_id = chat_request.user_id or str(uuid.uuid4())
            session_id = chat_request.session_id or str(uuid.uuid4())
            chat_request.session_id = session_id

            # Retrieve memory for this user and session
            chat_memory = self.memory_service.get_memory(user_id, session_id)
            # Generate an answer using the LLM and current memory
            answer_raw = self.llm_service.generate(
                chat_request.message,
                memory=chat_memory,
                user_id=user_id,
                session_id=session_id,
            )
            analysis = analyse_content(answer_raw)
            structured_payload = build_response_payload(analysis)
            answer_components = structured_payload.get("components") if isinstance(structured_payload, dict) else None
            # Persist the interaction (both question and answer) and update metadata
            self.memory_service.save_interaction(
                user_id=user_id,
                session_id=session_id,
                question=chat_request.message,
                answer=analysis.text,
                question_type=MessageContentType.TEXT,
                answer_type=analysis.content_type,
                answer_components=answer_components,
            )
            # Set session title on first message if missing
            session_meta = self.memory_service.get_session(user_id, session_id)
            if session_meta and session_meta.title is None and session_meta.message_count == 2:
                # Use the user's first message as title, truncated to 60 chars
                title = chat_request.message.strip()
                session_meta.title = title[:60]
                self.memory_service.persist_session(user_id, session_id)
            # Return response
            return ChatResponse(
                user_id=user_id,
                session_id=session_id,
                data=structured_payload,
            )
        except Exception as exc:
            logger.exception("LLM processing failed")
            raise ChatError("LLM processing failed") from exc

    # ------------------------------------------------------------------
    # Session management API

    def list_sessions(self, user_id: str) -> list[Conversation]:
        """Return all sessions for a user."""
        return self.memory_service.list_sessions(user_id)

    def get_session(self, user_id: str, session_id: str) -> Conversation | None:
        """Return a specific session for a user if it exists."""
        return self.memory_service.get_session(user_id, session_id)

    def delete_session(self, user_id: str, session_id: str) -> None:
        """Delete a session for a user."""
        self.memory_service.delete_session(user_id, session_id)

    def delete_all_sessions(self, user_id: str) -> None:
        """Delete all sessions for a user."""
        self.memory_service.delete_all_sessions(user_id)

    def delete_all_sessions_global(self) -> None:
        """Delete every session across all users."""
        logger.info("Deleting all sessions across all users")
        self.memory_service.delete_everything()

    # ------------------------------------------------------------------
    # Health and service info

    def get_dashboard_data(self) -> DashboardData:
        """Return analytics for all users and sessions."""
        sessions_by_user = self.memory_service.list_all_sessions()
        users: list[UserConversationStats] = []
        total_tokens = 0
        total_sessions = 0
        active_users = 0
        activity_threshold = datetime.utcnow() - timedelta(hours=24)

        for user_id, sessions in sessions_by_user.items():
            session_stats: list[ConversationStats] = []
            user_token_sum = 0
            last_active: datetime | None = None

            for session in sessions:
                tokens_used = self.llm_service.count_tokens(session.messages)
                latest_answer: dict[str, object] | None = None
                for message in reversed(session.messages):
                    if message.role == MessageRole.ASSISTANT:
                        latest_answer = {
                            "session_id": session.session_id,
                            "timestamp": message.timestamp,
                        }
                        break
                user_token_sum += tokens_used
                total_sessions += 1
                if last_active is None or session.updated_at > last_active:
                    last_active = session.updated_at
                session_stats.append(
                    ConversationStats(
                        session_id=session.session_id,
                        title=session.title,
                        message_count=session.message_count,
                        created_at=session.created_at,
                        updated_at=session.updated_at,
                        tokens_used=tokens_used,
                        latest_answer=latest_answer,
                    )
                )

            total_tokens += user_token_sum
            is_active = bool(last_active and last_active >= activity_threshold)
            if is_active:
                active_users += 1
            users.append(
                UserConversationStats(
                    user_id=user_id,
                    session_count=len(sessions),
                    total_tokens=user_token_sum,
                    last_active=last_active,
                    is_active=is_active,
                    sessions=session_stats,
                )
            )

        return DashboardData(
            total_users=len(sessions_by_user),
            active_users=active_users,
            total_sessions=total_sessions,
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
