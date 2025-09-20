"""Orchestration service combining memory and LLM interactions.

The ChatService receives user prompts, retrieves context from the
memory, asks the LLM for a response and then persists the question
and answer back into the memory.  It centralises error handling so
controllers can remain thin.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from functools import lru_cache
import json
import re
from loguru import logger

from ..config.llm_config import LlmConfig, get_llm_config
from ..config.app_config import AppConfig, get_app_config
from ..models.chat_response import ChatResponse
from ..models.chat_request import ChatRequest
from ..models.conversation import Conversation
from ..models.enums import MessageContentType
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

        This method handles multi‑user, multi‑session contexts.  It
        generates new user and session identifiers when they are not
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
            # Determine or generate user and session identifiers
            user_id = chat_request.user_id or str(uuid.uuid4())
            session_id = chat_request.session_id or str(uuid.uuid4())
            chat_request.session_id = session_id

            # Retrieve memory for this user and conversation
            chat_memory = self.memory_service.get_memory(user_id, session_id)
            # Generate an answer using the LLM and current memory
            answer_text = self.llm_service.generate(
                chat_request.message,
                memory=chat_memory,
                user_id=user_id,
                session_id=session_id,
            )
            answer_type = self._detect_content_type(answer_text)
            # Persist the interaction (both question and answer) and update metadata
            self.memory_service.save_interaction(
                user_id=user_id,
                conversation_id=session_id,
                question=chat_request.message,
                answer=answer_text,
                question_type=MessageContentType.TEXT,
                answer_type=answer_type,
            )
            # Set conversation title on first message if missing
            conversation = self.memory_service.get_conversation(user_id, session_id)
            if conversation and conversation.title is None and conversation.message_count == 2:
                # Use the user's first message as title, truncated to 60 chars
                title = chat_request.message.strip()
                conversation.title = title[:60]
            # Return response
            return ChatResponse(
                user_id=user_id,
                session_id=session_id,
                answer=answer_text,
                content_type=answer_type,
            )
        except Exception as exc:
            logger.exception("LLM processing failed")
            raise ChatError("LLM processing failed") from exc

    def _detect_content_type(self, content: str) -> MessageContentType:
        """Classify assistant output for downstream rendering."""
        stripped = content.strip()
        if not stripped:
            return MessageContentType.TEXT
        if self._looks_like_image(stripped):
            return MessageContentType.IMAGE
        if self._looks_like_table(stripped):
            return MessageContentType.TABLE
        parsed_json = self._parse_json(stripped)
        if parsed_json is not None:
            if self._looks_like_chart_spec(parsed_json):
                return MessageContentType.CHART
            return MessageContentType.JSON
        if self._looks_like_code(stripped):
            return MessageContentType.CODE
        if self._looks_like_html(stripped):
            return MessageContentType.HTML
        if self._looks_like_markdown(stripped):
            return MessageContentType.MARKDOWN
        return MessageContentType.TEXT

    @staticmethod
    def _looks_like_image(content: str) -> bool:
        """Return True when the content resembles an image payload."""
        # Direct image URL
        if re.match(r"^https?://\S+\.(png|jpe?g|gif|webp|svg)$", content, re.IGNORECASE):
            return True
        # Markdown image syntax ![alt](url)
        md_match = re.match(r"^!\[[^\]]*\]\((https?://[^\s)]+)\)$", content, re.IGNORECASE)
        if md_match and re.match(r"^https?://\S+\.(png|jpe?g|gif|webp|svg)$", md_match.group(1), re.IGNORECASE):
            return True
        return False

    @staticmethod
    def _looks_like_table(content: str) -> bool:
        """Return True when the content matches a table representation."""
        lowered = content.lower()
        if "<table" in lowered and "</table>" in lowered:
            return True
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        for idx in range(len(lines) - 1):
            header = lines[idx]
            separator = lines[idx + 1]
            if header.count("|") >= 2 and re.match(r"^\|?\s*:?-{3,}.*", separator):
                return True
        return False

    @staticmethod
    def _parse_json(content: str) -> object | None:
        """Return parsed JSON when the content is a valid payload, else None."""
        try:
            parsed = json.loads(content)
        except ValueError:
            return None
        return parsed if isinstance(parsed, (dict, list)) else None

    @staticmethod
    def _looks_like_chart_spec(payload: object) -> bool:
        """Heuristically determine whether parsed JSON resembles a chart spec."""
        if not isinstance(payload, dict):
            return False

        # Common structures (e.g. Chart.js, Vega/Vega-Lite style)
        chart_type = payload.get("type") or payload.get("chartType")
        data = payload.get("data") or payload.get("datasets") or payload.get("series")

        if isinstance(chart_type, str) and data:
            return True

        # Vega-Lite style specs nest data/mark/encoding keys
        if {"mark", "encoding"}.issubset(payload.keys()):
            return True

        # Look for explicit chart keywords to reduce false positives
        chart_keys = {"datasets", "series", "axes", "scales"}
        if any(key in payload for key in chart_keys):
            return True

        return False

    @staticmethod
    def _looks_like_code(content: str) -> bool:
        """Return True when content resembles a code block."""
        if "```" in content:
            return True
        code_lines = [line for line in content.splitlines() if line.strip()]
        if not code_lines:
            return False
        indented = sum(1 for line in code_lines if line.startswith(("    ", "\t")))
        return indented >= max(1, len(code_lines) // 2)

    @staticmethod
    def _looks_like_html(content: str) -> bool:
        """Return True if the content appears to contain generic HTML."""
        return bool(re.search(r"<[^>]+>", content) and re.search(r"</[^>]+>", content))

    @staticmethod
    def _looks_like_markdown(content: str) -> bool:
        """Return True when the content contains common Markdown features."""
        lines = content.splitlines()
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(("# ", "## ", "### ", "- ", "* ", "> ", "1. ")):
                return True
        if re.search(r"\[[^\]]+\]\([^)]+\)", content):
            return True
        if "```" in content:
            return True
        return False

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
        active_users = 0
        activity_threshold = datetime.utcnow() - timedelta(hours=24)

        for user_id, conversations in conversations_by_user.items():
            conversation_stats: list[ConversationStats] = []
            user_token_sum = 0
            last_active: datetime | None = None

            for conversation in conversations:
                tokens_used = self.llm_service.count_tokens(conversation.messages)
                user_token_sum += tokens_used
                total_conversations += 1
                if last_active is None or conversation.updated_at > last_active:
                    last_active = conversation.updated_at
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
            is_active = bool(last_active and last_active >= activity_threshold)
            if is_active:
                active_users += 1
            users.append(
                UserConversationStats(
                    user_id=user_id,
                    conversation_count=len(conversations),
                    total_tokens=user_token_sum,
                    last_active=last_active,
                    is_active=is_active,
                    conversations=conversation_stats,
                )
            )

        return DashboardData(
            total_users=len(conversations_by_user),
            active_users=active_users,
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
