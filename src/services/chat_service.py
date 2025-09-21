"""Orchestration service combining memory and LLM interactions.

The ChatService receives user prompts, retrieves context from the
memory, asks the LLM for a response and then persists the question
and answer back into the memory.  It centralises error handling so
controllers can remain thin.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
import json
import re
from typing import Any

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
from .memory_service import MemoryService
from .llm_service import LLMService


@dataclass
class ContentAnalysis:
    """Lightweight representation of parsed assistant output."""

    content_type: MessageContentType
    structured_data: Any | None
    text: str


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
            analysis = self._analyse_content(answer_raw)
            structured_payload = self._build_response_payload(analysis)
            # Persist the interaction (both question and answer) and update metadata
            self.memory_service.save_interaction(
                user_id=user_id,
                session_id=session_id,
                question=chat_request.message,
                answer=analysis.text,
                question_type=MessageContentType.TEXT,
                answer_type=analysis.content_type,
                answer_structured=structured_payload,
            )
            # Set session title on first message if missing
            session_meta = self.memory_service.get_session(user_id, session_id)
            if session_meta and session_meta.title is None and session_meta.message_count == 2:
                # Use the user's first message as title, truncated to 60 chars
                title = chat_request.message.strip()
                session_meta.title = title[:60]
            # Return response
            return ChatResponse(
                user_id=user_id,
                session_id=session_id,
                data=structured_payload,
            )
        except Exception as exc:
            logger.exception("LLM processing failed")
            raise ChatError("LLM processing failed") from exc

    def _analyse_content(self, content: str) -> ContentAnalysis:
        """Classify assistant output and provide optional structured data."""
        stripped = content.strip()
        if not stripped:
            return ContentAnalysis(MessageContentType.TEXT, None, content)

        image_payload = self._extract_image_payload(stripped)
        if image_payload is not None:
            return ContentAnalysis(MessageContentType.IMAGE, image_payload, content)

        table_payload = self._parse_table(stripped)
        if table_payload is not None:
            return ContentAnalysis(MessageContentType.TABLE, table_payload, content)

        list_payload = self._parse_list(stripped)
        if list_payload is not None:
            return ContentAnalysis(MessageContentType.LIST, list_payload, content)

        parsed_json = self._parse_json(stripped)
        if parsed_json is not None:
            if self._looks_like_chart_spec(parsed_json):
                return ContentAnalysis(MessageContentType.CHART, parsed_json, content)
            return ContentAnalysis(MessageContentType.JSON, parsed_json, content)

        code_payload = self._parse_code_block(content)
        if code_payload is not None:
            return ContentAnalysis(MessageContentType.CODE, code_payload, content)

        if self._looks_like_html(stripped):
            return ContentAnalysis(MessageContentType.HTML, None, content)

        if self._looks_like_markdown(stripped):
            return ContentAnalysis(MessageContentType.MARKDOWN, None, content)

        return ContentAnalysis(MessageContentType.TEXT, None, content)

    def _build_response_payload(self, analysis: ContentAnalysis) -> dict[str, Any]:
        """Normalise analysis output into a frontend-friendly structure."""
        content: dict[str, Any]
        if analysis.structured_data is not None:
            content = analysis.structured_data
        else:
            content = {"text": analysis.text}
        return {
            "data_type": analysis.content_type.value,
            "content": content,
        }

    @staticmethod
    def _extract_image_payload(content: str) -> dict[str, str] | None:
        """Return image details when the content resembles an image reference."""
        direct_url = re.match(r"^https?://\S+\.(png|jpe?g|gif|webp|svg)$", content, re.IGNORECASE)
        if direct_url:
            return {"url": content.strip(), "alt": ""}

        md_match = re.match(r"^!\[([^\]]*)\]\((https?://[^\s)]+)\)$", content.strip())
        if md_match:
            alt_text = md_match.group(1).strip()
            return {"url": md_match.group(2).strip(), "alt": alt_text}
        return None

    def _parse_table(self, content: str) -> dict[str, Any] | None:
        """Attempt to parse markdown or HTML table payloads."""
        markdown_table = self._parse_markdown_table(content)
        if markdown_table is not None:
            return markdown_table
        if "<table" in content.lower() and "</table>" in content.lower():
            return {"html": content}
        if self._looks_like_table(content):
            return {"raw": content}
        return None

    def _parse_markdown_table(self, content: str) -> dict[str, Any] | None:
        """Parse a Markdown table into headers/rows if present."""
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        if len(lines) < 2:
            return None

        for idx in range(len(lines) - 1):
            header_line = lines[idx]
            separator_line = lines[idx + 1]
            if "|" not in header_line:
                continue
            if not re.match(r"^\|?\s*:?-{3,}.*", separator_line):
                continue

            headers = [cell.strip() for cell in header_line.strip("|").split("|")]
            rows: list[dict[str, str]] = []
            for row_line in lines[idx + 2 :]:
                if "|" not in row_line:
                    break
                cells = [cell.strip() for cell in row_line.strip("|").split("|")]
                if len(cells) != len(headers):
                    continue
                rows.append(dict(zip(headers, cells)))

            if rows:
                return {"headers": headers, "rows": rows}
        return None

    def _parse_list(self, content: str) -> dict[str, Any] | None:
        """Detect ordered/unordered list structures and return items."""
        ordered_items = self._parse_ordered_list(content)
        if ordered_items:
            return {"ordered": True, "items": self._normalise_list_items(ordered_items)}

        unordered_items = self._parse_unordered_list(content)
        if unordered_items:
            return {"ordered": False, "items": self._normalise_list_items(unordered_items)}

        return None

    @staticmethod
    def _parse_ordered_list(content: str) -> list[str]:
        """Return ordered list items if the content resembles a numbered list."""
        items: list[str] = []
        current: list[str] | None = None
        for raw_line in content.splitlines():
            line = raw_line.rstrip()
            stripped = line.strip()
            if not stripped and current is not None:
                # blank line inside an item, keep it as separator
                current.append("")
                continue

            match = re.match(r"^(\d+)[\.)]\s+(.*)", stripped)
            if match:
                if current is not None:
                    items.append("\n".join(part for part in current if part is not None).strip())
                current = [match.group(2).strip()]
                continue

            if current is not None:
                if re.match(r"^#{1,6}\s", stripped):
                    items.append("\n".join(part for part in current if part is not None).strip())
                    current = None
                    continue
                bullet = re.match(r"^[-*+]\s+(.*)", stripped)
                if bullet:
                    current.append(f"- {bullet.group(1).strip()}")
                else:
                    current.append(stripped)

        if current is not None:
            items.append("\n".join(part for part in current if part is not None).strip())
        return [item for item in items if item]

    @staticmethod
    def _parse_unordered_list(content: str) -> list[str]:
        """Return bullet list items detected in the content."""
        items: list[str] = []
        current: list[str] | None = None
        for raw_line in content.splitlines():
            line = raw_line.rstrip()
            stripped = line.strip()
            if not stripped and current is not None:
                current.append("")
                continue

            match = re.match(r"^[-*+]\s+(.*)", stripped)
            if match:
                if current is not None:
                    items.append("\n".join(part for part in current if part is not None).strip())
                current = [match.group(1).strip()]
                continue

            if current is not None:
                if re.match(r"^#{1,6}\s", stripped):
                    items.append("\n".join(part for part in current if part is not None).strip())
                    current = None
                    continue
                nested = re.match(r"^(?:[-*+]|\d+[\.)])\s+(.*)", stripped)
                if nested:
                    current.append(f"- {nested.group(1).strip()}")
                else:
                    current.append(stripped)

        if current is not None:
            items.append("\n".join(part for part in current if part is not None).strip())
        return [item for item in items if item]

    @staticmethod
    def _normalise_list_items(items: list[str]) -> list[dict[str, Any]]:
        """Convert raw markdown list entries into structured payloads."""

        def extract_code_blocks(text: str) -> tuple[str, list[dict[str, str]]]:
            code_blocks: list[dict[str, str]] = []

            def _collect(match: re.Match[str]) -> str:
                language = (match.group(1) or "text").strip()
                code = match.group(2).strip()
                code_blocks.append({"language": language, "code": code})
                return ""

            cleaned = re.sub(r"```(\w+)?\n(.*?)```", _collect, text, flags=re.DOTALL)
            return cleaned.strip(), code_blocks

        normalised: list[dict[str, Any]] = []
        for raw in items:
            body, code_blocks = extract_code_blocks(raw)

            title: str | None = None
            description = body.strip()

            heading_match = re.match(r"^\*\*(.+?)\*\*:?\s*(.*)", description, re.DOTALL)
            if heading_match:
                title = heading_match.group(1).strip()
                description = heading_match.group(2).strip()

            bullet_points: list[str] = []
            remaining_lines: list[str] = []
            for line in description.splitlines():
                stripped = line.strip()
                bullet_match = re.match(r"^-\s+(.*)", stripped)
                if bullet_match:
                    bullet_points.append(bullet_match.group(1).strip())
                else:
                    remaining_lines.append(line)

            clean_description = "\n".join(line for line in remaining_lines if line.strip()).strip()

            entry: dict[str, Any] = {
                "raw": raw,
                "title": title or "",
                "description": clean_description or "",
                "bullets": bullet_points,
                "code_blocks": code_blocks,
            }
            normalised.append(entry)

        return normalised

    @staticmethod
    def _parse_code_block(content: str) -> dict[str, Any] | None:
        """Extract language, code, and optionally parsed data from fenced blocks."""
        match = re.search(r"```(\w+)?\n(.*?)```", content, re.DOTALL)
        if not match:
            return None
        language = (match.group(1) or "text").strip()
        code = match.group(2).strip()
        payload: dict[str, Any] = {"language": language, "code": code}
        if language.lower() in {"json", "javascript"}:
            parsed = ChatService._parse_json(code)
            if parsed is None:
                parsed = ChatService._parse_json(code.strip("`;"))
            if parsed is not None:
                payload["parsed"] = parsed
        return payload

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
            try:
                parsed = ast.literal_eval(content)
            except (ValueError, SyntaxError):
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
                            "content": message.content,
                            "content_type": message.content_type.value,
                            "structured_data": message.structured_data,
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
