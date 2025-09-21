"""Manage memory instances per user.

If the application needs to maintain separate session histories
for different users, this manager can be used to allocate and
retrieve ChatMemory objects keyed by a user identifier.  For now we
provide a basic implementation.
"""

from __future__ import annotations

from pathlib import Path
import json
import sqlite3
from datetime import datetime
from typing import Any, Dict

from ..config.llm_config import LlmConfig
from .chat_memory import ChatMemory
from ..models.conversation import Conversation
from ..models.chat_message import ChatMessage
from ..models.enums import MessageContentType, MessageRole
from loguru import logger


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
        self._sessions: Dict[str, Dict[str, Conversation]] = {}
        self._persist_root = Path("chroma_db")
        self._metadata_filename = "metadata.json"
        self._load_existing_sessions()

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
            persist_dir = self._session_directory(user_id, session_id)
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
        if user_id not in self._sessions:
            self._sessions[user_id] = {}
        if session_id not in self._sessions[user_id]:
            self._sessions[user_id][session_id] = Conversation(
                session_id=session_id,
                user_id=user_id,
                title=title,
            )
            self._persist_session(user_id, session_id)

    def add_message(self, user_id: str, session_id: str, message: "ChatMessage") -> None:
        """Append a message to the session metadata.

        This updates the stored :class:`Conversation` object with the
        new message.  If the session does not yet exist, it is
        created implicitly.
        """
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
        self._persist_session(user_id, session_id)

    def list_sessions(self, user_id: str) -> list["Conversation"]:
        """Return a list of sessions for a user.

        If the user has no sessions, an empty list is returned.
        """
        if user_id not in self._sessions:
            return []
        return list(self._sessions[user_id].values())

    def list_all_sessions(self) -> dict[str, list["Conversation"]]:
        """Return sessions grouped by user identifier."""
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
        metadata_path = self._metadata_path(user_id, session_id)
        try:
            if metadata_path.exists():
                metadata_path.unlink()
        except Exception:
            logger.warning("Failed to delete session metadata at {}", metadata_path)
        # Remove persisted vectors from disk if they exist
        import os
        import shutil
        persist_dir = self._session_directory(user_id, session_id)
        try:
            if os.path.isdir(persist_dir):
                shutil.rmtree(persist_dir)
        except Exception:
            # If deletion fails we log but do not raise
            pass

    def delete_all_sessions(self, user_id: str) -> None:
        """Clear all sessions and memories for a user."""
        # Delete each session directory (including sessions loaded from disk)
        session_ids = set(self._memories.get(user_id, {}).keys()) | set(
            self._sessions.get(user_id, {}).keys()
        )
        for session_id in list(session_ids):
            self.delete_session(user_id, session_id)
        # Clean up root user directory if empty
        import os
        import shutil
        user_dir = self._persist_root / user_id
        try:
            if os.path.isdir(user_dir) and not os.listdir(user_dir):
                shutil.rmtree(user_dir)
        except Exception:
            pass

    def persist_session(self, user_id: str, session_id: str) -> None:
        """Force a session metadata snapshot to disk."""
        self._persist_session(user_id, session_id)

    # ------------------------------------------------------------------
    # Persistence helpers

    def _session_directory(self, user_id: str, session_id: str) -> str:
        return str(self._persist_root / user_id / session_id)

    def _metadata_path(self, user_id: str, session_id: str) -> Path:
        return self._persist_root / user_id / session_id / self._metadata_filename

    def _persist_session(self, user_id: str, session_id: str) -> None:
        """Persist session metadata to disk for dashboard analytics."""
        session = self._sessions.get(user_id, {}).get(session_id)
        if session is None:
            return
        metadata_path = self._metadata_path(user_id, session_id)
        try:
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            payload = session.model_dump(mode="json")
            with metadata_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=True, indent=2)
        except Exception as exc:
            logger.warning(
                "Failed to persist session metadata for user=%s session=%s: %s",
                user_id,
                session_id,
                exc,
            )

    def _load_existing_sessions(self) -> None:
        """Load previously stored session metadata from disk."""
        root = self._persist_root
        if not root.exists() or not root.is_dir():
            return
        for user_dir in root.iterdir():
            if not user_dir.is_dir():
                continue
            user_id = user_dir.name
            for session_dir in user_dir.iterdir():
                if not session_dir.is_dir():
                    continue
                session_id = session_dir.name
                metadata_file = session_dir / self._metadata_filename
                session: Conversation | None = None
                loaded_from_metadata = False
                if metadata_file.is_file():
                    session = self._load_session_from_metadata(metadata_file)
                    loaded_from_metadata = session is not None
                if session is None:
                    session = self._load_session_from_vector_store(user_id, session_dir)
                if session is None:
                    continue
                self._upgrade_message_payloads(session)
                self._sessions.setdefault(user_id, {})[session.session_id] = session
                self._persist_session(user_id, session.session_id)

    def _load_session_from_metadata(self, metadata_file: Path) -> Conversation | None:
        try:
            with metadata_file.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            session = Conversation.model_validate(payload)
            self._upgrade_message_payloads(session)
            return session
        except Exception as exc:
            logger.warning(
                "Failed to load session metadata from %s: %s",
                metadata_file,
                exc,
            )
            return None

    def _load_session_from_vector_store(self, user_id: str, session_dir: Path) -> Conversation | None:
        """Reconstruct session metadata from Chroma storage when metadata is missing."""
        sqlite_path = session_dir / "chroma.sqlite3"
        if not sqlite_path.exists():
            return None
        try:
            connection = sqlite3.connect(sqlite_path)
        except Exception as exc:
            logger.warning(
                "Failed to open vector store at %s: %s",
                sqlite_path,
                exc,
            )
            return None

        try:
            cursor = connection.cursor()
            cursor.execute("SELECT id, created_at FROM embeddings ORDER BY created_at ASC")
            embedding_rows = cursor.fetchall()
            if not embedding_rows:
                return None
            cursor.execute(
                "SELECT id, string_value FROM embedding_metadata WHERE key = ?",
                ("chroma:document",),
            )
            doc_rows = cursor.fetchall()
            docs: dict[int, str] = {}
            for embed_id, doc_value in doc_rows:
                if doc_value:
                    docs[int(embed_id)] = str(doc_value)
            if not docs:
                return None

            messages: list[ChatMessage] = []
            session_created: datetime | None = None
            session_updated: datetime | None = None
            for embed_id, created_raw in embedding_rows:
                doc = docs.get(int(embed_id))
                if not doc:
                    continue
                created_at = self._parse_sqlite_timestamp(created_raw)
                if session_created is None or created_at < session_created:
                    session_created = created_at
                if session_updated is None or created_at > session_updated:
                    session_updated = created_at
                question, answer = self._split_document(doc)
                timestamp_str = created_at.isoformat()
                if question:
                    messages.append(
                        ChatMessage(
                            role=MessageRole.USER,
                            content=question,
                            content_type=MessageContentType.TEXT,
                            timestamp=timestamp_str,
                        )
                    )
                if answer:
                    messages.append(
                        ChatMessage(
                            role=MessageRole.ASSISTANT,
                            content=answer,
                            content_type=MessageContentType.TEXT,
                            timestamp=timestamp_str,
                        )
                    )

            if not messages:
                return None

            created_at = session_created or datetime.utcnow()
            updated_at = session_updated or created_at
            return Conversation(
                session_id=session_dir.name,
                user_id=user_id,
                title=None,
                created_at=created_at,
                updated_at=updated_at,
                message_count=len(messages),
                messages=messages,
            )
        except Exception as exc:
            logger.warning(
                "Failed to reconstruct session from %s: %s",
                sqlite_path,
                exc,
            )
            return None
        finally:
            try:
                connection.close()
            except Exception:
                pass

    @staticmethod
    def _parse_sqlite_timestamp(raw_value: object) -> datetime:
        if isinstance(raw_value, (int, float)):
            return datetime.fromtimestamp(raw_value)
        if isinstance(raw_value, bytes):
            raw_value = raw_value.decode("utf-8", errors="ignore")
        if isinstance(raw_value, str):
            cleaned = raw_value.replace("Z", "+00:00")
            try:
                return datetime.fromisoformat(cleaned)
            except ValueError:
                pass
        return datetime.utcnow()

    @staticmethod
    def _split_document(doc: str) -> tuple[str, str]:
        text = doc.strip()
        if text.lower().startswith("input:"):
            text = text[6:]
        parts = text.split("\noutput:", 1)
        question = parts[0].strip()
        answer = parts[1].strip() if len(parts) > 1 else ""
        return question, answer

    def _upgrade_message_payloads(self, session: Conversation) -> None:
        """Ensure stored structured payloads follow the new components schema."""
        for message in session.messages:
            if message.components is not None:
                continue
            # Legacy metadata may have been stored as a dict with data_type/content.
            legacy_payload = getattr(message, "structured_data", None)
            if isinstance(legacy_payload, dict):
                components = legacy_payload.get("components")
                if components is None:
                    components = [
                        {
                            "type": "custom",
                            "payload": {
                                "data": legacy_payload,
                                "content": (message.content or "").strip(),
                            },
                        }
                    ]
                message.components = components
