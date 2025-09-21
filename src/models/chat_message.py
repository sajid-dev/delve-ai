"""Models representing chat messages and related structures."""

from typing import Any

from pydantic import BaseModel, model_validator

from .enums import MessageContentType, MessageRole


class ChatMessage(BaseModel):
    """Represents a single message in a conversation.

    In addition to the role and content, each message is timestamped.
    The ``timestamp`` field records when the message was created in ISO
    8601 format (UTC).  This allows for ordering and auditing of
    conversations over time.  Clients are not required to supply a
    timestamp; if omitted, one will be generated automatically by
    services when the message is persisted.  Structured payloads (such as
    parsed tables or charts) can optionally be attached via
    ``components`` so that frontends can render richer widgets
    without re-parsing the raw text.
    """

    role: MessageRole
    content: str
    content_type: MessageContentType = MessageContentType.TEXT
    timestamp: str | None = None
    components: list[dict[str, Any]] | None = None

    @model_validator(mode="before")
    @classmethod
    def _migrate_structured_data(cls, data: Any) -> Any:
        """Backwards compatibility for legacy ``structured_data`` payloads."""
        if not isinstance(data, dict):
            return data

        if "components" in data:
            return data

        structured = data.pop("structured_data", None)
        if structured is None:
            return data

        if isinstance(structured, dict) and "components" in structured:
            data["components"] = structured.get("components")
        else:
            data["components"] = [
                {
                    "type": "custom",
                    "payload": {"data": structured},
                }
            ]
        return data
