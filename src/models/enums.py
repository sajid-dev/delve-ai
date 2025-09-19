"""Enumerations used across models."""

from enum import Enum


class MessageRole(str, Enum):
    """Enum for message roles in a conversation.

    The role field distinguishes between the sender of each message in the
    conversation.  ``USER`` denotes a human message, ``ASSISTANT`` denotes
    a reply from the AI model, and ``SYSTEM`` can be used for
    informational or system‑level messages.
    """

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class MessageContentType(str, Enum):
    """Enum describing the content type of a chat message for frontend rendering."""

    TEXT = "text"
    MARKDOWN = "markdown"
    CODE = "code"
    TABLE = "table"
    IMAGE = "image"
    JSON = "json"
    HTML = "html"
    CHART = "chart"
