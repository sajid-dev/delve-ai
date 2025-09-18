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