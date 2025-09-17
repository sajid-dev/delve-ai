"""Enumerations used across models."""

from enum import Enum


class MessageRole(str, Enum):
    """Role of a message within a conversation."""

    USER = "user"
    AI = "ai"