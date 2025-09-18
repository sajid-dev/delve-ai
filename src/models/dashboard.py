"""Pydantic models for admin dashboard analytics."""

from __future__ import annotations

from datetime import datetime
from typing import List

from pydantic import BaseModel


class ConversationStats(BaseModel):
    """Analytics for a single conversation."""

    conversation_id: str
    title: str | None
    message_count: int
    created_at: datetime
    updated_at: datetime
    tokens_used: int


class UserConversationStats(BaseModel):
    """Aggregated analytics for a user's conversations."""

    user_id: str
    conversation_count: int
    total_tokens: int
    conversations: List[ConversationStats]


class DashboardData(BaseModel):
    """Top-level container for admin dashboard analytics."""

    total_users: int
    total_conversations: int
    total_tokens: int
    users: List[UserConversationStats]
