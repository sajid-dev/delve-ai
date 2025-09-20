"""Pydantic models for admin dashboard analytics."""

from __future__ import annotations

from datetime import datetime
from typing import List

from pydantic import BaseModel


class ConversationStats(BaseModel):
    """Analytics for a single session."""

    session_id: str
    title: str | None
    message_count: int
    created_at: datetime
    updated_at: datetime
    tokens_used: int
    latest_answer: dict[str, object] | None = None


class UserConversationStats(BaseModel):
    """Aggregated analytics for a user's sessions."""

    user_id: str
    session_count: int
    total_tokens: int
    last_active: datetime | None
    is_active: bool
    sessions: List[ConversationStats]


class DashboardData(BaseModel):
    """Top-level container for admin dashboard analytics."""

    total_users: int
    active_users: int
    total_sessions: int
    total_tokens: int
    users: List[UserConversationStats]
