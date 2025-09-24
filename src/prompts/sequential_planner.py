"""Prompts for generating high-level execution plans."""

SEQUENTIAL_PLANNER_SYSTEM_PROMPT = (
    "You create concise plans before responding as the assistant. "
    "Return a numbered plan (3 bullets max) focusing on how to fulfil the request."
)

SEQUENTIAL_PLANNER_HUMAN_PROMPT = (
    "Base instructions:\n{system_prompt}\n\n"
    "Conversation context:\n{context}\n\n"
    "User request:\n{question}\n\n"
    "Produce the plan only."
)