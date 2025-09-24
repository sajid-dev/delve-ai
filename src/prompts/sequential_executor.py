"""Prompts used when executing a generated plan."""

SEQUENTIAL_EXECUTOR_SYSTEM_PROMPT = (
    "You are the assistant following a provided plan. Use the plan, context, "
    "and any tool data to craft a thorough yet concise reply."
)

SEQUENTIAL_EXECUTOR_HUMAN_PROMPT = (
    "Base instructions:\n{system_prompt}\n\n"
    "Conversation context:\n{context}\n\n"
    "Plan to follow:\n{plan}\n\n"
    "Additional tool context (can be <none>):\n{tool_context}\n\n"
    "User request:\n{question}"
)