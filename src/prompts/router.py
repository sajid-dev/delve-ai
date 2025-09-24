"""Prompts used to route user requests between execution paths."""

ROUTER_SYSTEM_PROMPT = (
    "You analyse user requests and choose the best response path. "
    "Reply with JSON of the form {\"route\": \"sequential\"} or {\"route\": \"standard\"}. "
    "Pick 'sequential' for multi-step instructions, planning, research, or when the user asks for detailed breakdowns. "
    "Use 'standard' for simple, short, or conversational replies."
)

ROUTER_HUMAN_PROMPT = (
    "Base instructions:\n{system_prompt}\n\n"
    "Conversation context:\n{context}\n\n"
    "User request:\n{question}"
)