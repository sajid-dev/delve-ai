"""Reusable LangChain prompt pipelines for chat generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from ..prompts import DEFAULT_SYSTEM_PROMPT


@dataclass(slots=True)
class PromptVariables:
    """Container for variables required by the summary prompt."""

    system_prompt: str
    user_prompt: str


class ChatChainManager:
    """Generate responses by summarising provided context without speculation."""

    def __init__(self, system_prompt: str | None = None) -> None:
        self._system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self._prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "{system_prompt}"),
                ("human", "{user_prompt}"),
            ]
        )

    @property
    def system_prompt(self) -> str:
        """Return the base system prompt for the chain."""

        return self._system_prompt

    def summarize(
        self,
        llm: ChatOpenAI,
        prompt: str,
        history_snippets: str | None,
        tool_context: str | None,
    ) -> str:
        """Return a summary that relies solely on supplied context and tool data."""

        system_message = self._build_system_message(history_snippets, tool_context)
        variables = PromptVariables(
            system_prompt=system_message,
            user_prompt=prompt,
        )
        response = (self._prompt_template | llm).invoke(variables.__dict__)
        content = getattr(response, "content", str(response))
        return content.strip()

    def _build_system_message(
        self, history_snippets: str | None, tool_context: str | None
    ) -> str:
        """Compose the system message with optional history and MCP context."""

        parts = [self._system_prompt]
        if history_snippets:
            parts.append("\nConversation context:\n" + history_snippets)
        else:
            parts.append("\nConversation context: <none>")

        if tool_context:
            parts.append("\nVerified MCP data:\n" + tool_context)
        else:
            parts.append("\nVerified MCP data: <none>")
        return "".join(parts)
