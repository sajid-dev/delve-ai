"""Reusable LangChain prompt pipelines for chat generation."""

from __future__ import annotations

import json
from typing import Any

from loguru import logger
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from ..prompts import (
    DEFAULT_SYSTEM_PROMPT,
    ROUTER_HUMAN_PROMPT,
    ROUTER_SYSTEM_PROMPT,
    SEQUENTIAL_EXECUTOR_HUMAN_PROMPT,
    SEQUENTIAL_EXECUTOR_SYSTEM_PROMPT,
    SEQUENTIAL_PLANNER_HUMAN_PROMPT,
    SEQUENTIAL_PLANNER_SYSTEM_PROMPT,
)


class ChatChainManager:
    """Encapsulates chat routing and sequential chain execution."""

    def __init__(self, system_prompt: str | None = None) -> None:
        self._system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

        self._prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "{system_prompt}"),
                ("human", "{user_prompt}"),
            ]
        )

        self._router_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    ROUTER_SYSTEM_PROMPT,
                ),
                (
                    "human",
                    ROUTER_HUMAN_PROMPT,
                ),
            ]
        )

        self._sequential_planner_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    SEQUENTIAL_PLANNER_SYSTEM_PROMPT,
                ),
                (
                    "human",
                    SEQUENTIAL_PLANNER_HUMAN_PROMPT,
                ),
            ]
        )

        self._sequential_executor_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    SEQUENTIAL_EXECUTOR_SYSTEM_PROMPT,
                ),
                (
                    "human",
                    SEQUENTIAL_EXECUTOR_HUMAN_PROMPT,
                ),
            ]
        )

    @property
    def system_prompt(self) -> str:
        """Return the base system prompt for all chains."""
        return self._system_prompt

    def build_system_message(self, history_snippets: str | None) -> str:
        """Construct the system prompt including contextual memory."""
        if history_snippets:
            return f"{self._system_prompt}\n\nContext:\n{history_snippets}"
        return f"{self._system_prompt}\n\nContext: <none>"

    def decide_generation_route(
        self,
        llm: ChatOpenAI,
        prompt: str,
        history_snippets: str | None,
    ) -> str:
        """Use the router prompt to choose between sequential and standard execution."""
        try:
            router_response = self._invoke_template(
                llm,
                self._router_template,
                {
                    "system_prompt": self._system_prompt,
                    "context": history_snippets or "<none>",
                    "question": prompt,
                },
            )
            logger.debug("Router raw response: {}", router_response)
        except Exception:
            logger.exception("Routing decision failed; defaulting to standard mode")
            return "standard"

        try:
            parsed = json.loads(router_response)
            route_value = str(parsed.get("route", "standard")).strip().lower()
        except Exception:
            logger.debug("Router response not valid JSON: {}", router_response)
            return "standard"

        if route_value not in {"standard", "sequential"}:
            logger.debug("Router returned unknown route {!r}; defaulting to standard", route_value)
            return "standard"
        logger.debug("Router resolved route={}", route_value)
        return route_value

    def generate_with_sequential_chain(
        self,
        llm: ChatOpenAI,
        prompt: str,
        history_snippets: str | None,
        tool_context: str | None = None,
    ) -> str:
        """Run the two-stage plan → execute flow for complex prompts."""
        context_text = history_snippets or "<none>"

        try:
            plan = self._invoke_template(
                llm,
                self._sequential_planner_template,
                {
                    "system_prompt": self._system_prompt,
                    "context": context_text,
                    "question": prompt,
                },
            )
            logger.debug(
                "Sequential planner output for prompt {!r}: {}",
                prompt[:80],
                plan,
            )
        except Exception:
            logger.exception("Sequential planner failed; falling back to standard execution")
            system_message = self.build_system_message(history_snippets)
            return self.invoke_standard(llm, system_message, prompt, tool_context)

        execution_response = self._invoke_template(
            llm,
            self._sequential_executor_template,
            {
                "system_prompt": self._system_prompt,
                "context": context_text,
                "plan": plan or "Plan could not be generated. Provide the best possible answer regardless.",
                "tool_context": tool_context or "<none>",
                "question": prompt,
            },
        )
        logger.debug(
            "Sequential executor produced response length={} for prompt snippet={}",
            len(execution_response),
            prompt[:80],
        )
        return execution_response

    def invoke_standard(
        self,
        llm: ChatOpenAI,
        system_message: str,
        prompt: str,
        tool_context: str | None = None,
    ) -> str:
        """Invoke the base chat chain with optional tool context."""
        human_content = prompt
        if tool_context:
            human_content = (
                f"{prompt}\n\n"
                f"Additional data gathered from MCP tools:\n{tool_context}"
            )
        chain = self._prompt_template | llm
        response = chain.invoke(
            {
                "system_prompt": system_message,
                "user_prompt": human_content,
            }
        )
        content = getattr(response, "content", str(response))
        return content.strip()

    def _invoke_template(
        self,
        llm: ChatOpenAI,
        template: ChatPromptTemplate,
        variables: dict[str, Any],
    ) -> str:
        """Execute a prompt template with the provided LLM and return string content."""
        chain = template | llm
        result = chain.invoke(variables)
        content = getattr(result, "content", str(result))
        return content.strip()
