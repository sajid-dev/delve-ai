"""Service encapsulating interactions with the language model.

Uses LangChain's ChatOpenAI integration to communicate with the
OpenAI Chat API.  The service accepts a prompt and a memory
instance and returns the generated response.
"""

from __future__ import annotations

import math

from loguru import logger
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from ..chains import ChatChainManager
from ..config.llm_config import LlmConfig, get_llm_config
from ..context.mcp_context import MCPContextCollector
from ..memory.chat_memory import ChatMemory
from ..models.chat_message import ChatMessage
from ..models.enums import MessageRole


class LLMService:
    """Service for generating responses from the language model.

    This service wraps LangChain's :class:`~langchain_openai.ChatOpenAI` class and
    constructs it based solely on a unified :class:`LlmConfig` instance.  All
    provider‑specific details (such as API keys, base URL and model name) are
    defined in a single configuration.  There is no longer any distinction
    between OpenAI and other providers – if a ``base_url`` is provided the
    client will communicate with the supplied endpoint, otherwise it will use
    the default OpenAI endpoint.  Additional parameters such as ``temperature``,
    ``max_tokens`` and ``timeout`` are forwarded verbatim to the underlying
    ChatOpenAI constructor.
    """

    def __init__(self, llm_config: LlmConfig | None = None) -> None:
        """Initialise the LLM service with the provided configuration.

        Parameters
        ----------
        llm_config: LlmConfig, optional
            A configuration instance specifying API credentials, base URL,
            model name and tuning parameters.  If omitted, the configuration
            will be loaded from environment variables via :func:`get_llm_config`.
        """
        # Load LLM configuration if none supplied
        self.llm_config = llm_config or get_llm_config()

        # Build keyword arguments for ChatOpenAI using the unified config.  At a
        # minimum the API key, model name and temperature are provided.  If a
        # base URL, max_tokens or timeout are specified they are also passed
        # through.  These keys correspond to aliases documented in the
        # ``langchain_openai`` API reference.
        self._llm_kwargs: dict[str, object] = {
            "api_key": self.llm_config.api_key,
            "model": self.llm_config.model,
            "temperature": self.llm_config.temperature,
        }
        if self.llm_config.base_url:
            self._llm_kwargs["base_url"] = self.llm_config.base_url
        # Pass optional tuning parameters only if they differ from defaults
        if self.llm_config.max_tokens:
            self._llm_kwargs["max_tokens"] = self.llm_config.max_tokens
        if self.llm_config.timeout:
            self._llm_kwargs["timeout"] = self.llm_config.timeout

        # Optional keyword arguments forwarded to the model constructor (e.g. user-level tags).
        self._model_kwargs: dict[str, object] = {}

        init_kwargs: dict[str, object] = dict(self._llm_kwargs)
        if self._model_kwargs:
            init_kwargs["model_kwargs"] = dict(self._model_kwargs)

        # Initialise the ChatOpenAI model with the assembled kwargs.  Unspecified
        # parameters will fall back to library defaults.
        self.llm = ChatOpenAI(**init_kwargs)

        # Chain manager encapsulates routing, sequential planning, and prompt execution.
        self.chain_manager = ChatChainManager()

        self._mcp_collector: MCPContextCollector | None = None
        if self.llm_config.mcp_enabled:
            self._mcp_collector = MCPContextCollector(self.llm_config.mcp)

    def generate(
        self,
        prompt: str,
        memory: ChatMemory,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> str:
        """Generate a response using the provided prompt and memory.

        This method wraps the LangChain call in a try/except block so that
        any exceptions are converted into a ChatError.  This ensures
        upstream callers can handle failures uniformly.

        Parameters
        ----------
        prompt: str
            The user's input message.
        memory: ChatMemory
            A chat memory instance providing conversation history.
        user_id: str, optional
            Identifier of the end user for the request.  When provided the
            identifier is forwarded to the underlying LLM so usage can be
            tracked per user.
        session_id: str, optional
            Identifier for the chat session (conversation). Used for logging and
            downstream tooling.

        Returns
        -------
        str
            The assistant's reply.

        Raises
        ------
        ChatError
            If an unexpected error occurs during generation.
        """
        logger.debug(
            "Generating response for prompt: {!r} user={} session={}",
            prompt,
            user_id,
            session_id,
        )
        try:
            llm = self._resolve_llm(user_id)
            history_snippets = memory.get_relevant_history(prompt)
            tool_context = self._collect_tool_context(prompt, session_id)
            return self.chain_manager.summarize(
                llm=llm,
                prompt=prompt,
                history_snippets=history_snippets,
                tool_context=tool_context,
            )
        except Exception as exc:
            logger.exception("LLM generation failed")
            from ..utils.error_handler import ChatError

            raise ChatError("LLM generation failed") from exc

    def _resolve_llm(self, user_id: str | None) -> ChatOpenAI:
        """Return an LLM instance optionally tagged with the user identifier."""

        if not user_id:
            return self.llm

        model_kwargs = {**self._model_kwargs, "user": user_id}
        return ChatOpenAI(**self._llm_kwargs, model_kwargs=model_kwargs)

    def _collect_tool_context(
        self, prompt: str, session_id: str | None
    ) -> str | None:
        """Gather MCP context when the collector is configured."""

        if self._mcp_collector is None or not self.llm_config.mcp_enabled:
            return None

        try:
            context = self._mcp_collector.collect_context(
                prompt,
                session_id=session_id,
            )
        except Exception:
            logger.exception("Failed to collect MCP tool context")
            return None

        if context:
            logger.debug(
                "Collected MCP tool context for session={} ({} characters)",
                session_id,
                len(context),
            )
        else:
            logger.debug("No MCP context gathered for session={}", session_id)
        return context

    # ------------------------------------------------------------------
    # Token accounting helpers

    def count_tokens(self, messages: list[ChatMessage]) -> int:
        """Return token usage for a sequence of chat messages.

        Falls back to a simple character-based heuristic if the underlying
        model does not expose token counting utilities.
        """
        if not messages:
            return 0

        if hasattr(self.llm, "get_num_tokens_from_messages"):
            lc_messages = []
            for message in messages:
                role = message.role
                content = message.content
                if role == MessageRole.USER:
                    lc_messages.append(HumanMessage(content=content))
                elif role == MessageRole.ASSISTANT:
                    lc_messages.append(AIMessage(content=content))
                else:
                    lc_messages.append(SystemMessage(content=content))
            try:
                return int(self.llm.get_num_tokens_from_messages(lc_messages))
            except Exception:
                # Fall through to heuristic if token counting fails
                pass

        # Simple heuristic: average of 4 characters per token with minimum of 1
        return sum(max(1, math.ceil(len(message.content) / 4)) for message in messages)
