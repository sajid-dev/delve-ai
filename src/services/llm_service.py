"""Service encapsulating interactions with the language model.

Uses LangChain's ChatOpenAI integration to communicate with the
OpenAI Chat API.  The service accepts a prompt and a memory
instance and returns the generated response.
"""

from __future__ import annotations

from loguru import logger
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI

from ..config.llm_config import LlmConfig, get_llm_config
from ..memory.chat_memory import ChatMemory


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
        # ``langchain_openai`` API reference【149861662473305†L170-L184】.
        llm_kwargs: dict[str, object] = {
            "api_key": self.llm_config.api_key,
            "model": self.llm_config.model,
            "temperature": self.llm_config.temperature,
        }
        if self.llm_config.base_url:
            llm_kwargs["base_url"] = self.llm_config.base_url
        # Pass optional tuning parameters only if they differ from defaults
        if self.llm_config.max_tokens:
            llm_kwargs["max_tokens"] = self.llm_config.max_tokens
        if self.llm_config.timeout:
            llm_kwargs["timeout"] = self.llm_config.timeout

        # Initialise the ChatOpenAI model with the assembled kwargs.  Unspecified
        # parameters will fall back to library defaults.
        self.llm = ChatOpenAI(**llm_kwargs)

    def generate(self, prompt: str, memory: ChatMemory) -> str:
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

        Returns
        -------
        str
            The assistant's reply.

        Raises
        ------
        ChatError
            If an unexpected error occurs during generation.
        """
        logger.debug("Generating response for prompt: %r", prompt)
        try:
            # Build a conversation chain combining the LLM and existing memory
            chain = ConversationChain(
                llm=self.llm,
                memory=memory.memory,
                verbose=False,
            )
            response = chain.predict(input=prompt)
            return response.strip()
        except Exception as exc:
            logger.exception("LLM generation failed")
            from ..utils.error_handler import ChatError

            raise ChatError("LLM generation failed") from exc