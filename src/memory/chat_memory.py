"""Chat memory implementation backed by ChromaDB and LangChain.

This class wraps a persistent vector store and provides an interface
to save question/answer pairs as context for future interactions.  The
memory is persisted on disk so that context survives application
restarts.
"""

from __future__ import annotations

from loguru import logger
from langchain.memory import VectorStoreRetrieverMemory
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

from ..config.llm_config import LlmConfig


class ChatMemory:
    """Persisted memory for storing and retrieving conversation context."""

    def __init__(self, llm_config: LlmConfig) -> None:
        """Initialise embeddings and the persistent vector store.

        The memory relies on embeddings to convert text into vector
        representations.  Here we instantiate :class:`OpenAIEmbeddings` using
        the unified LLM configuration.  The ``api_key`` and optional
        ``base_url`` are passed through so that both OpenAI and custom
        ChatGPT‑compatible endpoints are supported【149861662473305†L170-L184】.  Any
        exceptions during initialisation (for example, missing API keys or
        filesystem errors) are caught and re‑raised as :class:`ChatError` to
        provide a consistent error surface.
        """
        try:
            # Initialise the embedding model using the unified API key and base URL.
            embed_kwargs: dict[str, object] = {
                "api_key": llm_config.api_key,
            }
            if llm_config.base_url:
                embed_kwargs["base_url"] = llm_config.base_url
            embeddings = OpenAIEmbeddings(**embed_kwargs)

            # Create a persistent Chroma vector store using the embeddings.
            self.vectorstore = Chroma(
                persist_directory="chroma_db",
                embedding_function=embeddings,
            )
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
            # Use LangChain's VectorStoreRetrieverMemory to manage conversation history
            self.memory = VectorStoreRetrieverMemory(memory_key="history", retriever=retriever)
        except Exception as exc:
            logger.exception("Failed to initialise chat memory")
            from ..utils.error_handler import ChatError

            raise ChatError("Failed to initialise chat memory") from exc

    def save_interaction(self, question: str, answer: str) -> None:
        """Persist a question and answer to the memory store.

        Any exceptions raised by the vector store are caught and converted
        into a ChatError.  This prevents lower-level errors from leaking
        directly to the API layer.
        """
        logger.debug("Persisting Q/A pair to vector store")
        try:
            self.memory.save_context({"input": question}, {"output": answer})
        except Exception as exc:
            logger.exception("Error saving context to vector store")
            from ..utils.error_handler import ChatError

            raise ChatError("Failed to persist interaction") from exc