"""Chat memory implementation backed by ChromaDB and LangChain.

This class wraps a persistent vector store and provides an interface
to save question/answer pairs as context for future interactions.  The
memory is persisted on disk so that context survives application
restarts.
"""

from __future__ import annotations

from loguru import logger
from langchain.memory import VectorStoreRetrieverMemory
# Import the Chroma vector store from the dedicated integration package.
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from ..config.llm_config import LlmConfig


class ChatMemory:
    """Persisted memory for storing and retrieving conversation context.

    A ChatMemory instance encapsulates a Chroma vector store and a LangChain
    ``VectorStoreRetrieverMemory``.  It can be scoped to a particular user
    and conversation by specifying a ``persist_directory``.  This allows
    each conversation to maintain its own independent context stored on
    disk.  If no directory is provided, the default ``chroma_db`` root will
    be used.  Any exceptions during initialisation (for example, missing
    API keys or filesystem errors) are caught and re‑raised as
    :class:`ChatError` to provide a consistent error surface.
    """

    def __init__(self, llm_config: LlmConfig, persist_directory: str | None = None) -> None:
        # Determine the directory where vectors will be persisted.  Use a
        # conversation‑specific path if supplied, otherwise fall back to
        # the global ``chroma_db`` folder.
        directory = persist_directory or "chroma_db"
        try:
            # Initialise the embedding model using the unified API key and base URL.
            embed_kwargs: dict[str, object] = {
                "api_key": llm_config.api_key,
            }
            if llm_config.base_url:
                embed_kwargs["base_url"] = llm_config.base_url
            embeddings = OpenAIEmbeddings(**embed_kwargs)

            # Create a persistent Chroma vector store using the embeddings and per‑conversation directory.
            self.vectorstore = Chroma(
                persist_directory=directory,
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