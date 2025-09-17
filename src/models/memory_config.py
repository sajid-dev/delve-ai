"""Configuration options for memory retrieval."""

from pydantic import BaseModel


class MemoryConfig(BaseModel):
    """Configuration controlling how the memory is queried."""

    k: int = 5