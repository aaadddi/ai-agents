import ulid # (Universally Unique Lexicographically Sortable Identifier)
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from redisvl.schema.schema import IndexSchema

class MemoryType(str, Enum):
    """
    Defines the type of long-term memory for categorization and retrieval.

    EPISODIC: Personal experiences and user-specific preferences
              (e.g., "User prefers Delta airlines", "User visited Paris last year")

    SEMANTIC: General domain knowledge and facts
              (e.g., "Singapore requires passport", "Tokyo has excellent public transit")

    The type of a long-term memory.

    EPISODIC: User specific experiences and preferences

    SEMANTIC: General knowledge on top of the user's preferences and LLM's
    training data.
    """

    EPISODIC = "episodic"
    SEMANTIC = "semantic"


class Memory(BaseModel):
    """Represents a single long-term memory."""

    content: str
    memory_type: MemoryType
    metadata: str

class Memories(BaseModel):
    """
    A list of memories extracted from a conversation by an LLM.

    NOTE: OpenAI's structured output requires us to wrap the list in an object.
    """

    memories: List[Memory]

class StoredMemory(Memory):
    """A stored long-term memory"""

    id: str  # The redis key
    memory_id: ulid.ULID = Field(default_factory=lambda: ulid.ULID())
    created_at: datetime = Field(default_factory=datetime.now)
    user_id: Optional[str] = None
    thread_id: Optional[str] = None
    memory_type: Optional[MemoryType] = None

memory_schema = IndexSchema.from_dict({
        "index": {
            "name": "agent_memories",  # Index name for identification
            "prefix": "memory",       # Redis key prefix (memory:1, memory:2, etc.)
            "key_separator": ":",
            "storage_type": "json",
        },
        "fields": [
            {"name": "content", "type": "text"},
            {"name": "memory_type", "type": "tag"},
            {"name": "metadata", "type": "text"},
            {"name": "created_at", "type": "text"},
            {"name": "user_id", "type": "tag"},
            {"name": "memory_id", "type": "tag"},
            {
                "name": "embedding",
                "type": "vector",
                "attrs": {
                    "algorithm": "flat",
                    "dims": 3072,  # googleAI embedding dimension
                    "distance_metric": "cosine",
                    "datatype": "float32",
                },
            },
        ],
    }
)