import logging
from utils import get_redis_client, get_vertex_embed
from redisvl.index import SearchIndex
from redisvl.query import VectorRangeQuery
from redisvl.query.filter import Tag
from typing import Optional, List, Union
from memory_data_models import MemoryType, memory_schema, StoredMemory
import ulid
from datetime import datetime
from utils import SYSTEM_USER_ID
logger = logging.getLogger(__name__)

class MemoryUtils:
    def __init__(self) -> None:
        self.redis_client = get_redis_client()
        self.long_term_memory_index = self.create_long_term_memory_index(self.redis_client, memory_schema)
        self.vertex_embed = get_vertex_embed()


    def create_long_term_memory_index(self,redis_client, memory_schema, validate_on_load=True):
        try:
            long_term_memory_index = SearchIndex(
                schema=memory_schema,
                redis_client=redis_client,
                validate_on_load=validate_on_load
            )
            long_term_memory_index.create(overwrite=True)
            print("Long-term memory index ready")
            return long_term_memory_index
        except Exception as e:
            print(f"Error creating index: {e}")


    def similar_memory_exists(
        self,
        content: str,
        memory_type: MemoryType,
        user_id: str = SYSTEM_USER_ID,
        thread_id: Optional[str] = None,
        distance_threshold: float = 0.1,
    ) -> bool:
        """Check if a similar long-term memory already exists in Redis."""

        vertex_embed = get_vertex_embed()
        content_embedding = vertex_embed.embed(content)

        filters = (Tag("user_id") == user_id) & (Tag("memory_type") == memory_type)

        if thread_id:
            filters = filters & (Tag("thread_id") == thread_id)

        # Search for similar memories
        vector_query = VectorRangeQuery(
            vector=content_embedding,
            num_results=1,
            vector_field_name="embedding",
            filter_expression=filters,
            distance_threshold=distance_threshold,
            return_fields=["id"],
        )
        results = self.long_term_memory_index.query(vector_query)
        logger.debug(f"Similar memory search results: {results}")

        if results:
            logger.debug(
                f"{len(results)} similar {'memory' if results.count == 1 else 'memories'} found. First: "
                f"{results[0]['id']}. Skipping storage."
            )
            return True

        return False
    
    def store_memory(
        self,
        content: str,
        memory_type: MemoryType,
        user_id: str = SYSTEM_USER_ID,
        thread_id: Optional[str] = None,
        metadata: Optional[str] = None,
        ):
        """Store a long-term memory in Redis with deduplication.

            This function:
            1. Checks for similar existing memories to avoid duplicates
            2. Generates vector embeddings for semantic search
            3. Stores the memory with metadata for retrieval
            """
        if metadata is None:
            metadata = "{}"

        logger.info(f"Preparing to store memory: {content}")

        if self.similar_memory_exists(content, memory_type, user_id, thread_id):
            logger.info("Similar memory found, skipping storage")
            return

        embedding = self.vertex_embed.embed(content)

        memory_data = {
            "user_id": user_id or SYSTEM_USER_ID,
            "content": content,
            "memory_type": memory_type.value,
            "metadata": metadata,
            "created_at": datetime.now().isoformat(),
            "embedding": embedding,
            "memory_id": str(ulid.ULID()),
            "thread_id": thread_id,
        }

        try:
            self.long_term_memory_index.load([memory_data])
        except Exception as e:
            logger.error(f"Error storing memory: {e}")
            return

        logger.info(f"Stored {memory_type} memory: {content}")

    def retrieve_memories(
        self,
        query: str,
        memory_type: Union[Optional[MemoryType], List[MemoryType]] = None,
        user_id: str = SYSTEM_USER_ID,
        thread_id: Optional[str] = None,
        distance_threshold: float = 0.1,
        limit: int = 5,
    ) -> List[StoredMemory]:
        """Retrieve relevant memories from Redis using vector similarity search.

        """
        # Create vector query using query embedding
        logger.debug(f"Retrieving memories for query: {query}")
        vector_query = VectorRangeQuery(
            vector=self.vertex_embed.embed(query),
            return_fields=[
                "content",
                "memory_type", 
                "metadata",
                "created_at",
                "memory_id",
                "thread_id",
                "user_id",
            ],
            num_results=limit,
            vector_field_name="embedding",
            dialect=2,
            distance_threshold=distance_threshold,
        )

        # Build filter conditions
        base_filters = [f"@user_id:{{{user_id or SYSTEM_USER_ID}}}"]

        if memory_type:
            if isinstance(memory_type, list):
                base_filters.append(f"@memory_type:{{{'|'.join(memory_type)}}}")
            else:
                base_filters.append(f"@memory_type:{{{memory_type.value}}}")

        if thread_id:
            base_filters.append(f"@thread_id:{{{thread_id}}}")

        vector_query.set_filter(" ".join(base_filters))

        # Execute vector similarity search
        results = self.long_term_memory_index.query(vector_query)

        # Parse results into StoredMemory objects
        memories = []
        for doc in results:
            try:
                memory = StoredMemory(
                    id=doc["id"],
                    memory_id=doc["memory_id"],
                    user_id=doc["user_id"],
                    thread_id=doc.get("thread_id", None),
                    memory_type=MemoryType(doc["memory_type"]),
                    content=doc["content"],
                    created_at=doc["created_at"],
                    metadata=doc["metadata"],
                )
                memories.append(memory)
            except Exception as e:
                logger.error(f"Error parsing memory: {e}")
                continue
        return memories