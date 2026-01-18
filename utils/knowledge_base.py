"""Knowledge Base Manager for ResearchCrew

Provides long-term storage for research findings using ChromaDB as the vector database.
Enables agents to recall and reference past research across sessions.
"""

import hashlib
import logging
import os
from datetime import datetime
from typing import Any, Optional

import chromadb
from chromadb.config import Settings
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class KnowledgeEntry(BaseModel):
    """A single knowledge entry stored in the database."""

    id: str = Field(description="Unique identifier for this entry")
    content: str = Field(description="The main content/finding")
    source_url: Optional[str] = Field(default=None, description="Source URL if available")
    source_title: Optional[str] = Field(default=None, description="Source title")
    topic: Optional[str] = Field(default=None, description="Topic or category")
    session_id: Optional[str] = Field(default=None, description="Session that created this")
    confidence: Optional[str] = Field(default=None, description="Confidence level")
    created_at: str = Field(description="ISO timestamp when created")


class SearchResult(BaseModel):
    """A search result from the knowledge base."""

    entry: KnowledgeEntry = Field(description="The matched entry")
    score: float = Field(description="Similarity score (0-1, higher is more similar)")
    distance: float = Field(description="Distance metric from query")


class KnowledgeBaseManager:
    """Manages long-term knowledge storage using ChromaDB.

    This provides:
    - Vector similarity search for finding relevant past research
    - Metadata filtering (by topic, date, source)
    - Deduplication of similar content
    - Persistent storage (configurable path)

    For production, this can be replaced with Vertex AI Vector Search
    or Pinecone by implementing the same interface.
    """

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: str = "research_knowledge",
    ):
        """Initialize the knowledge base.

        Args:
            persist_directory: Path to store the database. If None, uses in-memory storage.
            collection_name: Name of the ChromaDB collection.
        """
        self.collection_name = collection_name

        # Configure ChromaDB settings
        if persist_directory:
            self.persist_directory = persist_directory
            os.makedirs(persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False),
            )
            logger.info(f"KnowledgeBase initialized with persistent storage: {persist_directory}")
        else:
            self.persist_directory = None
            self.client = chromadb.Client(
                settings=Settings(anonymized_telemetry=False),
            )
            logger.info("KnowledgeBase initialized with in-memory storage")

        # Get or create the collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "ResearchCrew knowledge base for research findings"},
        )

        logger.info(f"Collection '{collection_name}' ready with {self.collection.count()} entries")

    def _generate_id(self, content: str, source_url: Optional[str] = None) -> str:
        """Generate a unique ID for content based on hash.

        Args:
            content: The content to hash
            source_url: Optional source URL to include in hash

        Returns:
            A unique hash-based ID
        """
        hash_input = content
        if source_url:
            hash_input += source_url
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    def _check_duplicate(self, content: str, threshold: float = 0.95) -> Optional[str]:
        """Check if similar content already exists.

        Args:
            content: Content to check
            threshold: Similarity threshold (0-1) above which content is duplicate

        Returns:
            ID of duplicate entry if found, None otherwise
        """
        if self.collection.count() == 0:
            return None

        # Search for similar content
        results = self.collection.query(
            query_texts=[content],
            n_results=1,
            include=["distances"],
        )

        if results["ids"] and results["ids"][0]:
            # ChromaDB returns L2 distance by default
            # Convert to similarity score (lower distance = higher similarity)
            distance = results["distances"][0][0]
            # Approximate similarity from L2 distance
            # This is a rough heuristic - exact conversion depends on embedding space
            similarity = 1 / (1 + distance)

            if similarity >= threshold:
                return results["ids"][0][0]

        return None

    def add_entry(
        self,
        content: str,
        source_url: Optional[str] = None,
        source_title: Optional[str] = None,
        topic: Optional[str] = None,
        session_id: Optional[str] = None,
        confidence: Optional[str] = None,
        skip_duplicates: bool = True,
    ) -> Optional[KnowledgeEntry]:
        """Add a new entry to the knowledge base.

        Args:
            content: The main content/finding to store
            source_url: URL of the source
            source_title: Title of the source
            topic: Topic or category
            session_id: Session that created this entry
            confidence: Confidence level (high/medium/low)
            skip_duplicates: If True, skip adding if similar content exists

        Returns:
            The created entry, or None if skipped as duplicate
        """
        # Check for duplicates
        if skip_duplicates:
            duplicate_id = self._check_duplicate(content)
            if duplicate_id:
                logger.debug(f"Skipping duplicate content, existing ID: {duplicate_id}")
                return None

        # Generate ID
        entry_id = self._generate_id(content, source_url)
        created_at = datetime.now().isoformat()

        # Prepare metadata
        metadata = {
            "created_at": created_at,
        }
        if source_url:
            metadata["source_url"] = source_url
        if source_title:
            metadata["source_title"] = source_title
        if topic:
            metadata["topic"] = topic
        if session_id:
            metadata["session_id"] = session_id
        if confidence:
            metadata["confidence"] = confidence

        # Add to collection
        self.collection.add(
            ids=[entry_id],
            documents=[content],
            metadatas=[metadata],
        )

        entry = KnowledgeEntry(
            id=entry_id,
            content=content,
            source_url=source_url,
            source_title=source_title,
            topic=topic,
            session_id=session_id,
            confidence=confidence,
            created_at=created_at,
        )

        logger.info(f"Added knowledge entry: {entry_id}")
        return entry

    def search(
        self,
        query: str,
        n_results: int = 5,
        topic_filter: Optional[str] = None,
        min_confidence: Optional[str] = None,
    ) -> list[SearchResult]:
        """Search the knowledge base for relevant entries.

        Args:
            query: Search query text
            n_results: Maximum number of results to return
            topic_filter: Filter by topic (exact match)
            min_confidence: Filter by minimum confidence level

        Returns:
            List of search results sorted by relevance
        """
        if self.collection.count() == 0:
            return []

        # Build where filter
        where_filter = None
        if topic_filter:
            where_filter = {"topic": topic_filter}

        # Query the collection
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        search_results = []

        if results["ids"] and results["ids"][0]:
            for i, entry_id in enumerate(results["ids"][0]):
                content = results["documents"][0][i]
                metadata = results["metadatas"][0][i]
                distance = results["distances"][0][i]

                # Calculate similarity score from distance
                score = 1 / (1 + distance)

                # Apply confidence filter if specified
                if min_confidence:
                    confidence_order = {"high": 3, "medium": 2, "low": 1}
                    entry_conf = metadata.get("confidence", "low")
                    if confidence_order.get(entry_conf, 0) < confidence_order.get(
                        min_confidence, 0
                    ):
                        continue

                entry = KnowledgeEntry(
                    id=entry_id,
                    content=content,
                    source_url=metadata.get("source_url"),
                    source_title=metadata.get("source_title"),
                    topic=metadata.get("topic"),
                    session_id=metadata.get("session_id"),
                    confidence=metadata.get("confidence"),
                    created_at=metadata.get("created_at", ""),
                )

                search_results.append(
                    SearchResult(
                        entry=entry,
                        score=score,
                        distance=distance,
                    )
                )

        logger.debug(f"Search for '{query}' returned {len(search_results)} results")
        return search_results

    def get_entry(self, entry_id: str) -> Optional[KnowledgeEntry]:
        """Get a specific entry by ID.

        Args:
            entry_id: The entry ID

        Returns:
            The entry if found, None otherwise
        """
        results = self.collection.get(
            ids=[entry_id],
            include=["documents", "metadatas"],
        )

        if results["ids"]:
            content = results["documents"][0]
            metadata = results["metadatas"][0]

            return KnowledgeEntry(
                id=entry_id,
                content=content,
                source_url=metadata.get("source_url"),
                source_title=metadata.get("source_title"),
                topic=metadata.get("topic"),
                session_id=metadata.get("session_id"),
                confidence=metadata.get("confidence"),
                created_at=metadata.get("created_at", ""),
            )

        return None

    def delete_entry(self, entry_id: str) -> bool:
        """Delete an entry by ID.

        Args:
            entry_id: The entry ID to delete

        Returns:
            True if deleted, False if not found
        """
        try:
            self.collection.delete(ids=[entry_id])
            logger.info(f"Deleted knowledge entry: {entry_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to delete entry {entry_id}: {e}")
            return False

    def list_topics(self) -> list[str]:
        """List all unique topics in the knowledge base.

        Returns:
            List of topic names
        """
        # Get all entries with metadata
        results = self.collection.get(include=["metadatas"])

        topics = set()
        for metadata in results["metadatas"]:
            if metadata.get("topic"):
                topics.add(metadata["topic"])

        return sorted(list(topics))

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the knowledge base.

        Returns:
            Dictionary of statistics
        """
        total_entries = self.collection.count()

        # Get topic distribution
        results = self.collection.get(include=["metadatas"])
        topic_counts: dict[str, int] = {}
        confidence_counts: dict[str, int] = {}

        for metadata in results["metadatas"]:
            topic = metadata.get("topic", "uncategorized")
            topic_counts[topic] = topic_counts.get(topic, 0) + 1

            confidence = metadata.get("confidence", "unknown")
            confidence_counts[confidence] = confidence_counts.get(confidence, 0) + 1

        return {
            "total_entries": total_entries,
            "topics": topic_counts,
            "confidence_distribution": confidence_counts,
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory,
        }

    def clear(self) -> int:
        """Clear all entries from the knowledge base.

        Returns:
            Number of entries deleted
        """
        count = self.collection.count()
        if count > 0:
            # Get all IDs and delete
            all_ids = self.collection.get()["ids"]
            self.collection.delete(ids=all_ids)
            logger.info(f"Cleared {count} entries from knowledge base")
        return count


# Global instance management
_knowledge_base: Optional[KnowledgeBaseManager] = None


def get_knowledge_base(
    persist_directory: Optional[str] = None,
) -> KnowledgeBaseManager:
    """Get the global knowledge base instance.

    Args:
        persist_directory: Path for persistent storage (only used on first call)

    Returns:
        The singleton KnowledgeBaseManager instance
    """
    global _knowledge_base
    if _knowledge_base is None:
        # Default persist directory from environment or use local path
        if persist_directory is None:
            persist_directory = os.getenv(
                "RESEARCHCREW_KNOWLEDGE_PATH",
                ".knowledge_base",
            )
        _knowledge_base = KnowledgeBaseManager(persist_directory=persist_directory)
    return _knowledge_base


def reset_knowledge_base() -> None:
    """Reset the knowledge base instance (useful for testing)."""
    global _knowledge_base
    _knowledge_base = None
