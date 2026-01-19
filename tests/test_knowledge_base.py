"""Tests for Knowledge Base

Tests the long-term knowledge storage functionality using ChromaDB.
"""

import os
import shutil
import tempfile

from utils.knowledge_base import (
    KnowledgeBaseManager,
    KnowledgeEntry,
    SearchResult,
    get_knowledge_base,
    reset_knowledge_base,
)


class TestKnowledgeBaseManager:
    """Tests for the KnowledgeBaseManager class."""

    def setup_method(self, method):
        """Create a fresh in-memory knowledge base for each test."""
        reset_knowledge_base()
        # Use unique collection name per test to ensure isolation
        collection_name = f"test_{method.__name__}"
        self.kb = KnowledgeBaseManager(persist_directory=None, collection_name=collection_name)

    def test_add_entry(self):
        """Test adding a new entry."""
        entry = self.kb.add_entry(
            content="LangGraph is a library for building stateful multi-actor applications.",
            source_url="https://langchain.com/langgraph",
            source_title="LangGraph Documentation",
            topic="AI frameworks",
            confidence="high",
        )

        assert entry is not None
        assert entry.id is not None
        assert entry.content == "LangGraph is a library for building stateful multi-actor applications."
        assert entry.source_url == "https://langchain.com/langgraph"
        assert entry.topic == "AI frameworks"
        assert entry.confidence == "high"

    def test_add_entry_generates_id(self):
        """Test that IDs are generated consistently."""
        entry1 = self.kb.add_entry(
            content="Test content",
            source_url="https://example.com",
        )
        # Same content should generate same ID base
        # But deduplication should prevent adding
        entry2 = self.kb.add_entry(
            content="Test content",
            source_url="https://example.com",
            skip_duplicates=False,  # Force add to test ID generation
        )

        # IDs should be the same (same content + source)
        assert entry1.id == entry2.id

    def test_skip_duplicates(self):
        """Test that duplicate content is skipped."""
        entry1 = self.kb.add_entry(
            content="This is a unique finding about AI agents.",
            topic="AI",
        )
        entry2 = self.kb.add_entry(
            content="This is a unique finding about AI agents.",
            topic="AI",
            skip_duplicates=True,
        )

        assert entry1 is not None
        assert entry2 is None  # Should be skipped as duplicate
        assert self.kb.collection.count() == 1

    def test_search_returns_results(self):
        """Test that search returns relevant results."""
        # Add some entries
        self.kb.add_entry(
            content="LangGraph is built on LangChain for stateful agent applications.",
            topic="AI frameworks",
        )
        self.kb.add_entry(
            content="CrewAI uses role-based agents for task automation.",
            topic="AI frameworks",
        )
        self.kb.add_entry(
            content="Python is a popular programming language.",
            topic="programming",
        )

        # Search for AI frameworks
        results = self.kb.search("LangGraph agent framework", n_results=3)

        assert len(results) > 0
        assert results[0].score > 0
        # Most relevant should be about LangGraph
        assert "LangGraph" in results[0].entry.content

    def test_search_empty_database(self):
        """Test search on empty database returns empty list."""
        results = self.kb.search("anything")
        assert results == []

    def test_search_with_topic_filter(self):
        """Test search with topic filtering."""
        self.kb.add_entry(
            content="AI agents can perform complex tasks.",
            topic="AI",
        )
        self.kb.add_entry(
            content="Python is great for AI development.",
            topic="programming",
        )

        # Search with topic filter
        results = self.kb.search("AI", topic_filter="AI")

        assert len(results) == 1
        assert results[0].entry.topic == "AI"

    def test_get_entry(self):
        """Test retrieving a specific entry by ID."""
        entry = self.kb.add_entry(
            content="Test content for retrieval.",
            topic="test",
        )

        retrieved = self.kb.get_entry(entry.id)

        assert retrieved is not None
        assert retrieved.id == entry.id
        assert retrieved.content == entry.content

    def test_get_entry_not_found(self):
        """Test getting non-existent entry returns None."""
        result = self.kb.get_entry("non-existent-id")
        assert result is None

    def test_delete_entry(self):
        """Test deleting an entry."""
        entry = self.kb.add_entry(content="To be deleted.", topic="test")

        result = self.kb.delete_entry(entry.id)

        assert result is True
        assert self.kb.get_entry(entry.id) is None
        assert self.kb.collection.count() == 0

    def test_list_topics(self):
        """Test listing all topics."""
        self.kb.add_entry(content="Content 1", topic="AI")
        self.kb.add_entry(content="Content 2", topic="ML")
        self.kb.add_entry(content="Content 3", topic="AI")
        self.kb.add_entry(content="Content 4", topic="NLP")

        topics = self.kb.list_topics()

        assert len(topics) == 3
        assert "AI" in topics
        assert "ML" in topics
        assert "NLP" in topics

    def test_get_stats(self):
        """Test getting knowledge base statistics."""
        self.kb.add_entry(content="Content 1", topic="AI", confidence="high")
        self.kb.add_entry(content="Content 2", topic="AI", confidence="medium")
        self.kb.add_entry(content="Content 3", topic="ML", confidence="high")

        stats = self.kb.get_stats()

        assert stats["total_entries"] == 3
        assert stats["topics"]["AI"] == 2
        assert stats["topics"]["ML"] == 1
        assert stats["confidence_distribution"]["high"] == 2
        assert stats["confidence_distribution"]["medium"] == 1

    def test_clear(self):
        """Test clearing the knowledge base."""
        self.kb.add_entry(content="Content 1")
        self.kb.add_entry(content="Content 2")
        self.kb.add_entry(content="Content 3")

        count = self.kb.clear()

        assert count == 3
        assert self.kb.collection.count() == 0


class TestKnowledgeBasePersistence:
    """Tests for persistent storage."""

    def setup_method(self, method):
        """Create a temporary directory for persistence tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.collection_name = f"test_persist_{method.__name__}"

    def teardown_method(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        reset_knowledge_base()

    def test_persistence_across_instances(self):
        """Test that data persists across manager instances."""
        # Create first instance and add data
        kb1 = KnowledgeBaseManager(persist_directory=self.temp_dir, collection_name=self.collection_name)
        kb1.add_entry(content="Persistent content", topic="test")

        # Create second instance pointing to same directory
        kb2 = KnowledgeBaseManager(persist_directory=self.temp_dir, collection_name=self.collection_name)

        # Data should be available
        assert kb2.collection.count() == 1
        results = kb2.search("Persistent content")
        assert len(results) == 1
        assert results[0].entry.content == "Persistent content"


class TestGlobalKnowledgeBase:
    """Tests for the global knowledge base singleton."""

    def setup_method(self, method):
        """Reset knowledge base before each test."""
        reset_knowledge_base()
        # Use a unique test directory for each test
        self.test_dir = tempfile.mkdtemp()
        # Set environment variable to use test directory
        os.environ["RESEARCHCREW_KNOWLEDGE_PATH"] = self.test_dir

    def teardown_method(self):
        """Clean up after tests."""
        reset_knowledge_base()
        # Clean up test directory
        if hasattr(self, "test_dir") and os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)
        # Remove environment variable
        os.environ.pop("RESEARCHCREW_KNOWLEDGE_PATH", None)

    def test_get_knowledge_base_returns_same_instance(self):
        """Test that get_knowledge_base returns the same instance."""
        kb1 = get_knowledge_base()
        kb2 = get_knowledge_base()

        assert kb1 is kb2

    def test_reset_knowledge_base(self):
        """Test that reset creates a new instance."""
        kb1 = get_knowledge_base()
        kb1.add_entry(content="Test content for reset test")

        reset_knowledge_base()

        kb2 = get_knowledge_base()
        # Should be a different instance
        assert kb1 is not kb2


class TestKnowledgeEntry:
    """Tests for the KnowledgeEntry model."""

    def test_knowledge_entry_creation(self):
        """Test creating a KnowledgeEntry."""
        entry = KnowledgeEntry(
            id="test123",
            content="Test content",
            source_url="https://example.com",
            source_title="Example",
            topic="test",
            session_id="session-1",
            confidence="high",
            created_at="2025-01-18T10:00:00",
        )

        assert entry.id == "test123"
        assert entry.content == "Test content"
        assert entry.source_url == "https://example.com"
        assert entry.confidence == "high"

    def test_knowledge_entry_optional_fields(self):
        """Test KnowledgeEntry with optional fields omitted."""
        entry = KnowledgeEntry(
            id="test123",
            content="Test content",
            created_at="2025-01-18T10:00:00",
        )

        assert entry.source_url is None
        assert entry.topic is None
        assert entry.confidence is None


class TestSearchResult:
    """Tests for the SearchResult model."""

    def test_search_result_creation(self):
        """Test creating a SearchResult."""
        entry = KnowledgeEntry(
            id="test123",
            content="Test content",
            created_at="2025-01-18T10:00:00",
        )

        result = SearchResult(
            entry=entry,
            score=0.85,
            distance=0.15,
        )

        assert result.entry.id == "test123"
        assert result.score == 0.85
        assert result.distance == 0.15
