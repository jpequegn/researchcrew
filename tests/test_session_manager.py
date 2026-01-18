"""Tests for Session Manager

Tests the session management functionality for multi-turn conversations.
"""

import pytest
from utils.session_manager import (
    SessionManager,
    SessionState,
    ConversationTurn,
    get_session_manager,
    reset_session_manager,
)


class TestSessionManager:
    """Tests for the SessionManager class."""

    def setup_method(self):
        """Reset session manager before each test."""
        reset_session_manager()
        self.manager = SessionManager()

    def test_create_session(self):
        """Test creating a new session."""
        session = self.manager.create_session(user_id="test_user")

        assert session.session_id is not None
        assert session.user_id == "test_user"
        assert session.app_name == "researchcrew"
        assert len(session.conversation_history) == 0
        assert session.created_at is not None
        assert session.last_updated is not None

    def test_create_session_with_custom_id(self):
        """Test creating a session with a custom ID."""
        session = self.manager.create_session(
            user_id="test_user", session_id="custom-session-123"
        )

        assert session.session_id == "custom-session-123"
        assert session.user_id == "test_user"

    def test_get_session(self):
        """Test retrieving an existing session."""
        created = self.manager.create_session(user_id="test_user")
        retrieved = self.manager.get_session(created.session_id)

        assert retrieved is not None
        assert retrieved.session_id == created.session_id
        assert retrieved.user_id == created.user_id

    def test_get_session_not_found(self):
        """Test retrieving a non-existent session returns None."""
        result = self.manager.get_session("non-existent-id")
        assert result is None

    def test_get_or_create_session_creates_new(self):
        """Test get_or_create creates a new session when not found."""
        session = self.manager.get_or_create_session(
            session_id="new-session", user_id="test_user"
        )

        assert session.session_id == "new-session"
        assert session.user_id == "test_user"

    def test_get_or_create_session_returns_existing(self):
        """Test get_or_create returns existing session."""
        created = self.manager.create_session(user_id="test_user")
        retrieved = self.manager.get_or_create_session(session_id=created.session_id)

        assert retrieved.session_id == created.session_id

    def test_add_conversation_turn(self):
        """Test adding a conversation turn to a session."""
        session = self.manager.create_session(user_id="test_user")

        turn = self.manager.add_conversation_turn(
            session_id=session.session_id,
            query="What are AI agents?",
            summary="AI agents are autonomous systems...",
            findings_count=5,
            sources_count=3,
        )

        assert turn is not None
        assert turn.turn_id == 1
        assert turn.query == "What are AI agents?"
        assert turn.summary == "AI agents are autonomous systems..."
        assert turn.findings_count == 5
        assert turn.sources_count == 3

        # Verify session was updated
        updated_session = self.manager.get_session(session.session_id)
        assert len(updated_session.conversation_history) == 1
        assert updated_session.last_query == "What are AI agents?"
        assert updated_session.last_findings_summary == "AI agents are autonomous systems..."

    def test_add_multiple_conversation_turns(self):
        """Test adding multiple turns to track turn numbers."""
        session = self.manager.create_session(user_id="test_user")

        turn1 = self.manager.add_conversation_turn(
            session_id=session.session_id,
            query="First question",
            summary="First answer",
        )
        turn2 = self.manager.add_conversation_turn(
            session_id=session.session_id,
            query="Second question",
            summary="Second answer",
        )
        turn3 = self.manager.add_conversation_turn(
            session_id=session.session_id,
            query="Third question",
            summary="Third answer",
        )

        assert turn1.turn_id == 1
        assert turn2.turn_id == 2
        assert turn3.turn_id == 3

        updated_session = self.manager.get_session(session.session_id)
        assert len(updated_session.conversation_history) == 3
        assert updated_session.last_query == "Third question"

    def test_update_session_context(self):
        """Test updating session context with topics and facts."""
        session = self.manager.create_session(user_id="test_user")

        result = self.manager.update_session_context(
            session_id=session.session_id,
            topics=["AI agents", "LLMs"],
            key_facts=["Fact 1", "Fact 2"],
            sources=["https://example.com/1", "https://example.com/2"],
            findings_summary="Summary of findings",
        )

        assert result is True

        updated_session = self.manager.get_session(session.session_id)
        assert "AI agents" in updated_session.topics_researched
        assert "LLMs" in updated_session.topics_researched
        assert "Fact 1" in updated_session.key_facts
        assert "https://example.com/1" in updated_session.sources_used
        assert updated_session.last_findings_summary == "Summary of findings"

    def test_update_session_context_no_duplicates(self):
        """Test that duplicate topics/facts/sources are not added."""
        session = self.manager.create_session(user_id="test_user")

        # Add initial context
        self.manager.update_session_context(
            session_id=session.session_id,
            topics=["AI agents"],
            key_facts=["Fact 1"],
            sources=["https://example.com/1"],
        )

        # Add same context again
        self.manager.update_session_context(
            session_id=session.session_id,
            topics=["AI agents", "New topic"],
            key_facts=["Fact 1", "Fact 2"],
            sources=["https://example.com/1", "https://example.com/2"],
        )

        updated_session = self.manager.get_session(session.session_id)

        # Should only have one of each duplicate
        assert updated_session.topics_researched.count("AI agents") == 1
        assert updated_session.key_facts.count("Fact 1") == 1
        assert updated_session.sources_used.count("https://example.com/1") == 1

        # But should have the new ones
        assert "New topic" in updated_session.topics_researched
        assert "Fact 2" in updated_session.key_facts
        assert "https://example.com/2" in updated_session.sources_used

    def test_get_session_context_for_prompt_empty(self):
        """Test context generation for new session returns empty string."""
        session = self.manager.create_session(user_id="test_user")
        context = self.manager.get_session_context_for_prompt(session.session_id)

        assert context == ""

    def test_get_session_context_for_prompt_with_history(self):
        """Test context generation includes conversation history."""
        session = self.manager.create_session(user_id="test_user")

        # Add some history
        self.manager.add_conversation_turn(
            session_id=session.session_id,
            query="What are AI agents?",
            summary="AI agents are autonomous systems that can perform tasks.",
        )

        self.manager.update_session_context(
            session_id=session.session_id,
            topics=["AI agents", "autonomy"],
            key_facts=["AI agents can reason", "Agents use tools"],
        )

        context = self.manager.get_session_context_for_prompt(session.session_id)

        assert "Previous Research" in context
        assert "What are AI agents?" in context
        assert "Key Facts Discovered" in context
        assert "AI agents can reason" in context
        assert "Topics Researched" in context

    def test_delete_session(self):
        """Test deleting a session."""
        session = self.manager.create_session(user_id="test_user")
        session_id = session.session_id

        result = self.manager.delete_session(session_id)
        assert result is True

        # Should no longer exist
        assert self.manager.get_session(session_id) is None

    def test_delete_session_not_found(self):
        """Test deleting a non-existent session returns False."""
        result = self.manager.delete_session("non-existent")
        assert result is False

    def test_list_sessions(self):
        """Test listing all sessions."""
        self.manager.create_session(user_id="user1")
        self.manager.create_session(user_id="user2")
        self.manager.create_session(user_id="user1")

        all_sessions = self.manager.list_sessions()
        assert len(all_sessions) == 3

        user1_sessions = self.manager.list_sessions(user_id="user1")
        assert len(user1_sessions) == 2

    def test_get_session_stats(self):
        """Test getting session statistics."""
        session = self.manager.create_session(user_id="test_user")

        # Add some data
        self.manager.add_conversation_turn(
            session_id=session.session_id,
            query="Query 1",
        )
        self.manager.add_conversation_turn(
            session_id=session.session_id,
            query="Query 2",
        )
        self.manager.update_session_context(
            session_id=session.session_id,
            topics=["Topic 1", "Topic 2"],
            key_facts=["Fact 1"],
            sources=["https://example.com"],
        )

        stats = self.manager.get_session_stats(session.session_id)

        assert stats["session_id"] == session.session_id
        assert stats["total_turns"] == 2
        assert stats["topics_count"] == 2
        assert stats["facts_count"] == 1
        assert stats["sources_count"] == 1


class TestGlobalSessionManager:
    """Tests for the global session manager singleton."""

    def setup_method(self):
        """Reset session manager before each test."""
        reset_session_manager()

    def test_get_session_manager_returns_same_instance(self):
        """Test that get_session_manager returns the same instance."""
        manager1 = get_session_manager()
        manager2 = get_session_manager()

        assert manager1 is manager2

    def test_reset_session_manager(self):
        """Test that reset creates a new instance."""
        manager1 = get_session_manager()
        manager1.create_session(user_id="test")

        reset_session_manager()

        manager2 = get_session_manager()
        # Should be a new instance with no sessions
        assert len(manager2.list_sessions()) == 0


class TestConversationTurn:
    """Tests for the ConversationTurn model."""

    def test_conversation_turn_creation(self):
        """Test creating a ConversationTurn."""
        turn = ConversationTurn(
            turn_id=1,
            timestamp="2025-01-18T10:00:00",
            query="Test query",
            summary="Test summary",
            findings_count=5,
            sources_count=3,
        )

        assert turn.turn_id == 1
        assert turn.query == "Test query"
        assert turn.summary == "Test summary"
        assert turn.findings_count == 5
        assert turn.sources_count == 3

    def test_conversation_turn_optional_fields(self):
        """Test ConversationTurn with optional fields omitted."""
        turn = ConversationTurn(
            turn_id=1,
            timestamp="2025-01-18T10:00:00",
            query="Test query",
        )

        assert turn.summary is None
        assert turn.findings_count == 0
        assert turn.sources_count == 0


class TestSessionState:
    """Tests for the SessionState model."""

    def test_session_state_creation(self):
        """Test creating a SessionState."""
        state = SessionState(
            session_id="test-session",
            created_at="2025-01-18T10:00:00",
            last_updated="2025-01-18T10:00:00",
        )

        assert state.session_id == "test-session"
        assert state.app_name == "researchcrew"
        assert state.user_id == "default_user"
        assert state.conversation_history == []
        assert state.topics_researched == []
        assert state.key_facts == []
        assert state.sources_used == []

    def test_session_state_with_history(self):
        """Test SessionState with conversation history."""
        turn = ConversationTurn(
            turn_id=1,
            timestamp="2025-01-18T10:00:00",
            query="Test query",
        )

        state = SessionState(
            session_id="test-session",
            created_at="2025-01-18T10:00:00",
            last_updated="2025-01-18T10:00:00",
            conversation_history=[turn],
            topics_researched=["AI", "ML"],
            key_facts=["Fact 1"],
        )

        assert len(state.conversation_history) == 1
        assert state.conversation_history[0].query == "Test query"
        assert "AI" in state.topics_researched
