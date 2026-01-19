"""Tests for ResearchCrew Runner

Tests the runner's session integration for multi-turn conversations.
"""

from runner import ResearchCrewRunner, run_research
from utils.context_manager import reset_context_manager
from utils.session_manager import get_session_manager, reset_session_manager


class TestResearchCrewRunner:
    """Tests for the ResearchCrewRunner class."""

    def setup_method(self):
        """Reset session and context managers before each test."""
        reset_session_manager()
        reset_context_manager()
        self.runner = ResearchCrewRunner()

    def test_create_session(self):
        """Test creating a session through the runner."""
        session_id = self.runner.create_session(user_id="test_user")

        assert session_id is not None
        assert len(session_id) > 0

    def test_get_session(self):
        """Test getting a session through the runner."""
        session_id = self.runner.create_session(user_id="test_user")
        session = self.runner.get_session(session_id)

        assert session is not None
        assert session.session_id == session_id
        assert session.user_id == "test_user"

    def test_build_prompt_without_session(self):
        """Test prompt building without a session returns original query."""
        query = "What are AI agents?"
        result = self.runner.build_prompt_with_context(query)

        assert result == query

    def test_build_prompt_with_empty_session(self):
        """Test prompt building with new session includes system prompt and query."""
        session_id = self.runner.create_session()
        query = "What are AI agents?"

        result = self.runner.build_prompt_with_context(query, session_id)

        # New session has no history but includes formatting
        assert query in result
        assert "[Current Query]" in result

    def test_build_prompt_with_session_context(self):
        """Test prompt building includes session context."""
        session_id = self.runner.create_session()

        # Add some history to the session
        manager = get_session_manager()
        manager.add_conversation_turn(
            session_id=session_id,
            query="What are LLMs?",
            summary="LLMs are large language models.",
        )
        manager.update_session_context(
            session_id=session_id,
            topics=["LLMs"],
            key_facts=["LLMs can generate text"],
        )

        query = "Tell me more about their capabilities"
        result = self.runner.build_prompt_with_context(query, session_id)

        # Should include context sections
        assert "[Session Context]" in result
        assert "What are LLMs?" in result
        assert "Tell me more about their capabilities" in result
        assert "LLMs can generate text" in result

    def test_record_research_result(self):
        """Test recording research results in session."""
        session_id = self.runner.create_session()

        # Simulate a research result
        result = {
            "research": {
                "findings": [
                    {
                        "claim": "AI agents can reason",
                        "sources": [{"url": "https://example.com", "title": "Example"}],
                    },
                    {
                        "claim": "Agents use tools",
                        "sources": [{"url": "https://example2.com", "title": "Example2"}],
                    },
                ],
                "sub_queries": ["What is reasoning?", "How do tools work?"],
            },
            "report": {
                "content": "This is a comprehensive report about AI agents...",
                "sources_cited": 2,
            },
        }

        self.runner.record_research_result(
            session_id=session_id,
            query="What are AI agents?",
            result=result,
        )

        # Verify session was updated
        session = self.runner.get_session(session_id)
        assert len(session.conversation_history) == 1
        assert session.conversation_history[0].query == "What are AI agents?"
        assert session.conversation_history[0].findings_count == 2
        assert "AI agents can reason" in session.key_facts
        assert "https://example.com" in session.sources_used

    def test_run_creates_session(self):
        """Test that run creates a session when none provided."""
        result = self.runner.run("What are AI agents?")

        assert "session_id" in result
        assert result["session_id"] is not None
        assert result["turn_number"] == 1

    def test_run_uses_existing_session(self):
        """Test that run uses an existing session."""
        session_id = self.runner.create_session()
        result = self.runner.run("What are AI agents?", session_id=session_id)

        assert result["session_id"] == session_id

    def test_get_session_history(self):
        """Test getting session history."""
        session_id = self.runner.create_session()

        # Add some history
        manager = get_session_manager()
        manager.add_conversation_turn(
            session_id=session_id,
            query="First query",
            summary="First answer",
        )
        manager.add_conversation_turn(
            session_id=session_id,
            query="Second query",
            summary="Second answer",
        )

        history = self.runner.get_session_history(session_id)

        assert len(history) == 2
        assert history[0]["query"] == "First query"
        assert history[1]["query"] == "Second query"

    def test_get_session_stats(self):
        """Test getting session statistics."""
        session_id = self.runner.create_session()

        # Add some history
        manager = get_session_manager()
        manager.add_conversation_turn(
            session_id=session_id,
            query="Test query",
        )
        manager.update_session_context(
            session_id=session_id,
            topics=["AI"],
            key_facts=["Fact 1"],
        )

        stats = self.runner.get_session_stats(session_id)

        assert stats["session_id"] == session_id
        assert stats["total_turns"] == 1
        assert stats["topics_count"] == 1
        assert stats["facts_count"] == 1


class TestRunResearchFunction:
    """Tests for the run_research convenience function."""

    def setup_method(self):
        """Reset session and context managers before each test."""
        reset_session_manager()
        reset_context_manager()

    def test_run_research_creates_session(self):
        """Test run_research creates a session."""
        result = run_research("What are AI agents?")

        assert "session_id" in result
        assert result["session_id"] is not None

    def test_run_research_with_session_id(self):
        """Test run_research with provided session ID."""
        # First call creates session
        result1 = run_research("First query")
        session_id = result1["session_id"]

        # Second call uses same session
        result2 = run_research("Second query", session_id=session_id)

        assert result2["session_id"] == session_id


class TestMultiTurnConversation:
    """Integration tests for multi-turn conversation scenarios."""

    def setup_method(self):
        """Reset session and context managers before each test."""
        reset_session_manager()
        reset_context_manager()
        self.runner = ResearchCrewRunner()

    def test_follow_up_question_has_context(self):
        """Test that follow-up questions include previous context."""
        session_id = self.runner.create_session()

        # Simulate first turn
        manager = get_session_manager()
        manager.add_conversation_turn(
            session_id=session_id,
            query="What are the main AI frameworks in 2025?",
            summary="The main AI frameworks include LangGraph, CrewAI, and Google ADK.",
            findings_count=3,
        )
        manager.update_session_context(
            session_id=session_id,
            topics=["AI frameworks", "LangGraph", "CrewAI", "Google ADK"],
            key_facts=[
                "LangGraph is built on LangChain",
                "CrewAI focuses on role-based agents",
                "Google ADK integrates with Vertex AI",
            ],
        )

        # Build follow-up prompt
        follow_up = self.runner.build_prompt_with_context("Tell me more about Google ADK", session_id)

        # Should include context from previous turn
        assert "AI frameworks" in follow_up
        assert "Google ADK integrates with Vertex AI" in follow_up
        assert "Tell me more about Google ADK" in follow_up

    def test_session_accumulates_knowledge(self):
        """Test that session accumulates knowledge across turns."""
        session_id = self.runner.create_session()
        manager = get_session_manager()

        # First turn
        manager.add_conversation_turn(
            session_id=session_id,
            query="What is LangGraph?",
        )
        manager.update_session_context(
            session_id=session_id,
            topics=["LangGraph"],
            key_facts=["LangGraph is for building agents"],
        )

        # Second turn
        manager.add_conversation_turn(
            session_id=session_id,
            query="What is CrewAI?",
        )
        manager.update_session_context(
            session_id=session_id,
            topics=["CrewAI"],
            key_facts=["CrewAI uses role-based agents"],
        )

        # Third turn
        manager.add_conversation_turn(
            session_id=session_id,
            query="Compare them",
        )

        # Session should have accumulated all knowledge
        session = manager.get_session(session_id)
        assert "LangGraph" in session.topics_researched
        assert "CrewAI" in session.topics_researched
        assert "LangGraph is for building agents" in session.key_facts
        assert "CrewAI uses role-based agents" in session.key_facts
        assert len(session.conversation_history) == 3
