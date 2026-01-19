"""Integration Tests for ResearchCrew Workflows

Tests end-to-end workflows and agent coordination.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from runner import ResearchCrewRunner, run_research
from utils.session_manager import reset_session_manager, get_session_manager
from utils.context_manager import reset_context_manager


class TestResearchWorkflow:
    """Tests for the research workflow integration."""

    def setup_method(self):
        """Reset managers before each test."""
        reset_session_manager()
        reset_context_manager()

    def test_single_query_workflow(self):
        """Test a single research query workflow."""
        runner = ResearchCrewRunner()
        result = runner.run("What are AI agents?")

        # Should return expected structure
        assert "session_id" in result
        assert "turn_number" in result
        assert result["turn_number"] == 1
        assert "query" in result

    def test_multi_turn_workflow(self):
        """Test multi-turn conversation workflow."""
        runner = ResearchCrewRunner()

        # First turn
        result1 = runner.run("What are AI agent frameworks?")
        session_id = result1["session_id"]
        assert result1["turn_number"] == 1

        # Add a conversation turn to simulate first turn completion
        # (run() returns turn number but doesn't add to history - that's done by record_research_result)
        manager = get_session_manager()
        manager.add_conversation_turn(
            session_id=session_id,
            query="What are AI agent frameworks?",
            summary="AI agent frameworks include LangGraph, CrewAI, and AutoGen.",
        )

        # Second turn using same session
        result2 = runner.run("Tell me more about LangGraph", session_id=session_id)

        assert result2["session_id"] == session_id
        assert result2["turn_number"] == 2

    def test_session_context_accumulation(self):
        """Test that session context accumulates across turns."""
        runner = ResearchCrewRunner()
        session_id = runner.create_session()

        # Simulate adding context
        manager = get_session_manager()
        manager.add_conversation_turn(
            session_id=session_id,
            query="What is LangGraph?",
            summary="LangGraph is a graph-based agent framework.",
        )
        manager.update_session_context(
            session_id=session_id,
            topics=["LangGraph", "agents"],
            key_facts=["LangGraph uses graph-based workflows"],
        )

        # Build prompt with context
        prompt = runner.build_prompt_with_context(
            "How does it compare to CrewAI?",
            session_id=session_id,
        )

        # Context should be included
        assert "LangGraph" in prompt
        assert "graph-based" in prompt.lower()

    def test_context_compression_in_long_sessions(self):
        """Test that context is compressed in long sessions."""
        runner = ResearchCrewRunner()
        session_id = runner.create_session()

        # Add many turns to simulate long session
        manager = get_session_manager()
        for i in range(20):
            manager.add_conversation_turn(
                session_id=session_id,
                query=f"Question {i} about AI agents and frameworks " * 10,
                summary=f"Answer {i} with detailed information about topic " * 10,
            )
            manager.update_session_context(
                session_id=session_id,
                key_facts=[f"Fact {i}: Important detail about the topic " * 5],
            )

        # Should still be able to build context
        prompt = runner.build_prompt_with_context(
            "Summary question",
            session_id=session_id,
            auto_compress=True,
        )

        assert len(prompt) > 0
        # Usage should be tracked
        usage = runner.get_context_usage()
        assert usage is not None


class TestSessionManagement:
    """Tests for session management in workflows."""

    def setup_method(self):
        """Reset managers before each test."""
        reset_session_manager()
        reset_context_manager()

    def test_create_session_with_user_id(self):
        """Test creating session with user ID."""
        runner = ResearchCrewRunner()
        session_id = runner.create_session(user_id="user-123")

        session = runner.get_session(session_id)
        assert session is not None
        assert session.user_id == "user-123"

    def test_get_session_history(self):
        """Test getting session history."""
        runner = ResearchCrewRunner()
        session_id = runner.create_session()

        # Add history
        manager = get_session_manager()
        manager.add_conversation_turn(
            session_id=session_id,
            query="First question",
            summary="First answer",
        )
        manager.add_conversation_turn(
            session_id=session_id,
            query="Second question",
            summary="Second answer",
        )

        history = runner.get_session_history(session_id)

        assert len(history) == 2
        assert history[0]["query"] == "First question"
        assert history[1]["query"] == "Second question"

    def test_get_session_stats(self):
        """Test getting session statistics."""
        runner = ResearchCrewRunner()
        session_id = runner.create_session()

        # Add some activity
        manager = get_session_manager()
        manager.add_conversation_turn(
            session_id=session_id,
            query="Test query",
        )
        manager.update_session_context(
            session_id=session_id,
            topics=["Topic 1", "Topic 2"],
            key_facts=["Fact 1"],
        )

        stats = runner.get_session_stats(session_id)

        assert stats["session_id"] == session_id
        assert stats["total_turns"] == 1
        assert stats["topics_count"] == 2
        assert stats["facts_count"] == 1


class TestResultRecording:
    """Tests for recording research results."""

    def setup_method(self):
        """Reset managers before each test."""
        reset_session_manager()
        reset_context_manager()

    def test_record_research_result(self):
        """Test recording a research result."""
        runner = ResearchCrewRunner()
        session_id = runner.create_session()

        # Simulate research result
        result = {
            "research": {
                "findings": [
                    {
                        "claim": "AI agents can use tools",
                        "sources": [{"url": "https://example.com", "title": "Source"}],
                    }
                ],
                "sub_queries": ["How do tools work?"],
            },
            "report": {
                "content": "Research report about AI agents...",
                "sources_cited": 1,
            },
        }

        runner.record_research_result(
            session_id=session_id,
            query="What are AI agents?",
            result=result,
        )

        # Check session was updated
        session = runner.get_session(session_id)
        assert len(session.conversation_history) == 1
        assert "AI agents can use tools" in session.key_facts
        assert "https://example.com" in session.sources_used


class TestConvenienceFunction:
    """Tests for the run_research convenience function."""

    def setup_method(self):
        """Reset managers before each test."""
        reset_session_manager()
        reset_context_manager()

    def test_run_research_creates_session(self):
        """Test that run_research creates a new session."""
        result = run_research("What are AI agents?")

        assert "session_id" in result
        assert result["session_id"] is not None

    def test_run_research_with_existing_session(self):
        """Test run_research with existing session."""
        # Create first session
        result1 = run_research("First query")
        session_id = result1["session_id"]

        # Continue with same session
        result2 = run_research("Follow-up query", session_id=session_id)

        assert result2["session_id"] == session_id


class TestContextEstimation:
    """Tests for context token estimation."""

    def setup_method(self):
        """Reset managers before each test."""
        reset_session_manager()
        reset_context_manager()

    def test_estimate_context_tokens(self):
        """Test estimating context tokens."""
        runner = ResearchCrewRunner()
        session_id = runner.create_session()

        # Add some context
        manager = get_session_manager()
        manager.add_conversation_turn(
            session_id=session_id,
            query="Previous question about AI",
            summary="Summary of previous answer",
        )
        manager.update_session_context(
            session_id=session_id,
            key_facts=["Important fact 1", "Important fact 2"],
        )

        estimate = runner.estimate_context_tokens(
            session_id=session_id,
            query="New question about agents",
        )

        assert "total_tokens" in estimate
        assert "query_tokens" in estimate
        assert "history_tokens" in estimate
        assert "facts_tokens" in estimate
        assert "remaining_tokens" in estimate
        assert estimate["total_tokens"] > 0

    def test_estimate_without_session(self):
        """Test estimation without session."""
        runner = ResearchCrewRunner()

        estimate = runner.estimate_context_tokens(
            session_id="nonexistent",
            query="Test query",
        )

        assert estimate["total_tokens"] > 0
        assert estimate["history_tokens"] == 0
        assert estimate["facts_tokens"] == 0


class TestErrorHandling:
    """Tests for error handling in workflows."""

    def setup_method(self):
        """Reset managers before each test."""
        reset_session_manager()
        reset_context_manager()

    def test_nonexistent_session_handling(self):
        """Test handling of nonexistent session."""
        runner = ResearchCrewRunner()

        session = runner.get_session("nonexistent-session-id")
        assert session is None

    def test_empty_query_handling(self):
        """Test handling of empty query."""
        runner = ResearchCrewRunner()
        result = runner.run("")

        # Should still return valid structure
        assert "session_id" in result

    def test_build_prompt_without_session(self):
        """Test building prompt without session context."""
        runner = ResearchCrewRunner()
        query = "What are AI agents?"

        prompt = runner.build_prompt_with_context(query)

        # Should return original query
        assert prompt == query


class TestAgentImports:
    """Tests that verify all agents can be imported."""

    def test_import_researcher(self):
        """Test researcher agent import."""
        from agents.researcher import researcher_agent
        assert researcher_agent is not None

    def test_import_synthesizer(self):
        """Test synthesizer agent import."""
        from agents.synthesizer import synthesizer_agent
        assert synthesizer_agent is not None

    def test_import_fact_checker(self):
        """Test fact checker agent import."""
        from agents.fact_checker import fact_checker_agent
        assert fact_checker_agent is not None

    def test_import_writer(self):
        """Test writer agent import."""
        from agents.writer import writer_agent
        assert writer_agent is not None

    def test_import_orchestrator(self):
        """Test orchestrator agent import."""
        from agents.orchestrator import orchestrator_agent
        assert orchestrator_agent is not None


class TestToolImports:
    """Tests that verify all tools can be imported."""

    def test_import_search_tools(self):
        """Test search tools import."""
        from tools.search import web_search, read_url
        assert web_search is not None
        assert read_url is not None

    def test_import_knowledge_tools(self):
        """Test knowledge tools import."""
        from tools.knowledge import knowledge_search, save_to_knowledge
        assert knowledge_search is not None
        assert save_to_knowledge is not None
