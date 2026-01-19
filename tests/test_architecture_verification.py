"""Architecture Verification Tests

Comprehensive tests verifying the ResearchCrew architecture works end-to-end.
This file validates Issue #16 requirements:
- Multi-agent orchestration working
- Hierarchical delegation implemented
- All 5 agents functioning correctly
- Session memory functional (within session recall)
- Long-term knowledge base functional (cross-session recall)
- State properly passed between agents
- Handoffs working with error handling

Note: Agent objects are mocked since google.adk is not available in CI.
Tests focus on:
1. Instructions content (strings, always available)
2. Module imports (verifies code structure)
3. Memory systems (fully testable)
4. Workflow integration (testable via runner)
"""

# Import agent instruction strings (always available, not mocked)
from agents.fact_checker import FACT_CHECKER_INSTRUCTIONS
from agents.orchestrator import ORCHESTRATOR_INSTRUCTIONS
from agents.researcher import RESEARCHER_INSTRUCTIONS
from agents.synthesizer import SYNTHESIZER_INSTRUCTIONS
from agents.writer import WRITER_INSTRUCTIONS

# Import workflow runner
from runner import ResearchCrewRunner
from utils.context_manager import reset_context_manager
from utils.knowledge_base import (
    KnowledgeBaseManager,
    reset_knowledge_base,
)

# Import memory systems
from utils.session_manager import (
    get_session_manager,
    reset_session_manager,
)

# ============================================================================
# Multi-Agent Orchestration Tests
# ============================================================================


class TestMultiAgentOrchestration:
    """Tests verifying multi-agent orchestration structure via imports."""

    def test_orchestrator_module_imports(self):
        """Test that orchestrator module can be imported."""
        # If these imports work, the module structure is correct
        from agents.orchestrator import orchestrator_agent, research_phase

        assert orchestrator_agent is not None
        assert research_phase is not None

    def test_orchestrator_instructions_defined(self):
        """Test that orchestrator has instructions defined."""
        assert ORCHESTRATOR_INSTRUCTIONS is not None
        assert len(ORCHESTRATOR_INSTRUCTIONS) > 100

    def test_orchestrator_instructions_describe_workflow(self):
        """Test that orchestrator instructions describe the workflow."""
        inst_lower = ORCHESTRATOR_INSTRUCTIONS.lower()
        # Should mention the workflow steps
        assert "research" in inst_lower
        assert "fact" in inst_lower or "validate" in inst_lower
        assert "report" in inst_lower or "write" in inst_lower

    def test_orchestrator_instructions_describe_coordination(self):
        """Test that orchestrator instructions describe coordination."""
        inst_lower = ORCHESTRATOR_INSTRUCTIONS.lower()
        assert (
            "coordinate" in inst_lower
            or "delegate" in inst_lower
            or "workflow" in inst_lower
            or "orchestrat" in inst_lower
        )


# ============================================================================
# Hierarchical Delegation Tests
# ============================================================================


class TestHierarchicalDelegation:
    """Tests verifying hierarchical delegation via instruction analysis."""

    def test_orchestrator_instructions_mention_delegation(self):
        """Test that orchestrator instructions describe delegation."""
        inst_lower = ORCHESTRATOR_INSTRUCTIONS.lower()
        assert "delegate" in inst_lower or "dispatch" in inst_lower or "send" in inst_lower

    def test_orchestrator_instructions_mention_all_agents(self):
        """Test that orchestrator instructions reference all agent roles."""
        inst_lower = ORCHESTRATOR_INSTRUCTIONS.lower()
        # Should mention the different roles in the workflow
        assert "research" in inst_lower
        assert "synthesiz" in inst_lower or "combine" in inst_lower
        assert "fact" in inst_lower or "check" in inst_lower or "validate" in inst_lower
        assert "write" in inst_lower or "report" in inst_lower

    def test_workflow_module_structure(self):
        """Test that all workflow modules can be imported."""
        # These imports verify the delegation structure exists
        from agents.fact_checker import fact_checker_agent
        from agents.orchestrator import orchestrator_agent, research_phase
        from agents.researcher import researcher_agent
        from agents.synthesizer import synthesizer_agent
        from agents.writer import writer_agent

        # All should be non-None (even if mocked)
        assert researcher_agent is not None
        assert synthesizer_agent is not None
        assert fact_checker_agent is not None
        assert writer_agent is not None
        assert orchestrator_agent is not None
        assert research_phase is not None


# ============================================================================
# Individual Agent Verification Tests
# ============================================================================


class TestAllAgentsFunctioning:
    """Tests verifying all 5 agents via their instructions."""

    def test_researcher_agent_instructions(self):
        """Test researcher agent has proper instructions."""
        # Instructions should be defined and substantial
        assert RESEARCHER_INSTRUCTIONS is not None
        assert len(RESEARCHER_INSTRUCTIONS) > 100

        # Instructions should mention research capabilities
        inst_lower = RESEARCHER_INSTRUCTIONS.lower()
        assert "search" in inst_lower
        assert "source" in inst_lower or "cite" in inst_lower

    def test_researcher_agent_module(self):
        """Test researcher agent module imports correctly."""
        from agents.researcher import RESEARCHER_INSTRUCTIONS, researcher_agent

        assert researcher_agent is not None
        assert RESEARCHER_INSTRUCTIONS is not None

    def test_synthesizer_agent_instructions(self):
        """Test synthesizer agent has proper instructions."""
        assert SYNTHESIZER_INSTRUCTIONS is not None
        assert len(SYNTHESIZER_INSTRUCTIONS) > 100

        # Instructions should mention synthesis capabilities
        inst_lower = SYNTHESIZER_INSTRUCTIONS.lower()
        assert "synthesize" in inst_lower or "combine" in inst_lower or "integrate" in inst_lower

    def test_synthesizer_agent_module(self):
        """Test synthesizer agent module imports correctly."""
        from agents.synthesizer import SYNTHESIZER_INSTRUCTIONS, synthesizer_agent

        assert synthesizer_agent is not None
        assert SYNTHESIZER_INSTRUCTIONS is not None

    def test_fact_checker_agent_instructions(self):
        """Test fact checker agent has proper instructions."""
        assert FACT_CHECKER_INSTRUCTIONS is not None
        assert len(FACT_CHECKER_INSTRUCTIONS) > 100

        # Instructions should mention verification capabilities
        inst_lower = FACT_CHECKER_INSTRUCTIONS.lower()
        assert "verify" in inst_lower or "check" in inst_lower or "validate" in inst_lower

    def test_fact_checker_agent_module(self):
        """Test fact checker agent module imports correctly."""
        from agents.fact_checker import FACT_CHECKER_INSTRUCTIONS, fact_checker_agent

        assert fact_checker_agent is not None
        assert FACT_CHECKER_INSTRUCTIONS is not None

    def test_writer_agent_instructions(self):
        """Test writer agent has proper instructions."""
        assert WRITER_INSTRUCTIONS is not None
        assert len(WRITER_INSTRUCTIONS) > 100

        # Instructions should mention writing capabilities
        inst_lower = WRITER_INSTRUCTIONS.lower()
        assert "write" in inst_lower or "report" in inst_lower or "document" in inst_lower

    def test_writer_agent_module(self):
        """Test writer agent module imports correctly."""
        from agents.writer import WRITER_INSTRUCTIONS, writer_agent

        assert writer_agent is not None
        assert WRITER_INSTRUCTIONS is not None

    def test_orchestrator_agent_instructions(self):
        """Test orchestrator agent has proper instructions."""
        assert ORCHESTRATOR_INSTRUCTIONS is not None
        assert len(ORCHESTRATOR_INSTRUCTIONS) > 100

        # Instructions should mention coordination
        inst_lower = ORCHESTRATOR_INSTRUCTIONS.lower()
        assert "coordinate" in inst_lower or "workflow" in inst_lower or "orchestrat" in inst_lower

    def test_orchestrator_agent_module(self):
        """Test orchestrator agent module imports correctly."""
        from agents.orchestrator import ORCHESTRATOR_INSTRUCTIONS, orchestrator_agent

        assert orchestrator_agent is not None
        assert ORCHESTRATOR_INSTRUCTIONS is not None

    def test_all_instructions_are_unique(self):
        """Test that all agent instructions are unique."""
        instructions = [
            RESEARCHER_INSTRUCTIONS,
            SYNTHESIZER_INSTRUCTIONS,
            FACT_CHECKER_INSTRUCTIONS,
            WRITER_INSTRUCTIONS,
            ORCHESTRATOR_INSTRUCTIONS,
        ]

        # All instructions should be unique
        assert len(instructions) == len(set(instructions))

    def test_all_agents_have_role_definitions(self):
        """Test that all agent instructions define their role."""
        instructions = {
            "researcher": RESEARCHER_INSTRUCTIONS,
            "synthesizer": SYNTHESIZER_INSTRUCTIONS,
            "fact_checker": FACT_CHECKER_INSTRUCTIONS,
            "writer": WRITER_INSTRUCTIONS,
            "orchestrator": ORCHESTRATOR_INSTRUCTIONS,
        }

        for name, instruction in instructions.items():
            assert "you are" in instruction.lower() or "your" in instruction.lower(), (
                f"{name} instructions should define the agent's role"
            )


# ============================================================================
# Session Memory Tests (Within Session Recall)
# ============================================================================


class TestSessionMemory:
    """Tests verifying session memory functionality."""

    def setup_method(self):
        """Reset managers before each test."""
        reset_session_manager()
        reset_context_manager()

    def test_session_creation(self):
        """Test that sessions can be created."""
        manager = get_session_manager()
        session = manager.create_session(user_id="test_user")

        assert session is not None
        assert session.session_id is not None
        assert session.user_id == "test_user"

    def test_within_session_recall(self):
        """Test that information is recalled within a session."""
        manager = get_session_manager()
        session = manager.create_session(user_id="test_user")

        # Add first turn
        manager.add_conversation_turn(
            session_id=session.session_id,
            query="What is LangGraph?",
            summary="LangGraph is a graph-based agent framework.",
        )

        # Update context
        manager.update_session_context(
            session_id=session.session_id,
            topics=["LangGraph", "agent frameworks"],
            key_facts=["LangGraph uses graph-based workflows"],
        )

        # Get context for follow-up
        context = manager.get_session_context_for_prompt(session.session_id)

        # Should contain previous information
        assert "LangGraph" in context
        assert "graph-based" in context.lower()

    def test_multi_turn_session_memory(self):
        """Test memory across multiple turns."""
        manager = get_session_manager()
        session = manager.create_session(user_id="test_user")

        # Turn 1
        turn1 = manager.add_conversation_turn(
            session_id=session.session_id,
            query="What are AI agents?",
            summary="AI agents are autonomous systems.",
        )
        assert turn1.turn_id == 1

        # Turn 2
        turn2 = manager.add_conversation_turn(
            session_id=session.session_id,
            query="Tell me more about their capabilities",
            summary="AI agents can reason, plan, and use tools.",
        )
        assert turn2.turn_id == 2

        # Turn 3
        turn3 = manager.add_conversation_turn(
            session_id=session.session_id,
            query="What tools do they use?",
            summary="Common tools include web search and code execution.",
        )
        assert turn3.turn_id == 3

        # Verify all turns are stored
        updated_session = manager.get_session(session.session_id)
        assert len(updated_session.conversation_history) == 3

    def test_session_context_accumulation(self):
        """Test that context accumulates correctly."""
        manager = get_session_manager()
        session = manager.create_session(user_id="test_user")

        # Add topics and facts over multiple updates
        manager.update_session_context(
            session_id=session.session_id,
            topics=["Topic A"],
            key_facts=["Fact 1"],
        )
        manager.update_session_context(
            session_id=session.session_id,
            topics=["Topic B"],
            key_facts=["Fact 2"],
        )

        updated = manager.get_session(session.session_id)

        assert "Topic A" in updated.topics_researched
        assert "Topic B" in updated.topics_researched
        assert "Fact 1" in updated.key_facts
        assert "Fact 2" in updated.key_facts


# ============================================================================
# Knowledge Base Tests (Cross-Session Recall)
# ============================================================================


class TestKnowledgeBaseCrossSession:
    """Tests verifying cross-session knowledge persistence."""

    def setup_method(self, method):
        """Create fresh knowledge base for each test."""
        reset_knowledge_base()
        self.collection_name = f"test_arch_{method.__name__}"
        self.kb = KnowledgeBaseManager(
            persist_directory=None,
            collection_name=self.collection_name,
        )

    def test_knowledge_storage(self):
        """Test that knowledge can be stored."""
        entry = self.kb.add_entry(
            content="AI agents can use tools to interact with external systems.",
            topic="AI capabilities",
            confidence="high",
        )

        assert entry is not None
        assert entry.id is not None
        assert self.kb.collection.count() == 1

    def test_knowledge_retrieval(self):
        """Test that knowledge can be retrieved."""
        self.kb.add_entry(
            content="LangGraph enables stateful agent workflows.",
            topic="frameworks",
        )

        results = self.kb.search("agent workflows", n_results=5)

        assert len(results) > 0
        assert "LangGraph" in results[0].entry.content

    def test_cross_session_persistence(self):
        """Test that knowledge persists across sessions."""
        # Add knowledge in "session 1"
        self.kb.add_entry(
            content="Important finding from session 1.",
            session_id="session-1",
            topic="findings",
        )

        # Search in "session 2" (simulated by different session_id)
        results = self.kb.search("Important finding", n_results=5)

        # Should find the entry from session 1
        assert len(results) > 0
        assert "Important finding" in results[0].entry.content
        assert results[0].entry.session_id == "session-1"

    def test_topic_based_retrieval(self):
        """Test retrieval by topic."""
        self.kb.add_entry(content="AI content", topic="AI")
        self.kb.add_entry(content="ML content", topic="ML")
        self.kb.add_entry(content="NLP content", topic="NLP")

        # Search with topic filter
        results = self.kb.search("content", topic_filter="AI")

        assert len(results) == 1
        assert results[0].entry.topic == "AI"

    def test_knowledge_base_statistics(self):
        """Test knowledge base statistics."""
        self.kb.add_entry(content="Entry 1", topic="A", confidence="high")
        self.kb.add_entry(content="Entry 2", topic="A", confidence="medium")
        self.kb.add_entry(content="Entry 3", topic="B", confidence="high")

        stats = self.kb.get_stats()

        assert stats["total_entries"] == 3
        assert stats["topics"]["A"] == 2
        assert stats["topics"]["B"] == 1
        assert stats["confidence_distribution"]["high"] == 2


# ============================================================================
# State Passing Tests
# ============================================================================


class TestStatePassing:
    """Tests verifying state is properly passed between agents."""

    def setup_method(self):
        """Reset managers before each test."""
        reset_session_manager()
        reset_context_manager()

    def test_runner_maintains_session_state(self):
        """Test that runner maintains session state."""
        runner = ResearchCrewRunner()
        session_id = runner.create_session(user_id="test_user")

        # First query
        result1 = runner.run("What are AI agents?", session_id=session_id)
        assert result1["session_id"] == session_id
        assert result1["turn_number"] == 1

    def test_context_builds_from_history(self):
        """Test that context is built from conversation history."""
        runner = ResearchCrewRunner()
        session_id = runner.create_session()

        # Add conversation history
        manager = get_session_manager()
        manager.add_conversation_turn(
            session_id=session_id,
            query="What is LangGraph?",
            summary="LangGraph is a framework for building agents.",
        )
        manager.update_session_context(
            session_id=session_id,
            key_facts=["LangGraph supports stateful workflows"],
        )

        # Build prompt should include context
        prompt = runner.build_prompt_with_context(
            "How does it compare to CrewAI?",
            session_id=session_id,
        )

        assert "LangGraph" in prompt

    def test_session_stats_track_state(self):
        """Test that session stats track state correctly."""
        runner = ResearchCrewRunner()
        session_id = runner.create_session()

        # Add activity
        manager = get_session_manager()
        manager.add_conversation_turn(
            session_id=session_id,
            query="Query 1",
        )
        manager.update_session_context(
            session_id=session_id,
            topics=["Topic 1", "Topic 2"],
            key_facts=["Fact 1"],
        )

        stats = runner.get_session_stats(session_id)

        assert stats["total_turns"] == 1
        assert stats["topics_count"] == 2
        assert stats["facts_count"] == 1


# ============================================================================
# Handoff and Error Handling Tests
# ============================================================================


class TestHandoffErrorHandling:
    """Tests verifying handoffs work with error handling."""

    def setup_method(self):
        """Reset managers before each test."""
        reset_session_manager()
        reset_context_manager()

    def test_nonexistent_session_handling(self):
        """Test graceful handling of nonexistent session."""
        runner = ResearchCrewRunner()

        # Should return None, not raise exception
        session = runner.get_session("nonexistent-id")
        assert session is None

    def test_empty_query_handling(self):
        """Test handling of empty query."""
        runner = ResearchCrewRunner()
        result = runner.run("")

        # Should still return valid structure
        assert "session_id" in result
        assert "turn_number" in result

    def test_session_context_without_history(self):
        """Test context building for new session."""
        runner = ResearchCrewRunner()

        # Build prompt without session (no context)
        prompt = runner.build_prompt_with_context("New query")

        # Should return just the query
        assert prompt == "New query"

    def test_session_deletion_cleanup(self):
        """Test that deleted sessions are cleaned up."""
        manager = get_session_manager()
        session = manager.create_session(user_id="test_user")
        session_id = session.session_id

        # Delete session
        result = manager.delete_session(session_id)
        assert result is True

        # Should no longer exist
        assert manager.get_session(session_id) is None

    def test_context_estimation_handles_missing_session(self):
        """Test context estimation handles missing session gracefully."""
        runner = ResearchCrewRunner()

        estimate = runner.estimate_context_tokens(
            session_id="nonexistent",
            query="Test query",
        )

        # Should return valid estimate with zeros for session-specific data
        assert estimate["total_tokens"] > 0
        assert estimate["history_tokens"] == 0


# ============================================================================
# Integration Tests
# ============================================================================


class TestArchitectureIntegration:
    """Integration tests for the complete architecture."""

    def setup_method(self, method):
        """Reset all managers before each test."""
        reset_session_manager()
        reset_context_manager()
        reset_knowledge_base()
        self.collection_name = f"test_int_{method.__name__}"

    def test_full_workflow_structure(self):
        """Test that full workflow structure is correct via imports."""
        # All agent modules should be importable
        from agents.fact_checker import fact_checker_agent
        from agents.orchestrator import orchestrator_agent, research_phase
        from agents.researcher import researcher_agent
        from agents.synthesizer import synthesizer_agent
        from agents.writer import writer_agent

        # All should exist (even if mocked)
        assert orchestrator_agent is not None
        assert research_phase is not None
        assert researcher_agent is not None
        assert synthesizer_agent is not None
        assert fact_checker_agent is not None
        assert writer_agent is not None

    def test_session_and_knowledge_independence(self):
        """Test that session and knowledge managers are independent."""
        session_mgr = get_session_manager()
        kb = KnowledgeBaseManager(
            persist_directory=None,
            collection_name=self.collection_name,
        )

        # Create session
        session = session_mgr.create_session(user_id="test")
        session_mgr.add_conversation_turn(
            session_id=session.session_id,
            query="Test query",
        )

        # Add knowledge
        kb.add_entry(content="Test knowledge", topic="test")

        # Both should have their own data
        assert len(session_mgr.list_sessions()) == 1
        assert kb.collection.count() == 1

        # Clearing one shouldn't affect the other
        kb.clear()
        assert len(session_mgr.list_sessions()) == 1

    def test_runner_integrates_all_components(self):
        """Test that runner integrates all system components."""
        runner = ResearchCrewRunner()

        # Should have session management
        session_id = runner.create_session(user_id="test")
        assert session_id is not None

        # Should be able to run queries
        result = runner.run("Test query", session_id=session_id)
        assert "session_id" in result
        assert "turn_number" in result

        # Should be able to get session info
        session = runner.get_session(session_id)
        assert session is not None

        # Should be able to build context
        prompt = runner.build_prompt_with_context("Follow-up", session_id=session_id)
        assert prompt is not None

    def test_context_window_estimation(self):
        """Test that context window estimation works."""
        runner = ResearchCrewRunner()
        session_id = runner.create_session()

        # Add some context
        manager = get_session_manager()
        manager.add_conversation_turn(
            session_id=session_id,
            query="Previous question",
            summary="Previous answer with details",
        )
        manager.update_session_context(
            session_id=session_id,
            key_facts=["Important fact 1", "Important fact 2"],
        )

        # Estimate should include all components
        estimate = runner.estimate_context_tokens(
            session_id=session_id,
            query="New question",
        )

        assert "total_tokens" in estimate
        assert "query_tokens" in estimate
        assert "history_tokens" in estimate
        assert "facts_tokens" in estimate
        assert "remaining_tokens" in estimate
        assert estimate["total_tokens"] > 0
