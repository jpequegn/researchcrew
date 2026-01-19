"""Tests for OpenTelemetry Tracing Module

Tests the tracing utilities for ResearchCrew.
"""

from unittest.mock import Mock, patch

import pytest

from utils.tracing import (
    add_trace_context,
    get_span_id,
    get_trace_id,
    get_tracer,
    init_tracing,
    reset_tracing,
    trace_agent,
    trace_llm_call,
    trace_span,
    trace_tool,
)


class TestTracingInitialization:
    """Tests for tracing initialization."""

    def setup_method(self):
        """Reset tracing before each test."""
        reset_tracing()

    def teardown_method(self):
        """Reset tracing after each test."""
        reset_tracing()

    def test_init_tracing_console(self):
        """Test initialization with console exporter."""
        provider = init_tracing(exporter_type="console")
        assert provider is not None

    def test_init_tracing_none(self):
        """Test initialization with no exporter."""
        provider = init_tracing(exporter_type="none")
        assert provider is not None

    def test_init_tracing_idempotent(self):
        """Test that multiple init calls return same provider."""
        provider1 = init_tracing(exporter_type="none")
        provider2 = init_tracing(exporter_type="none")
        assert provider1 is provider2

    def test_get_tracer(self):
        """Test getting a tracer."""
        tracer = get_tracer()
        assert tracer is not None

    def test_get_tracer_auto_init(self):
        """Test that get_tracer auto-initializes."""
        # After reset, should auto-init
        tracer = get_tracer("test_module")
        assert tracer is not None


class TestTraceSpan:
    """Tests for trace_span context manager."""

    def setup_method(self):
        """Reset tracing before each test."""
        reset_tracing()
        init_tracing(exporter_type="none")

    def teardown_method(self):
        """Reset tracing after each test."""
        reset_tracing()

    def test_trace_span_basic(self):
        """Test basic span creation."""
        with trace_span("test_operation") as span:
            assert span is not None
            span.set_attribute("test_key", "test_value")

    def test_trace_span_with_attributes(self):
        """Test span creation with initial attributes."""
        attrs = {"key1": "value1", "key2": 42}
        with trace_span("test_operation", attributes=attrs) as span:
            assert span is not None

    def test_trace_span_nested(self):
        """Test nested spans."""
        with trace_span("outer") as outer_span:
            outer_span.set_attribute("level", "outer")
            with trace_span("inner") as inner_span:
                inner_span.set_attribute("level", "inner")
                assert inner_span is not None
            assert outer_span is not None


class TestTraceToolDecorator:
    """Tests for the trace_tool decorator."""

    def setup_method(self):
        """Reset tracing before each test."""
        reset_tracing()
        init_tracing(exporter_type="none")

    def teardown_method(self):
        """Reset tracing after each test."""
        reset_tracing()

    def test_trace_tool_basic(self):
        """Test basic tool tracing."""

        @trace_tool()
        def my_tool(query: str) -> str:
            return f"Result for {query}"

        result = my_tool("test")
        assert result == "Result for test"

    def test_trace_tool_with_name(self):
        """Test tool tracing with custom name."""

        @trace_tool(tool_name="custom.tool")
        def my_tool() -> str:
            return "result"

        result = my_tool()
        assert result == "result"

    def test_trace_tool_handles_exception(self):
        """Test that tool tracing handles exceptions."""

        @trace_tool()
        def failing_tool() -> str:
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            failing_tool()

    def test_trace_tool_records_args(self):
        """Test that tool tracing records arguments."""

        @trace_tool(record_args=True)
        def my_tool(query: str, count: int = 5) -> str:
            return f"{query}: {count}"

        result = my_tool("test", count=10)
        assert result == "test: 10"


class TestTraceAgentDecorator:
    """Tests for the trace_agent decorator."""

    def setup_method(self):
        """Reset tracing before each test."""
        reset_tracing()
        init_tracing(exporter_type="none")

    def teardown_method(self):
        """Reset tracing after each test."""
        reset_tracing()

    def test_trace_agent_basic(self):
        """Test basic agent tracing."""

        @trace_agent("test_agent")
        def run_agent(query: str) -> dict:
            return {"result": query}

        result = run_agent("test query")
        assert result == {"result": "test query"}

    def test_trace_agent_handles_exception(self):
        """Test that agent tracing handles exceptions."""

        @trace_agent()
        def failing_agent() -> dict:
            raise RuntimeError("Agent failed")

        with pytest.raises(RuntimeError):
            failing_agent()


class TestTraceLLMCall:
    """Tests for trace_llm_call function."""

    def setup_method(self):
        """Reset tracing before each test."""
        reset_tracing()
        init_tracing(exporter_type="none")

    def teardown_method(self):
        """Reset tracing after each test."""
        reset_tracing()

    def test_trace_llm_call_success(self):
        """Test recording successful LLM call."""
        with trace_span("test"):
            trace_llm_call(
                model_name="gemini-2.0-flash",
                prompt="What is AI?",
                response="AI is artificial intelligence.",
                token_count=50,
            )
        # Should not raise

    def test_trace_llm_call_with_error(self):
        """Test recording failed LLM call."""
        with trace_span("test"):
            trace_llm_call(
                model_name="gemini-2.0-flash",
                prompt="What is AI?",
                error=Exception("API error"),
            )
        # Should not raise


class TestAddTraceContext:
    """Tests for add_trace_context function."""

    def setup_method(self):
        """Reset tracing before each test."""
        reset_tracing()
        init_tracing(exporter_type="none")

    def teardown_method(self):
        """Reset tracing after each test."""
        reset_tracing()

    def test_add_trace_context(self):
        """Test adding research context to span."""
        with trace_span("test"):
            add_trace_context(
                query="What are AI agents?",
                session_id="session-123",
                user_id="user-456",
                turn_number=1,
            )
        # Should not raise


class TestTraceIdentifiers:
    """Tests for trace and span ID functions."""

    def setup_method(self):
        """Reset tracing before each test."""
        reset_tracing()
        init_tracing(exporter_type="none")

    def teardown_method(self):
        """Reset tracing after each test."""
        reset_tracing()

    def test_get_trace_id_in_span(self):
        """Test getting trace ID within a span."""
        with trace_span("test"):
            trace_id = get_trace_id()
            assert trace_id is not None
            assert len(trace_id) == 32  # 128-bit hex

    def test_get_span_id_in_span(self):
        """Test getting span ID within a span."""
        with trace_span("test"):
            span_id = get_span_id()
            assert span_id is not None
            assert len(span_id) == 16  # 64-bit hex

    def test_get_trace_id_no_span(self):
        """Test getting trace ID without active span."""
        # Outside any span context
        get_trace_id()
        # May return None or invalid trace ID


class TestToolTracingIntegration:
    """Integration tests for tool tracing."""

    def setup_method(self):
        """Reset tracing before each test."""
        reset_tracing()
        init_tracing(exporter_type="none")

    def teardown_method(self):
        """Reset tracing after each test."""
        reset_tracing()

    def test_search_tool_tracing(self):
        """Test that search tools have tracing."""
        # Import after tracing init
        from tools.search import web_search

        # Mock httpx to avoid network call
        with patch("tools.search.httpx.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = "<html><body></body></html>"
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            with trace_span("test_workflow"):
                result = web_search("test query")

            # Should return result without error
            assert "test query" in result

    def test_knowledge_tool_tracing(self):
        """Test that knowledge tools have tracing."""
        from tools.knowledge import knowledge_search

        with patch("tools.knowledge.get_knowledge_base") as mock_get_kb:
            mock_kb = Mock()
            mock_kb.search.return_value = []
            mock_get_kb.return_value = mock_kb

            with trace_span("test_workflow"):
                result = knowledge_search("test query")

            assert "no relevant" in result.lower()


class TestRunnerTracingIntegration:
    """Integration tests for runner tracing."""

    def setup_method(self):
        """Reset tracing and managers before each test."""
        reset_tracing()
        init_tracing(exporter_type="none")

        from utils.context_manager import reset_context_manager
        from utils.session_manager import reset_session_manager

        reset_session_manager()
        reset_context_manager()

    def teardown_method(self):
        """Reset tracing after each test."""
        reset_tracing()

    def test_runner_tracing(self):
        """Test that runner includes trace_id in result."""
        from runner import ResearchCrewRunner

        runner = ResearchCrewRunner()
        result = runner.run("What are AI agents?")

        assert "session_id" in result
        assert "trace_id" in result
        # Trace ID should be a valid hex string
        if result["trace_id"]:
            assert len(result["trace_id"]) == 32
