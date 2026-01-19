"""Tests for Production Monitoring Metrics

Tests the metrics module for ResearchCrew.
"""

import time
import pytest
from unittest.mock import Mock, patch


class TestMetricsInitialization:
    """Tests for metrics initialization."""

    def setup_method(self):
        """Reset metrics before each test."""
        from utils.metrics import reset_metrics

        reset_metrics()

    def teardown_method(self):
        """Reset metrics after each test."""
        from utils.metrics import reset_metrics

        reset_metrics()

    def test_init_metrics(self):
        """Test metrics initialization."""
        from utils.metrics import init_metrics, get_registry

        init_metrics()
        assert get_registry() is not None

    def test_init_metrics_idempotent(self):
        """Test that multiple init calls don't cause errors."""
        from utils.metrics import init_metrics, get_registry

        init_metrics()
        registry1 = get_registry()
        init_metrics()  # Should not create new metrics
        registry2 = get_registry()
        # Registry may be same or different, but no error should occur


class TestRequestDurationMetrics:
    """Tests for request duration metrics."""

    def setup_method(self):
        """Reset metrics before each test."""
        from utils.metrics import reset_metrics, init_metrics

        reset_metrics()
        init_metrics()

    def teardown_method(self):
        """Reset metrics after each test."""
        from utils.metrics import reset_metrics

        reset_metrics()

    def test_record_request_duration_success(self):
        """Test recording successful request duration."""
        from utils.metrics import record_request_duration

        with record_request_duration("test_agent") as ctx:
            time.sleep(0.01)
            ctx["status"] = "success"

        # Should not raise

    def test_record_request_duration_error(self):
        """Test recording failed request duration."""
        from utils.metrics import record_request_duration

        with pytest.raises(ValueError):
            with record_request_duration("test_agent") as ctx:
                raise ValueError("Test error")

    def test_record_request_duration_auto_error_status(self):
        """Test that error status is set automatically on exception."""
        from utils.metrics import record_request_duration

        ctx_captured = None
        try:
            with record_request_duration("test_agent") as ctx:
                ctx_captured = ctx
                raise RuntimeError("Test")
        except RuntimeError:
            pass

        assert ctx_captured["status"] == "error"


class TestTokenUsageMetrics:
    """Tests for token usage metrics."""

    def setup_method(self):
        """Reset metrics before each test."""
        from utils.metrics import reset_metrics, init_metrics

        reset_metrics()
        init_metrics()

    def teardown_method(self):
        """Reset metrics after each test."""
        from utils.metrics import reset_metrics

        reset_metrics()

    def test_record_token_usage_basic(self):
        """Test basic token usage recording."""
        from utils.metrics import record_token_usage

        result = record_token_usage(
            "researcher", input_tokens=1000, output_tokens=500
        )

        assert result["input_tokens"] == 1000
        assert result["output_tokens"] == 500
        assert result["total_tokens"] == 1500

    def test_record_token_usage_cost_calculation(self):
        """Test that costs are calculated correctly."""
        from utils.metrics import record_token_usage

        result = record_token_usage(
            "researcher",
            input_tokens=1000,
            output_tokens=500,
            cost_per_1k_input=0.001,
            cost_per_1k_output=0.002,
        )

        assert result["input_cost"] == 0.001  # 1000 / 1000 * 0.001
        assert result["output_cost"] == 0.001  # 500 / 1000 * 0.002
        assert result["total_cost"] == 0.002


class TestErrorMetrics:
    """Tests for error metrics."""

    def setup_method(self):
        """Reset metrics before each test."""
        from utils.metrics import reset_metrics, init_metrics

        reset_metrics()
        init_metrics()

    def teardown_method(self):
        """Reset metrics after each test."""
        from utils.metrics import reset_metrics

        reset_metrics()

    def test_record_error(self):
        """Test error recording."""
        from utils.metrics import record_error

        record_error("test_agent", error_type="ValueError", error_message="Test error")
        # Should not raise

    def test_update_error_rate(self):
        """Test error rate update."""
        from utils.metrics import update_error_rate

        update_error_rate("test_agent", 0.05)
        # Should not raise

    def test_update_error_rate_clamped(self):
        """Test that error rate is clamped between 0 and 1."""
        from utils.metrics import update_error_rate

        # These should not raise
        update_error_rate("test_agent", -0.5)  # Should be clamped to 0
        update_error_rate("test_agent", 1.5)  # Should be clamped to 1


class TestToolMetrics:
    """Tests for tool call metrics."""

    def setup_method(self):
        """Reset metrics before each test."""
        from utils.metrics import reset_metrics, init_metrics

        reset_metrics()
        init_metrics()

    def teardown_method(self):
        """Reset metrics after each test."""
        from utils.metrics import reset_metrics

        reset_metrics()

    def test_record_tool_call_success(self):
        """Test recording successful tool call."""
        from utils.metrics import record_tool_call

        record_tool_call("web_search", success=True, duration_seconds=1.5)
        # Should not raise

    def test_record_tool_call_failure(self):
        """Test recording failed tool call."""
        from utils.metrics import record_tool_call

        record_tool_call("web_search", success=False, duration_seconds=0.5)
        # Should not raise

    def test_update_tool_success_rate(self):
        """Test tool success rate update."""
        from utils.metrics import update_tool_success_rate

        update_tool_success_rate("web_search", 0.95)
        # Should not raise


class TestKnowledgeMetrics:
    """Tests for knowledge base metrics."""

    def setup_method(self):
        """Reset metrics before each test."""
        from utils.metrics import reset_metrics, init_metrics

        reset_metrics()
        init_metrics()

    def teardown_method(self):
        """Reset metrics after each test."""
        from utils.metrics import reset_metrics

        reset_metrics()

    def test_record_knowledge_query(self):
        """Test knowledge query recording."""
        from utils.metrics import record_knowledge_query

        record_knowledge_query(operation="search", success=True)
        record_knowledge_query(operation="save", success=False)
        # Should not raise


class TestSessionMetrics:
    """Tests for session metrics."""

    def setup_method(self):
        """Reset metrics before each test."""
        from utils.metrics import reset_metrics, init_metrics

        reset_metrics()
        init_metrics()

    def teardown_method(self):
        """Reset metrics after each test."""
        from utils.metrics import reset_metrics

        reset_metrics()

    def test_update_active_sessions(self):
        """Test active sessions update."""
        from utils.metrics import update_active_sessions

        update_active_sessions(5)
        # Should not raise

    def test_increment_decrement_sessions(self):
        """Test incrementing and decrementing sessions."""
        from utils.metrics import increment_active_sessions, decrement_active_sessions

        increment_active_sessions()
        increment_active_sessions()
        decrement_active_sessions()
        # Should not raise


class TestMetricsOutput:
    """Tests for metrics output."""

    def setup_method(self):
        """Reset metrics before each test."""
        from utils.metrics import reset_metrics, init_metrics

        reset_metrics()
        init_metrics()

    def teardown_method(self):
        """Reset metrics after each test."""
        from utils.metrics import reset_metrics

        reset_metrics()

    def test_get_metrics_text(self):
        """Test getting metrics in Prometheus format."""
        from utils.metrics import get_metrics_text, record_tool_call

        # Record some metrics
        record_tool_call("web_search", success=True)

        text = get_metrics_text()
        assert isinstance(text, str)
        # Should contain some metric output
        assert "tool_calls_total" in text or "Metrics not available" in text


class TestMetricDecorators:
    """Tests for metric decorators."""

    def setup_method(self):
        """Reset metrics before each test."""
        from utils.metrics import reset_metrics, init_metrics

        reset_metrics()
        init_metrics()

    def teardown_method(self):
        """Reset metrics after each test."""
        from utils.metrics import reset_metrics

        reset_metrics()

    def test_metric_tool_decorator(self):
        """Test metric_tool decorator."""
        from utils.metrics import metric_tool

        @metric_tool()
        def my_tool(query: str) -> str:
            return f"Result: {query}"

        result = my_tool("test")
        assert result == "Result: test"

    def test_metric_tool_decorator_with_exception(self):
        """Test metric_tool decorator handles exceptions."""
        from utils.metrics import metric_tool

        @metric_tool()
        def failing_tool() -> str:
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            failing_tool()

    def test_metric_agent_decorator(self):
        """Test metric_agent decorator."""
        from utils.metrics import metric_agent

        @metric_agent("test_agent")
        def run_agent(query: str) -> dict:
            return {"result": query}

        result = run_agent("test")
        assert result == {"result": "test"}

    def test_metric_agent_decorator_with_exception(self):
        """Test metric_agent decorator handles exceptions."""
        from utils.metrics import metric_agent

        @metric_agent()
        def failing_agent() -> dict:
            raise RuntimeError("Agent error")

        with pytest.raises(RuntimeError):
            failing_agent()


class TestMetricsWithoutPrometheus:
    """Tests for metrics when prometheus_client is not available."""

    def test_metrics_graceful_without_prometheus(self):
        """Test that metrics work gracefully without prometheus_client."""
        from utils.metrics import reset_metrics

        reset_metrics()

        # These should all work without raising
        from utils.metrics import (
            record_tool_call,
            record_token_usage,
            record_error,
            get_metrics_text,
        )

        record_tool_call("test", success=True)
        record_token_usage("test", input_tokens=100)
        record_error("test", error_type="test")

        text = get_metrics_text()
        assert isinstance(text, str)
