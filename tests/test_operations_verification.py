"""Operations Verification Tests

Comprehensive tests validating the ResearchCrew operational capabilities.
This file validates Issue #18 requirements:
- Tracing enabled and visualizable
- Monitoring/metrics infrastructure functional
- Circuit breakers implemented
- Retry logic with exponential backoff
"""

import time

import pytest

from utils.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitOpenError,
    CircuitState,
    circuit_breaker_call,
    clear_circuit_breakers,
    get_circuit_breaker,
)
from utils.metrics import (
    get_metrics_text,
    init_metrics,
    record_error,
    record_request_duration,
    record_token_usage,
    record_tool_call,
    reset_metrics,
)
from utils.resilience import (
    PermanentError,
    RateLimitError,
    RetryConfig,
    RetryPolicy,
    TransientError,
    classify_error,
    get_retry_wait,
    retry_with_backoff,
)
from utils.tracing import (
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

# ============================================================================
# Tracing Verification Tests
# ============================================================================


class TestTracingOperational:
    """Tests verifying tracing is operational."""

    def setup_method(self):
        """Reset tracing before each test."""
        reset_tracing()

    def teardown_method(self):
        """Reset tracing after each test."""
        reset_tracing()

    def test_tracing_can_be_initialized(self):
        """Verify tracing can be initialized."""
        provider = init_tracing(exporter_type="none")
        assert provider is not None

    def test_tracer_can_be_obtained(self):
        """Verify tracer can be obtained."""
        init_tracing(exporter_type="none")
        tracer = get_tracer()
        assert tracer is not None

    def test_spans_can_be_created(self):
        """Verify spans can be created."""
        init_tracing(exporter_type="none")

        with trace_span("test_operation") as span:
            assert span is not None
            # Do some work
            time.sleep(0.001)

    def test_spans_have_context(self):
        """Verify spans include relevant context."""
        init_tracing(exporter_type="none")

        attributes = {
            "agent.name": "test_agent",
            "tool.name": "test_tool",
            "operation.type": "research",
        }

        with trace_span("test_with_context", attributes=attributes) as span:
            # Span should be created with attributes
            assert span is not None

    def test_nested_spans_work(self):
        """Verify nested spans create proper hierarchy."""
        init_tracing(exporter_type="none")

        with trace_span("parent_operation") as parent:
            with trace_span("child_operation") as child:
                with trace_span("grandchild_operation") as grandchild:
                    # All spans should be created
                    assert parent is not None
                    assert child is not None
                    assert grandchild is not None

    def test_trace_id_can_be_retrieved(self):
        """Verify trace ID can be retrieved for correlation."""
        init_tracing(exporter_type="none")

        with trace_span("test_operation"):
            trace_id = get_trace_id()
            # Should have a trace ID
            assert trace_id is not None or trace_id == "no-trace"

    def test_span_id_can_be_retrieved(self):
        """Verify span ID can be retrieved."""
        init_tracing(exporter_type="none")

        with trace_span("test_operation"):
            span_id = get_span_id()
            assert span_id is not None or span_id == "no-span"


class TestTracingDecorators:
    """Tests for tracing decorators."""

    def setup_method(self):
        """Reset tracing before each test."""
        reset_tracing()
        init_tracing(exporter_type="none")

    def teardown_method(self):
        """Reset tracing after each test."""
        reset_tracing()

    def test_trace_tool_decorator(self):
        """Verify trace_tool decorator works."""

        @trace_tool("test_tool")
        def my_tool(query: str) -> str:
            return f"Result for: {query}"

        result = my_tool("test query")
        assert result == "Result for: test query"

    def test_trace_agent_decorator(self):
        """Verify trace_agent decorator works."""

        @trace_agent("test_agent")
        def my_agent(input_data: str) -> str:
            return f"Agent processed: {input_data}"

        result = my_agent("test input")
        assert result == "Agent processed: test input"

    def test_trace_llm_call_function(self):
        """Verify trace_llm_call function creates span correctly."""
        # trace_llm_call is a context manager for tracing LLM calls
        with trace_span("test_operation"):
            # Call trace_llm_call as a function
            trace_llm_call(model_name="gemini-2.0-flash", prompt="test prompt", response="test response")
            # Should not raise
            assert True


# ============================================================================
# Metrics Verification Tests
# ============================================================================


class TestMetricsOperational:
    """Tests verifying metrics are operational."""

    def setup_method(self):
        """Reset metrics before each test."""
        reset_metrics()
        init_metrics()

    def test_metrics_can_be_initialized(self):
        """Verify metrics can be initialized."""
        # Already initialized in setup
        # Just verify no error
        init_metrics()

    def test_request_duration_can_be_recorded(self):
        """Verify request duration can be recorded."""
        # record_request_duration is a context manager
        with record_request_duration(agent="researcher"):
            time.sleep(0.01)  # Simulate some work

        # Should not raise
        assert True

    def test_errors_can_be_recorded(self):
        """Verify errors can be recorded."""
        record_error(
            agent="test_agent",
            error_type="connection_error",
            error_message="Connection refused",
        )

        # Should not raise
        assert True

    def test_tokens_can_be_recorded(self):
        """Verify token usage can be recorded."""
        result = record_token_usage(
            agent="researcher",
            input_tokens=500,
            output_tokens=200,
        )

        # Should return token info with cost
        assert result is not None
        assert "input_tokens" in result
        assert "output_tokens" in result
        assert "total_cost" in result

    def test_token_usage_calculates_cost(self):
        """Verify token usage includes cost calculation."""
        result = record_token_usage(
            agent="researcher",
            input_tokens=1000,
            output_tokens=500,
        )

        # Should have calculated cost
        assert result["total_cost"] >= 0
        assert result["input_cost"] >= 0
        assert result["output_cost"] >= 0

    def test_tool_calls_can_be_recorded(self):
        """Verify tool calls can be recorded."""
        record_tool_call(
            tool="web_search",
            success=True,
            duration_seconds=0.25,
        )
        record_tool_call(
            tool="web_search",
            success=False,
            duration_seconds=0.5,
        )

        # Should not raise
        assert True

    def test_metrics_text_available(self):
        """Verify metrics can be retrieved in Prometheus format."""
        with record_request_duration(agent="test"):
            time.sleep(0.01)
        record_error(agent="test", error_type="test", error_message="test")

        metrics_text = get_metrics_text()
        assert metrics_text is not None
        assert isinstance(metrics_text, str)


# ============================================================================
# Circuit Breaker Verification Tests
# ============================================================================


class TestCircuitBreakerOperational:
    """Tests verifying circuit breakers are operational."""

    def setup_method(self):
        """Reset circuit breakers before each test."""
        clear_circuit_breakers()

    def test_circuit_breaker_can_be_created(self):
        """Verify circuit breaker can be created."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=30.0,
        )
        cb = CircuitBreaker(name="test_breaker", config=config)
        assert cb is not None
        assert cb.name == "test_breaker"

    def test_circuit_breaker_starts_closed(self):
        """Verify circuit breaker starts in CLOSED state."""
        cb = CircuitBreaker("test")
        assert cb.state == CircuitState.CLOSED

    def test_circuit_breaker_opens_after_failures(self):
        """Verify circuit opens after reaching failure threshold."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker(name="test_opens", config=config)

        def failing_func():
            raise Exception("Test failure")

        # Trigger failures by calling through the circuit breaker
        for _ in range(3):
            try:
                cb.call(failing_func)
            except Exception:
                pass

        assert cb.state == CircuitState.OPEN

    def test_open_circuit_rejects_calls(self):
        """Verify open circuit rejects calls."""
        config = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker(name="test_rejects", config=config)

        def failing_func():
            raise Exception("Test failure")

        # Open the circuit with one failure
        try:
            cb.call(failing_func)
        except Exception:
            pass

        assert cb.state == CircuitState.OPEN

        # Subsequent calls should be rejected
        def success_func():
            return "success"

        with pytest.raises(CircuitOpenError):
            cb.call(success_func)

    def test_circuit_breaker_decorator(self):
        """Verify circuit breaker can be used via decorator."""
        call_count = 0

        @circuit_breaker_call("test_decorator")
        def my_function():
            nonlocal call_count
            call_count += 1
            return "success"

        result = my_function()
        assert result == "success"
        assert call_count == 1

    def test_circuit_breaker_tracks_success(self):
        """Verify circuit breaker stays closed on success."""
        config = CircuitBreakerConfig(failure_threshold=5)
        cb = CircuitBreaker(name="test_success", config=config)

        def success_func():
            return "success"

        # Execute successful calls
        result = cb.call(success_func)
        assert result == "success"

        # Circuit should stay closed
        assert cb.state == CircuitState.CLOSED

    def test_global_circuit_breaker_registry(self):
        """Verify circuit breakers can be retrieved by name."""
        cb1 = get_circuit_breaker("web_search")
        cb2 = get_circuit_breaker("web_search")

        # Should return same instance
        assert cb1 is cb2

    def test_circuit_breaker_statistics(self):
        """Verify circuit breaker tracks statistics."""
        config = CircuitBreakerConfig(failure_threshold=10)
        cb = CircuitBreaker(name="stats_test", config=config)

        def success_func():
            return "success"

        def failing_func():
            raise Exception("test")

        # Execute some calls
        cb.call(success_func)
        try:
            cb.call(failing_func)
        except Exception:
            pass
        cb.call(success_func)

        stats = cb.get_stats()
        assert stats.total_calls >= 2
        assert stats.name == "stats_test"


# ============================================================================
# Retry Logic Verification Tests
# ============================================================================


class TestRetryLogicOperational:
    """Tests verifying retry logic is operational."""

    def test_retry_succeeds_on_first_try(self):
        """Verify retry returns immediately on success."""
        call_count = 0

        # Use default policy
        @retry_with_backoff(policy=RetryPolicy.DEFAULT)
        def success_function():
            nonlocal call_count
            call_count += 1
            return "success"

        result = success_function()
        assert result == "success"
        assert call_count == 1

    def test_retry_on_transient_error(self):
        """Verify transient errors are retried."""
        call_count = 0

        # Use a custom config with fast retries for testing
        config = RetryConfig(
            max_attempts=4,
            min_wait=0.01,
            max_wait=0.1,
        )

        @retry_with_backoff(policy=config)
        def failing_then_success():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise TransientError("Temporary failure")
            return "success"

        result = failing_then_success()
        assert result == "success"
        assert call_count == 3

    def test_permanent_errors_not_retried(self):
        """Verify permanent errors fail fast without retry."""
        call_count = 0

        @retry_with_backoff(policy=RetryPolicy.DEFAULT)
        def permanent_failure():
            nonlocal call_count
            call_count += 1
            raise PermanentError("Invalid input")

        with pytest.raises(PermanentError):
            permanent_failure()

        assert call_count == 1  # Only called once

    def test_exponential_backoff_calculation(self):
        """Verify exponential backoff is calculated correctly."""
        # Configure retry with no jitter for predictable values
        config = RetryConfig(
            min_wait=1.0,
            max_wait=60.0,
            multiplier=2.0,
            jitter=0.0,  # No jitter for predictable tests
        )

        error = TransientError("test")

        delay_1 = get_retry_wait(error, attempt=1, policy=config)
        delay_2 = get_retry_wait(error, attempt=2, policy=config)
        delay_3 = get_retry_wait(error, attempt=3, policy=config)

        # Exponential growth: 1, 2, 4
        assert delay_1 == 1.0
        assert delay_2 == 2.0
        assert delay_3 == 4.0

    def test_backoff_with_jitter(self):
        """Verify jitter adds randomness to backoff."""
        # Configure retry with jitter
        config = RetryConfig(
            min_wait=1.0,
            max_wait=60.0,
            multiplier=2.0,
            jitter=0.5,  # 50% jitter
        )

        error = TransientError("test")

        # With jitter, delays should vary
        delays = [get_retry_wait(error, attempt=1, policy=config) for _ in range(10)]

        # Not all delays should be identical (due to jitter)
        assert len(set(delays)) > 1

    def test_backoff_respects_max_delay(self):
        """Verify backoff respects maximum delay."""
        max_delay = 5.0

        config = RetryConfig(
            min_wait=1.0,
            max_wait=max_delay,
            multiplier=2.0,
            jitter=0.0,  # No jitter for predictable test
        )

        error = TransientError("test")
        delay = get_retry_wait(error, attempt=10, policy=config)  # Would be 512 without cap

        assert delay <= max_delay


class TestErrorClassificationOperational:
    """Tests verifying error classification is operational."""

    def test_timeout_classified_as_transient(self):
        """Verify timeout errors are classified as transient."""
        error = Exception("Connection timeout")
        classified = classify_error(error)
        assert isinstance(classified, TransientError)

    def test_rate_limit_classified_correctly(self):
        """Verify rate limit errors are classified correctly."""
        error = Exception("Rate limit exceeded")
        classified = classify_error(error)
        assert isinstance(classified, RateLimitError)

    def test_auth_error_classified_as_permanent(self):
        """Verify auth errors are classified as permanent."""
        error = Exception("Unauthorized: invalid API key")
        classified = classify_error(error)
        assert isinstance(classified, PermanentError)


# ============================================================================
# Integration Tests
# ============================================================================


class TestOperationsIntegration:
    """Integration tests for operational capabilities."""

    def setup_method(self):
        """Reset all systems before each test."""
        reset_tracing()
        reset_metrics()
        clear_circuit_breakers()
        init_tracing(exporter_type="none")
        init_metrics()

    def teardown_method(self):
        """Reset all systems after each test."""
        reset_tracing()
        reset_metrics()
        clear_circuit_breakers()

    def test_tracing_and_metrics_together(self):
        """Verify tracing and metrics work together."""
        with trace_span("test_operation", attributes={"agent": "test"}):
            with record_request_duration(agent="test"):
                time.sleep(0.01)
            record_tool_call(tool="test_tool", success=True, duration_seconds=0.05)

        # Both should complete without error
        assert True

    def test_circuit_breaker_with_retry(self):
        """Verify circuit breaker and retry work together."""
        call_count = 0

        # Use a fast retry config for testing
        retry_config = RetryConfig(max_attempts=3, min_wait=0.01, max_wait=0.1)

        @circuit_breaker_call("integration_test")
        @retry_with_backoff(policy=retry_config)
        def operation_with_resilience():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise TransientError("Temporary issue")
            return "success"

        result = operation_with_resilience()
        assert result == "success"
        assert call_count == 2

    def test_full_operational_flow(self):
        """Verify complete operational flow works."""
        # Use a fast retry config for testing
        retry_config = RetryConfig(max_attempts=3, min_wait=0.01, max_wait=0.1)

        @trace_tool("web_search")
        @circuit_breaker_call("full_flow_test")
        @retry_with_backoff(policy=retry_config)
        def search_with_observability(query: str) -> str:
            record_tool_call(tool="web_search", success=True, duration_seconds=0.1)
            return f"Results for: {query}"

        # Execute the operation
        with trace_span("research_workflow"):
            result = search_with_observability("AI agents")
            with record_request_duration(agent="researcher"):
                time.sleep(0.01)

        assert "AI agents" in result

    def test_operations_dont_interfere(self):
        """Verify operations don't interfere with each other."""
        # Create multiple circuit breakers with low threshold for testing
        config = CircuitBreakerConfig(failure_threshold=3)
        cb1 = get_circuit_breaker("operation_1_test", config=config)
        cb2 = get_circuit_breaker("operation_2_test", config=config)

        def failing_func():
            raise Exception("test failure")

        # Open one circuit by causing failures
        for _ in range(3):
            try:
                cb1.call(failing_func)
            except Exception:
                pass

        assert cb1.state == CircuitState.OPEN
        assert cb2.state == CircuitState.CLOSED  # Should be independent

    def test_metrics_tracked_across_operations(self):
        """Verify metrics are tracked across multiple operations."""
        for _ in range(5):
            with record_request_duration(agent="test_agent"):
                time.sleep(0.001)

        for i in range(3):
            record_tool_call(tool="test_tool", success=i < 2, duration_seconds=0.05)

        record_error(agent="test_agent", error_type="test", error_message="test error")

        # Get metrics text
        metrics_text = get_metrics_text()
        assert metrics_text is not None
        assert isinstance(metrics_text, str)
