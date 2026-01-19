"""Tests for Debugging Utilities

Tests the debugging module for ResearchCrew.
"""

import time
import pytest
from datetime import datetime, timedelta


class TestFailureInjector:
    """Tests for FailureInjector."""

    def setup_method(self):
        """Reset injector before each test."""
        from utils.debugging import FailureInjector

        FailureInjector.reset()

    def teardown_method(self):
        """Reset injector after each test."""
        from utils.debugging import FailureInjector

        FailureInjector.reset()

    def test_tool_failure_injection(self):
        """Test that tool failure injection works."""
        from utils.debugging import FailureInjector

        with FailureInjector.tool_failure("test_tool", error_rate=1.0):
            assert FailureInjector.should_fail("tool_test_tool")

        # After context, injection should be removed
        assert not FailureInjector.should_fail("tool_test_tool")

    def test_tool_failure_error_rate(self):
        """Test that error rate controls failure probability."""
        from utils.debugging import FailureInjector

        with FailureInjector.tool_failure("test_tool", error_rate=0.0):
            assert not FailureInjector.should_fail("tool_test_tool")

    def test_timeout_failure_injection(self):
        """Test timeout failure injection."""
        from utils.debugging import FailureInjector

        with FailureInjector.timeout_failure("test_component", delay_seconds=5.0):
            injection = FailureInjector.get_injection("timeout_test_component")
            assert injection is not None
            assert injection["delay_seconds"] == 5.0

    def test_token_limit_failure_injection(self):
        """Test token limit failure injection."""
        from utils.debugging import FailureInjector

        with FailureInjector.token_limit_failure(max_tokens=100):
            injection = FailureInjector.get_injection("token_limit")
            assert injection is not None
            assert injection["max_tokens"] == 100

    def test_record_failure(self):
        """Test failure event recording."""
        from utils.debugging import FailureInjector, FailureType
        from utils import reset_metrics, init_metrics

        reset_metrics()
        init_metrics()

        event = FailureInjector.record_failure(
            failure_type=FailureType.TOOL_ERROR,
            component="test_component",
            error_message="Test error message",
            context={"key": "value"},
        )

        assert event.failure_type == FailureType.TOOL_ERROR
        assert event.component == "test_component"
        assert event.error_message == "Test error message"
        assert event.context == {"key": "value"}

        # Check history
        history = FailureInjector.get_failure_history()
        assert len(history) == 1
        assert history[0] == event

    def test_failure_history_filtering(self):
        """Test filtering failure history."""
        from utils.debugging import FailureInjector, FailureType
        from utils import reset_metrics, init_metrics

        reset_metrics()
        init_metrics()

        # Record multiple failures
        FailureInjector.record_failure(
            FailureType.TOOL_ERROR, "component_a", "Error A"
        )
        FailureInjector.record_failure(
            FailureType.TIMEOUT, "component_b", "Error B"
        )
        FailureInjector.record_failure(
            FailureType.TOOL_ERROR, "component_b", "Error C"
        )

        # Filter by component
        history_a = FailureInjector.get_failure_history(component="component_a")
        assert len(history_a) == 1

        # Filter by type
        history_tool = FailureInjector.get_failure_history(failure_type=FailureType.TOOL_ERROR)
        assert len(history_tool) == 2

    def test_failure_event_to_dict(self):
        """Test FailureEvent serialization."""
        from utils.debugging import FailureEvent, FailureType

        event = FailureEvent(
            timestamp=datetime(2025, 1, 1, 12, 0, 0),
            failure_type=FailureType.TOOL_ERROR,
            component="test",
            error_message="Test",
            trace_id="abc123",
            span_id="def456",
            context={"key": "value"},
        )

        d = event.to_dict()
        assert d["timestamp"] == "2025-01-01T12:00:00"
        assert d["failure_type"] == "tool_error"
        assert d["component"] == "test"
        assert d["trace_id"] == "abc123"


class TestDebugContext:
    """Tests for DebugContext."""

    def test_from_trace_id(self):
        """Test creating debug context from trace ID."""
        from utils.debugging import DebugContext

        ctx = DebugContext.from_trace_id("test-trace-123")
        assert ctx.trace_id == "test-trace-123"
        assert ctx.start_time is not None

    def test_add_span(self):
        """Test adding span data."""
        from utils.debugging import DebugContext

        ctx = DebugContext.from_trace_id("test")
        ctx.add_span({"name": "test_span", "duration": 1.5})

        assert len(ctx.spans) == 1
        assert ctx.spans[0]["name"] == "test_span"

    def test_add_error(self):
        """Test adding error data."""
        from utils.debugging import DebugContext

        ctx = DebugContext.from_trace_id("test")
        ctx.add_error({"type": "TestError", "message": "Test"})

        assert len(ctx.errors) == 1

    def test_summary(self):
        """Test debug context summary generation."""
        from utils.debugging import DebugContext

        ctx = DebugContext.from_trace_id("test-123")
        ctx.add_span({"name": "span1"})
        ctx.add_error({"type": "TestError", "message": "Test error"})

        summary = ctx.summary()
        assert "test-123" in summary
        assert "Spans: 1" in summary
        assert "Errors: 1" in summary
        assert "TestError" in summary


class TestDiagnoseFailure:
    """Tests for diagnose_failure function."""

    def setup_method(self):
        """Reset state before each test."""
        from utils.debugging import FailureInjector
        from utils import reset_metrics, init_metrics

        FailureInjector.reset()
        reset_metrics()
        init_metrics()

    def teardown_method(self):
        """Reset state after each test."""
        from utils.debugging import FailureInjector
        from utils import reset_metrics

        FailureInjector.reset()
        reset_metrics()

    def test_diagnose_no_failures(self):
        """Test diagnosis when no failures present."""
        from utils.debugging import diagnose_failure

        report = diagnose_failure(component="test", time_range_minutes=5)

        assert "findings" in report
        assert any(f.get("type") == "no_recent_failures" for f in report["findings"])

    def test_diagnose_with_failures(self):
        """Test diagnosis when failures are present."""
        from utils.debugging import diagnose_failure, FailureInjector, FailureType

        # Record some failures
        FailureInjector.record_failure(
            FailureType.TOOL_ERROR, "test", "Error 1"
        )
        FailureInjector.record_failure(
            FailureType.TOOL_ERROR, "test", "Error 2"
        )

        report = diagnose_failure(component="test", time_range_minutes=5)

        assert any(f.get("type") == "recent_failures" for f in report["findings"])
        assert any(f.get("type") == "failure_pattern" for f in report["findings"])

    def test_diagnose_recommendations(self):
        """Test that appropriate recommendations are generated."""
        from utils.debugging import diagnose_failure, FailureInjector, FailureType

        # Record tool errors
        FailureInjector.record_failure(
            FailureType.TOOL_ERROR, "test", "Error"
        )

        report = diagnose_failure(component="test", time_range_minutes=5)

        assert len(report["recommendations"]) > 0
        assert any("tool" in r.lower() for r in report["recommendations"])


class TestGetDebugReport:
    """Tests for get_debug_report function."""

    def setup_method(self):
        """Reset state before each test."""
        from utils.debugging import FailureInjector
        from utils import reset_metrics, init_metrics

        FailureInjector.reset()
        reset_metrics()
        init_metrics()

    def teardown_method(self):
        """Reset state after each test."""
        from utils.debugging import FailureInjector
        from utils import reset_metrics

        FailureInjector.reset()
        reset_metrics()

    def test_basic_report(self):
        """Test basic debug report generation."""
        from utils.debugging import get_debug_report

        report = get_debug_report()

        assert "DEBUG REPORT" in report
        assert "Recent Failure History" in report
        assert "Prometheus Metrics" in report
        assert "Active Failure Injections" in report

    def test_report_with_failures(self):
        """Test report includes failure history."""
        from utils.debugging import get_debug_report, FailureInjector, FailureType

        FailureInjector.record_failure(
            FailureType.TOOL_ERROR, "test", "Test error"
        )

        report = get_debug_report(include_failure_history=True)

        assert "test" in report.lower()
        assert "tool_error" in report.lower()

    def test_report_with_active_injections(self):
        """Test report shows active injections."""
        from utils.debugging import get_debug_report, FailureInjector

        with FailureInjector.tool_failure("test_tool"):
            report = get_debug_report()
            assert "tool_test_tool" in report


class TestInjectFailureDecorator:
    """Tests for inject_failure_decorator."""

    def setup_method(self):
        """Reset state before each test."""
        from utils.debugging import FailureInjector
        from utils import reset_metrics, init_metrics

        FailureInjector.reset()
        reset_metrics()
        init_metrics()

    def teardown_method(self):
        """Reset state after each test."""
        from utils.debugging import FailureInjector
        from utils import reset_metrics

        FailureInjector.reset()
        reset_metrics()

    def test_decorator_success(self):
        """Test decorated function succeeds when no failure injected."""
        from utils.debugging import inject_failure_decorator, FailureType

        @inject_failure_decorator(FailureType.TOOL_ERROR, error_rate=0.0)
        def my_function(x: int) -> int:
            return x * 2

        result = my_function(5)
        assert result == 10

    def test_decorator_failure(self):
        """Test decorated function fails when failure injected."""
        from utils.debugging import inject_failure_decorator, FailureType

        @inject_failure_decorator(FailureType.TOOL_ERROR, error_rate=1.0)
        def my_function(x: int) -> int:
            return x * 2

        with pytest.raises(RuntimeError) as exc_info:
            my_function(5)

        assert "[INJECTED]" in str(exc_info.value)


class TestDebugSpan:
    """Tests for debug_span context manager."""

    def setup_method(self):
        """Reset state before each test."""
        from utils.debugging import FailureInjector
        from utils import reset_tracing, init_tracing, reset_metrics, init_metrics

        FailureInjector.reset()
        reset_tracing()
        init_tracing(exporter_type="none")
        reset_metrics()
        init_metrics()

    def teardown_method(self):
        """Reset state after each test."""
        from utils.debugging import FailureInjector
        from utils import reset_tracing, reset_metrics

        FailureInjector.reset()
        reset_tracing()
        reset_metrics()

    def test_debug_span_success(self):
        """Test debug span with successful execution."""
        from utils.debugging import debug_span

        with debug_span("test_operation") as ctx:
            ctx["input"] = "test"
            time.sleep(0.01)

        assert ctx["status"] == "success"
        assert ctx["duration"] > 0
        assert ctx["input"] == "test"

    def test_debug_span_error(self):
        """Test debug span with error."""
        from utils.debugging import debug_span

        ctx_captured = None
        with pytest.raises(ValueError):
            with debug_span("test_operation") as ctx:
                ctx_captured = ctx
                raise ValueError("Test error")

        assert ctx_captured["status"] == "error"
        assert ctx_captured["error"] == "Test error"
        assert ctx_captured["error_type"] == "ValueError"

    def test_debug_span_records_failure(self):
        """Test that debug span records failure in history."""
        from utils.debugging import debug_span, FailureInjector

        try:
            with debug_span("test_op", record_errors=True):
                raise RuntimeError("Test")
        except RuntimeError:
            pass

        history = FailureInjector.get_failure_history(component="test_op")
        assert len(history) == 1


class TestFailureTypes:
    """Tests for FailureType enum."""

    def test_all_failure_types(self):
        """Test all failure types are defined."""
        from utils.debugging import FailureType

        expected_types = [
            "TOOL_ERROR",
            "TIMEOUT",
            "RATE_LIMIT",
            "INVALID_RESPONSE",
            "TOKEN_LIMIT",
            "STATE_CORRUPTION",
            "HALLUCINATION",
        ]

        for ft in expected_types:
            assert hasattr(FailureType, ft)

    def test_failure_type_values(self):
        """Test failure type enum values."""
        from utils.debugging import FailureType

        assert FailureType.TOOL_ERROR.value == "tool_error"
        assert FailureType.TIMEOUT.value == "timeout"
        assert FailureType.TOKEN_LIMIT.value == "token_limit"


class TestIntegrationWithObservability:
    """Integration tests with observability stack."""

    def setup_method(self):
        """Reset all observability tools."""
        from utils.debugging import FailureInjector
        from utils import (
            reset_tracing,
            init_tracing,
            reset_metrics,
            init_metrics,
        )

        FailureInjector.reset()
        reset_tracing()
        reset_metrics()
        init_tracing(exporter_type="none")
        init_metrics()

    def teardown_method(self):
        """Reset all observability tools."""
        from utils.debugging import FailureInjector
        from utils import reset_tracing, reset_metrics

        FailureInjector.reset()
        reset_tracing()
        reset_metrics()

    def test_failure_records_to_metrics(self):
        """Test that failures are recorded in metrics."""
        from utils.debugging import FailureInjector, FailureType
        from utils import get_metrics_text

        FailureInjector.record_failure(
            FailureType.TOOL_ERROR, "test_agent", "Test error"
        )

        metrics = get_metrics_text()
        assert "agent_error_total" in metrics

    def test_debug_span_creates_trace(self):
        """Test that debug_span creates trace spans."""
        from utils.debugging import debug_span
        from utils import get_trace_id

        with debug_span("test_operation") as ctx:
            trace_id = get_trace_id()
            # Trace ID should be available within span
            assert ctx.get("trace_id") is not None or trace_id is not None

    def test_full_debugging_workflow(self):
        """Test complete debugging workflow."""
        from utils.debugging import (
            debug_span,
            FailureInjector,
            FailureType,
            diagnose_failure,
            get_debug_report,
        )

        # Simulate a failure scenario
        try:
            with debug_span("failing_operation"):
                FailureInjector.record_failure(
                    FailureType.TOOL_ERROR,
                    "failing_operation",
                    "Simulated failure",
                )
                raise RuntimeError("Operation failed")
        except RuntimeError:
            pass

        # Diagnose the failure
        diagnosis = diagnose_failure(
            component="failing_operation",
            time_range_minutes=1,
        )

        assert len(diagnosis["findings"]) > 0

        # Generate debug report
        report = get_debug_report()
        assert "failing_operation" in report.lower()
