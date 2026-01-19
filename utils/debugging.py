"""Debugging Utilities for ResearchCrew

Provides tools for diagnosing and debugging agent failures,
including intentional failure injection for testing observability.

Usage:
    from utils.debugging import (
        FailureInjector,
        DebugContext,
        diagnose_failure,
        get_debug_report,
    )

    # Inject a failure for testing
    with FailureInjector.tool_failure("web_search", error_rate=0.5):
        result = web_search("test query")

    # Get debug context for a trace
    debug = DebugContext.from_trace_id("abc123")
    print(debug.summary())
"""

import logging
import time
import random
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Generator, Optional, TypeVar

from utils.tracing import get_trace_id, get_span_id, trace_span
from utils.metrics import (
    record_error,
    record_tool_call,
    get_metrics_text,
)
from utils.logging_config import get_logger

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class FailureType(Enum):
    """Types of failures that can be injected for testing."""

    TOOL_ERROR = "tool_error"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    INVALID_RESPONSE = "invalid_response"
    TOKEN_LIMIT = "token_limit"
    STATE_CORRUPTION = "state_corruption"
    HALLUCINATION = "hallucination"


@dataclass
class FailureEvent:
    """Record of a failure event for debugging."""

    timestamp: datetime
    failure_type: FailureType
    component: str
    error_message: str
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/storage."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "failure_type": self.failure_type.value,
            "component": self.component,
            "error_message": self.error_message,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "context": self.context,
        }


class FailureInjector:
    """Utility for injecting failures into the system for testing observability.

    This allows testing error handling, tracing, and metrics collection
    by simulating various failure scenarios.
    """

    _active_injections: dict[str, dict[str, Any]] = {}
    _failure_history: list[FailureEvent] = []

    @classmethod
    def reset(cls) -> None:
        """Reset all active injections and history."""
        cls._active_injections.clear()
        cls._failure_history.clear()

    @classmethod
    @contextmanager
    def tool_failure(
        cls,
        tool_name: str,
        error_rate: float = 1.0,
        error_type: str = "ToolExecutionError",
        error_message: str = "Simulated tool failure",
    ) -> Generator[None, None, None]:
        """Inject failures into a specific tool.

        Args:
            tool_name: Name of the tool to inject failures into.
            error_rate: Probability of failure (0.0 to 1.0).
            error_type: Type of error to raise.
            error_message: Error message.

        Example:
            with FailureInjector.tool_failure("web_search", error_rate=0.5):
                # 50% of web_search calls will fail
                result = web_search("test")
        """
        injection_id = f"tool_{tool_name}"
        cls._active_injections[injection_id] = {
            "type": FailureType.TOOL_ERROR,
            "error_rate": error_rate,
            "error_type": error_type,
            "error_message": error_message,
        }

        try:
            yield
        finally:
            cls._active_injections.pop(injection_id, None)

    @classmethod
    @contextmanager
    def timeout_failure(
        cls,
        component: str,
        delay_seconds: float = 30.0,
        probability: float = 1.0,
    ) -> Generator[None, None, None]:
        """Inject timeout failures.

        Args:
            component: Component to inject timeouts into.
            delay_seconds: How long to delay (simulating slow response).
            probability: Probability of timeout occurring.
        """
        injection_id = f"timeout_{component}"
        cls._active_injections[injection_id] = {
            "type": FailureType.TIMEOUT,
            "delay_seconds": delay_seconds,
            "probability": probability,
        }

        try:
            yield
        finally:
            cls._active_injections.pop(injection_id, None)

    @classmethod
    @contextmanager
    def token_limit_failure(
        cls,
        max_tokens: int = 100,
        probability: float = 1.0,
    ) -> Generator[None, None, None]:
        """Inject token limit exceeded failures.

        Args:
            max_tokens: Simulated maximum token limit.
            probability: Probability of limit being hit.
        """
        injection_id = "token_limit"
        cls._active_injections[injection_id] = {
            "type": FailureType.TOKEN_LIMIT,
            "max_tokens": max_tokens,
            "probability": probability,
        }

        try:
            yield
        finally:
            cls._active_injections.pop(injection_id, None)

    @classmethod
    def should_fail(cls, injection_id: str) -> bool:
        """Check if a failure should be triggered.

        Args:
            injection_id: ID of the injection to check.

        Returns:
            True if the failure should occur.
        """
        injection = cls._active_injections.get(injection_id)
        if not injection:
            return False

        probability = injection.get("probability", injection.get("error_rate", 1.0))
        return random.random() < probability

    @classmethod
    def get_injection(cls, injection_id: str) -> Optional[dict[str, Any]]:
        """Get injection configuration if active."""
        return cls._active_injections.get(injection_id)

    @classmethod
    def record_failure(
        cls,
        failure_type: FailureType,
        component: str,
        error_message: str,
        context: Optional[dict[str, Any]] = None,
    ) -> FailureEvent:
        """Record a failure event.

        Args:
            failure_type: Type of failure.
            component: Component that failed.
            error_message: Error message.
            context: Additional context.

        Returns:
            The recorded FailureEvent.
        """
        event = FailureEvent(
            timestamp=datetime.now(timezone.utc),
            failure_type=failure_type,
            component=component,
            error_message=error_message,
            trace_id=get_trace_id(),
            span_id=get_span_id(),
            context=context or {},
        )

        cls._failure_history.append(event)

        # Also record in metrics
        record_error(
            agent=component,
            error_type=failure_type.value,
            error_message=error_message,
        )

        logger.error(
            f"Failure recorded: {failure_type.value}",
            extra={
                "component": component,
                "error_message": error_message,
                "trace_id": event.trace_id,
                "failure_type": failure_type.value,
            },
        )

        return event

    @classmethod
    def get_failure_history(
        cls,
        component: Optional[str] = None,
        failure_type: Optional[FailureType] = None,
        since: Optional[datetime] = None,
    ) -> list[FailureEvent]:
        """Get failure history with optional filters.

        Args:
            component: Filter by component name.
            failure_type: Filter by failure type.
            since: Only include failures after this time.

        Returns:
            List of matching failure events.
        """
        history = cls._failure_history

        if component:
            history = [e for e in history if e.component == component]

        if failure_type:
            history = [e for e in history if e.failure_type == failure_type]

        if since:
            history = [e for e in history if e.timestamp >= since]

        return history


@dataclass
class DebugContext:
    """Context for debugging a specific request or trace."""

    trace_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    spans: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    logs: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_trace_id(cls, trace_id: str) -> "DebugContext":
        """Create a debug context from a trace ID.

        Note: In a real implementation, this would query the trace backend.
        """
        return cls(
            trace_id=trace_id,
            start_time=datetime.now(timezone.utc),
        )

    def add_span(self, span_data: dict[str, Any]) -> None:
        """Add span data to the context."""
        self.spans.append(span_data)

    def add_error(self, error_data: dict[str, Any]) -> None:
        """Add error data to the context."""
        self.errors.append(error_data)

    def add_log(self, log_data: dict[str, Any]) -> None:
        """Add log entry to the context."""
        self.logs.append(log_data)

    def summary(self) -> str:
        """Generate a human-readable debug summary."""
        lines = [
            f"=== Debug Context for Trace: {self.trace_id} ===",
            f"Start Time: {self.start_time.isoformat()}",
            f"End Time: {self.end_time.isoformat() if self.end_time else 'In Progress'}",
            f"",
            f"Spans: {len(self.spans)}",
            f"Errors: {len(self.errors)}",
            f"Log Entries: {len(self.logs)}",
        ]

        if self.errors:
            lines.append("")
            lines.append("=== Errors ===")
            for i, error in enumerate(self.errors, 1):
                lines.append(f"{i}. {error.get('type', 'Unknown')}: {error.get('message', 'No message')}")

        return "\n".join(lines)


def diagnose_failure(
    trace_id: Optional[str] = None,
    component: Optional[str] = None,
    time_range_minutes: int = 5,
) -> dict[str, Any]:
    """Diagnose a failure using available observability data.

    Args:
        trace_id: Specific trace to analyze.
        component: Component to focus on.
        time_range_minutes: Time range to search.

    Returns:
        Diagnostic report with findings.
    """
    report: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trace_id": trace_id,
        "component": component,
        "findings": [],
        "recommendations": [],
    }

    # Check failure history
    since = datetime.now(timezone.utc)
    since = since.replace(
        minute=since.minute - time_range_minutes if since.minute >= time_range_minutes else 0
    )

    failures = FailureInjector.get_failure_history(
        component=component,
        since=since,
    )

    if failures:
        report["findings"].append({
            "type": "recent_failures",
            "count": len(failures),
            "failures": [f.to_dict() for f in failures[-5:]],  # Last 5
        })

        # Analyze failure patterns
        failure_types = {}
        for f in failures:
            ft = f.failure_type.value
            failure_types[ft] = failure_types.get(ft, 0) + 1

        most_common = max(failure_types.items(), key=lambda x: x[1])
        report["findings"].append({
            "type": "failure_pattern",
            "most_common_failure": most_common[0],
            "occurrence_count": most_common[1],
        })

        # Add recommendations based on failure type
        if most_common[0] == FailureType.TOOL_ERROR.value:
            report["recommendations"].append(
                "Check tool implementation and external service availability"
            )
        elif most_common[0] == FailureType.TIMEOUT.value:
            report["recommendations"].append(
                "Consider increasing timeout values or adding retry logic"
            )
        elif most_common[0] == FailureType.TOKEN_LIMIT.value:
            report["recommendations"].append(
                "Implement context compression or reduce prompt size"
            )
        elif most_common[0] == FailureType.RATE_LIMIT.value:
            report["recommendations"].append(
                "Add rate limiting with exponential backoff"
            )

    if not failures:
        report["findings"].append({
            "type": "no_recent_failures",
            "message": f"No failures found in the last {time_range_minutes} minutes",
        })

    return report


def get_debug_report(
    include_metrics: bool = True,
    include_failure_history: bool = True,
    format_type: str = "text",
) -> str:
    """Generate a comprehensive debug report.

    Args:
        include_metrics: Include Prometheus metrics.
        include_failure_history: Include failure history.
        format_type: Output format ("text" or "json").

    Returns:
        Formatted debug report.
    """
    lines = [
        "=" * 60,
        "RESEARCHCREW DEBUG REPORT",
        f"Generated: {datetime.now(timezone.utc).isoformat()}Z",
        "=" * 60,
        "",
    ]

    if include_failure_history:
        lines.append("=== Recent Failure History ===")
        failures = FailureInjector.get_failure_history()
        if failures:
            for f in failures[-10:]:  # Last 10
                lines.append(
                    f"  [{f.timestamp.strftime('%H:%M:%S')}] "
                    f"{f.failure_type.value}: {f.component} - {f.error_message}"
                )
        else:
            lines.append("  No failures recorded")
        lines.append("")

    if include_metrics:
        lines.append("=== Prometheus Metrics ===")
        metrics_text = get_metrics_text()
        # Show summary of key metrics
        for line in metrics_text.split("\n"):
            if line.startswith("agent_error_total") or \
               line.startswith("tool_calls_total") or \
               line.startswith("# HELP"):
                if not line.startswith("# HELP"):
                    lines.append(f"  {line}")
        lines.append("")

    lines.append("=== Active Failure Injections ===")
    if FailureInjector._active_injections:
        for name, config in FailureInjector._active_injections.items():
            lines.append(f"  {name}: {config.get('type', 'unknown').value}")
    else:
        lines.append("  No active injections")
    lines.append("")

    lines.append("=" * 60)

    return "\n".join(lines)


def inject_failure_decorator(
    failure_type: FailureType,
    error_rate: float = 0.1,
    error_message: str = "Injected failure",
) -> Callable[[F], F]:
    """Decorator to inject failures into functions for testing.

    Args:
        failure_type: Type of failure to inject.
        error_rate: Probability of failure (0.0 to 1.0).
        error_message: Error message when failure occurs.

    Returns:
        Decorated function.

    Example:
        @inject_failure_decorator(FailureType.TOOL_ERROR, error_rate=0.1)
        def my_tool(query: str) -> str:
            ...
    """
    import functools

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if random.random() < error_rate:
                FailureInjector.record_failure(
                    failure_type=failure_type,
                    component=func.__name__,
                    error_message=error_message,
                    context={"args": str(args)[:100], "kwargs": str(kwargs)[:100]},
                )
                raise RuntimeError(f"[INJECTED] {error_message}")

            return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


@contextmanager
def debug_span(
    name: str,
    record_errors: bool = True,
) -> Generator[dict[str, Any], None, None]:
    """Context manager for debugging a code section.

    Combines tracing, logging, and error recording.

    Args:
        name: Name of the debug span.
        record_errors: Whether to record errors in failure history.

    Yields:
        Context dictionary for adding debug info.

    Example:
        with debug_span("process_query") as ctx:
            ctx["query"] = query
            result = process(query)
            ctx["result_size"] = len(result)
    """
    ctx: dict[str, Any] = {
        "name": name,
        "start_time": time.time(),
    }

    with trace_span(f"debug.{name}") as span:
        ctx["trace_id"] = get_trace_id()
        ctx["span_id"] = get_span_id()

        try:
            yield ctx
            ctx["status"] = "success"
        except Exception as e:
            ctx["status"] = "error"
            ctx["error"] = str(e)
            ctx["error_type"] = type(e).__name__

            if record_errors:
                FailureInjector.record_failure(
                    failure_type=FailureType.TOOL_ERROR,
                    component=name,
                    error_message=str(e),
                    context=ctx,
                )

            span.set_attribute("error", True)
            span.set_attribute("error.message", str(e))
            raise
        finally:
            ctx["duration"] = time.time() - ctx["start_time"]

            logger.debug(
                f"Debug span completed: {name}",
                extra={
                    "span_name": name,
                    "duration": ctx["duration"],
                    "status": ctx.get("status", "unknown"),
                    "trace_id": ctx.get("trace_id"),
                },
            )
