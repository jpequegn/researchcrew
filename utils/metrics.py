"""Production Monitoring Metrics for ResearchCrew

Provides Prometheus-compatible metrics for monitoring agent health,
performance, and costs.

Usage:
    from utils.metrics import (
        record_request_duration,
        record_token_usage,
        record_error,
        record_tool_call,
    )

    # Record a request
    with record_request_duration("orchestrator"):
        result = await run_agent(query)

    # Record token usage
    record_token_usage("researcher", input_tokens=1500, output_tokens=500)
"""

import logging
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any, Optional, TypeVar

try:
    from prometheus_client import (
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

from utils.tracing import get_trace_id

logger = logging.getLogger(__name__)

# Type variable for generic function decorators
F = TypeVar("F", bound=Callable[..., Any])

# Default registry
_registry: Optional["CollectorRegistry"] = None

# Metrics instances
_request_duration: Optional["Histogram"] = None
_token_usage: Optional["Counter"] = None
_error_count: Optional["Counter"] = None
_error_rate: Optional["Gauge"] = None
_cost_per_request: Optional["Histogram"] = None
_tool_calls: Optional["Counter"] = None
_tool_success_rate: Optional["Gauge"] = None
_knowledge_queries: Optional["Counter"] = None
_active_sessions: Optional["Gauge"] = None

# Cost per 1000 tokens (example rates, should be configurable)
DEFAULT_COST_PER_1K_INPUT = 0.00025  # $0.25 per 1M input tokens
DEFAULT_COST_PER_1K_OUTPUT = 0.00125  # $1.25 per 1M output tokens


def init_metrics(registry: Optional["CollectorRegistry"] = None) -> None:
    """Initialize Prometheus metrics.

    Args:
        registry: Optional custom registry. Creates a new one if not provided.
    """
    global _registry, _request_duration, _token_usage, _error_count
    global _error_rate, _cost_per_request, _tool_calls, _tool_success_rate
    global _knowledge_queries, _active_sessions

    if not PROMETHEUS_AVAILABLE:
        logger.warning("prometheus_client not installed, metrics will be no-ops")
        return

    _registry = registry or CollectorRegistry()

    # Request duration histogram
    _request_duration = Histogram(
        "agent_request_duration_seconds",
        "Time spent processing agent requests",
        labelnames=["agent", "status"],
        buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
        registry=_registry,
    )

    # Token usage counter
    _token_usage = Counter(
        "agent_token_usage_total",
        "Total tokens used by agents",
        labelnames=["agent", "token_type"],
        registry=_registry,
    )

    # Error count
    _error_count = Counter(
        "agent_error_total",
        "Total errors by agent",
        labelnames=["agent", "error_type"],
        registry=_registry,
    )

    # Error rate gauge (updated periodically)
    _error_rate = Gauge(
        "agent_error_rate",
        "Current error rate by agent (0-1)",
        labelnames=["agent"],
        registry=_registry,
    )

    # Cost per request histogram
    _cost_per_request = Histogram(
        "agent_cost_per_request_dollars",
        "Cost per request in dollars",
        labelnames=["agent"],
        buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
        registry=_registry,
    )

    # Tool calls counter
    _tool_calls = Counter(
        "tool_calls_total",
        "Total tool calls",
        labelnames=["tool", "status"],
        registry=_registry,
    )

    # Tool success rate gauge
    _tool_success_rate = Gauge(
        "tool_success_rate",
        "Current success rate per tool (0-1)",
        labelnames=["tool"],
        registry=_registry,
    )

    # Knowledge base queries counter
    _knowledge_queries = Counter(
        "knowledge_base_queries_total",
        "Total knowledge base queries",
        labelnames=["operation", "status"],
        registry=_registry,
    )

    # Active sessions gauge
    _active_sessions = Gauge(
        "active_sessions_total",
        "Number of active sessions",
        registry=_registry,
    )

    logger.info("Prometheus metrics initialized")


def get_registry() -> Optional["CollectorRegistry"]:
    """Get the metrics registry."""
    return _registry


def get_metrics_text() -> str:
    """Get metrics in Prometheus text format.

    Returns:
        Metrics in Prometheus exposition format.
    """
    if not PROMETHEUS_AVAILABLE or _registry is None:
        return "# Metrics not available\n"

    return generate_latest(_registry).decode("utf-8")


def reset_metrics() -> None:
    """Reset all metrics. Useful for testing."""
    global _registry, _request_duration, _token_usage, _error_count
    global _error_rate, _cost_per_request, _tool_calls, _tool_success_rate
    global _knowledge_queries, _active_sessions

    _registry = None
    _request_duration = None
    _token_usage = None
    _error_count = None
    _error_rate = None
    _cost_per_request = None
    _tool_calls = None
    _tool_success_rate = None
    _knowledge_queries = None
    _active_sessions = None


@contextmanager
def record_request_duration(
    agent: str,
) -> Generator[dict[str, Any], None, None]:
    """Context manager to record request duration.

    Args:
        agent: Name of the agent.

    Yields:
        A context dictionary for recording additional info.

    Example:
        with record_request_duration("orchestrator") as ctx:
            result = await run_agent(query)
            ctx["status"] = "success" if result else "failure"
    """
    ctx: dict[str, Any] = {"status": "success", "start_time": time.time()}

    try:
        yield ctx
    except Exception:
        ctx["status"] = "error"
        raise
    finally:
        duration = time.time() - ctx["start_time"]

        if _request_duration is not None:
            _request_duration.labels(agent=agent, status=ctx["status"]).observe(duration)

        logger.debug(
            "Request completed",
            extra={
                "agent": agent,
                "status": ctx["status"],
                "duration_seconds": duration,
                "trace_id": get_trace_id(),
            },
        )


def record_token_usage(
    agent: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cost_per_1k_input: float = DEFAULT_COST_PER_1K_INPUT,
    cost_per_1k_output: float = DEFAULT_COST_PER_1K_OUTPUT,
) -> dict[str, float]:
    """Record token usage and calculate cost.

    Args:
        agent: Name of the agent.
        input_tokens: Number of input tokens used.
        output_tokens: Number of output tokens generated.
        cost_per_1k_input: Cost per 1000 input tokens.
        cost_per_1k_output: Cost per 1000 output tokens.

    Returns:
        Dictionary with token counts and calculated cost.
    """
    if _token_usage is not None:
        if input_tokens > 0:
            _token_usage.labels(agent=agent, token_type="input").inc(input_tokens)
        if output_tokens > 0:
            _token_usage.labels(agent=agent, token_type="output").inc(output_tokens)

    # Calculate cost
    input_cost = (input_tokens / 1000) * cost_per_1k_input
    output_cost = (output_tokens / 1000) * cost_per_1k_output
    total_cost = input_cost + output_cost

    if _cost_per_request is not None and total_cost > 0:
        _cost_per_request.labels(agent=agent).observe(total_cost)

    result = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
    }

    logger.debug(
        "Token usage recorded",
        extra={
            "agent": agent,
            **result,
            "trace_id": get_trace_id(),
        },
    )

    return result


def record_error(
    agent: str,
    error_type: str = "unknown",
    error_message: str | None = None,
) -> None:
    """Record an agent error.

    Args:
        agent: Name of the agent.
        error_type: Type/category of the error.
        error_message: Optional error message.
    """
    if _error_count is not None:
        _error_count.labels(agent=agent, error_type=error_type).inc()

    logger.warning(
        "Agent error recorded",
        extra={
            "agent": agent,
            "error_type": error_type,
            "error_message": error_message,
            "trace_id": get_trace_id(),
        },
    )


def update_error_rate(agent: str, error_rate: float) -> None:
    """Update the error rate gauge for an agent.

    Args:
        agent: Name of the agent.
        error_rate: Error rate as a float between 0 and 1.
    """
    if _error_rate is not None:
        _error_rate.labels(agent=agent).set(max(0.0, min(1.0, error_rate)))


def record_tool_call(
    tool: str,
    success: bool = True,
    duration_seconds: float | None = None,
) -> None:
    """Record a tool call.

    Args:
        tool: Name of the tool.
        success: Whether the call was successful.
        duration_seconds: Optional duration of the call.
    """
    status = "success" if success else "failure"

    if _tool_calls is not None:
        _tool_calls.labels(tool=tool, status=status).inc()

    logger.debug(
        "Tool call recorded",
        extra={
            "tool": tool,
            "status": status,
            "duration_seconds": duration_seconds,
            "trace_id": get_trace_id(),
        },
    )


def update_tool_success_rate(tool: str, success_rate: float) -> None:
    """Update the success rate gauge for a tool.

    Args:
        tool: Name of the tool.
        success_rate: Success rate as a float between 0 and 1.
    """
    if _tool_success_rate is not None:
        _tool_success_rate.labels(tool=tool).set(max(0.0, min(1.0, success_rate)))


def record_knowledge_query(
    operation: str = "search",
    success: bool = True,
) -> None:
    """Record a knowledge base operation.

    Args:
        operation: Type of operation (search, save, etc.).
        success: Whether the operation was successful.
    """
    status = "success" if success else "failure"

    if _knowledge_queries is not None:
        _knowledge_queries.labels(operation=operation, status=status).inc()


def update_active_sessions(count: int) -> None:
    """Update the active sessions count.

    Args:
        count: Current number of active sessions.
    """
    if _active_sessions is not None:
        _active_sessions.set(max(0, count))


def increment_active_sessions() -> None:
    """Increment the active sessions count by 1."""
    if _active_sessions is not None:
        _active_sessions.inc()


def decrement_active_sessions() -> None:
    """Decrement the active sessions count by 1."""
    if _active_sessions is not None:
        _active_sessions.dec()


def metric_tool(tool_name: str | None = None) -> Callable[[F], F]:
    """Decorator to automatically record metrics for tool calls.

    Args:
        tool_name: Name for the tool (defaults to function name).

    Returns:
        Decorated function with metrics recording.

    Example:
        @metric_tool()
        def web_search(query: str) -> str:
            ...
    """
    import functools

    def decorator(func: F) -> F:
        name = tool_name or func.__name__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            success = True

            try:
                result = func(*args, **kwargs)
                return result
            except Exception:
                success = False
                raise
            finally:
                duration = time.time() - start_time
                record_tool_call(name, success=success, duration_seconds=duration)

        return wrapper  # type: ignore

    return decorator


def metric_agent(agent_name: str | None = None) -> Callable[[F], F]:
    """Decorator to automatically record metrics for agent operations.

    Args:
        agent_name: Name for the agent (defaults to function name).

    Returns:
        Decorated function with metrics recording.

    Example:
        @metric_agent("researcher")
        async def run_researcher(query: str) -> dict:
            ...
    """
    import functools

    def decorator(func: F) -> F:
        name = agent_name or func.__name__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with record_request_duration(name) as ctx:
                try:
                    result = func(*args, **kwargs)
                    ctx["status"] = "success"
                    return result
                except Exception as e:
                    ctx["status"] = "error"
                    record_error(name, error_type=type(e).__name__, error_message=str(e))
                    raise

        return wrapper  # type: ignore

    return decorator
