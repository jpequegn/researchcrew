"""OpenTelemetry Tracing for ResearchCrew

Provides distributed tracing capabilities for visualizing and debugging
full agent execution workflows.

Usage:
    from utils.tracing import get_tracer, trace_tool, trace_agent

    tracer = get_tracer()
    with tracer.start_as_current_span("my_operation") as span:
        span.set_attribute("key", "value")
        # do work
"""

import functools
import logging
import os
from contextlib import contextmanager
from typing import Any, Callable, Generator, Optional, TypeVar

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace import Span, SpanKind, Status, StatusCode

logger = logging.getLogger(__name__)

# Type variable for generic function decorators
F = TypeVar("F", bound=Callable[..., Any])

# Global tracer provider
_tracer_provider: Optional[TracerProvider] = None
_initialized: bool = False

# Service name and version
SERVICE_NAME = "researchcrew"
SERVICE_VERSION = "0.1.0"


def init_tracing(
    service_name: str = SERVICE_NAME,
    service_version: str = SERVICE_VERSION,
    exporter_type: str = "console",
    otlp_endpoint: Optional[str] = None,
) -> TracerProvider:
    """Initialize OpenTelemetry tracing.

    Args:
        service_name: Name of the service for trace identification.
        service_version: Version of the service.
        exporter_type: Type of exporter - "console", "otlp", or "none".
        otlp_endpoint: OTLP endpoint URL (required if exporter_type is "otlp").

    Returns:
        The configured TracerProvider.
    """
    global _tracer_provider, _initialized

    if _initialized and _tracer_provider:
        return _tracer_provider

    # Create resource with service information
    resource = Resource.create(
        {
            "service.name": service_name,
            "service.version": service_version,
            "deployment.environment": os.environ.get("ENVIRONMENT", "development"),
        }
    )

    # Create tracer provider
    _tracer_provider = TracerProvider(resource=resource)

    # Configure exporter based on type
    if exporter_type == "console":
        exporter = ConsoleSpanExporter()
        _tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
        logger.info("Tracing initialized with console exporter")

    elif exporter_type == "otlp":
        if not otlp_endpoint:
            otlp_endpoint = os.environ.get(
                "OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"
            )

        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )

            exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
            _tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
            logger.info(f"Tracing initialized with OTLP exporter at {otlp_endpoint}")
        except ImportError:
            logger.warning("OTLP exporter not available, falling back to console")
            exporter = ConsoleSpanExporter()
            _tracer_provider.add_span_processor(BatchSpanProcessor(exporter))

    elif exporter_type == "none":
        logger.info("Tracing initialized with no exporter (traces will be discarded)")

    else:
        logger.warning(f"Unknown exporter type: {exporter_type}, using console")
        exporter = ConsoleSpanExporter()
        _tracer_provider.add_span_processor(BatchSpanProcessor(exporter))

    # Set the global tracer provider
    trace.set_tracer_provider(_tracer_provider)
    _initialized = True

    return _tracer_provider


def get_tracer(name: str = SERVICE_NAME) -> trace.Tracer:
    """Get a tracer instance.

    Args:
        name: Name for the tracer (typically module name).

    Returns:
        A Tracer instance.
    """
    if not _initialized:
        # Auto-initialize with console exporter for development
        init_tracing()

    return trace.get_tracer(name)


def reset_tracing() -> None:
    """Reset the tracing configuration. Useful for testing."""
    global _tracer_provider, _initialized
    _tracer_provider = None
    _initialized = False


@contextmanager
def trace_span(
    name: str,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: Optional[dict[str, Any]] = None,
) -> Generator[Span, None, None]:
    """Context manager for creating a traced span.

    Args:
        name: Name of the span.
        kind: Kind of span (INTERNAL, CLIENT, SERVER, etc.).
        attributes: Initial attributes to set on the span.

    Yields:
        The active span.

    Example:
        with trace_span("my_operation", attributes={"query": "test"}) as span:
            result = do_work()
            span.set_attribute("result_count", len(result))
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(name, kind=kind) as span:
        if attributes:
            for key, value in attributes.items():
                if value is not None:
                    span.set_attribute(key, _sanitize_attribute(value))
        yield span


def _sanitize_attribute(value: Any) -> Any:
    """Sanitize attribute value for OpenTelemetry.

    OpenTelemetry only accepts certain types: str, bool, int, float,
    and sequences of those types.
    """
    if isinstance(value, (str, bool, int, float)):
        return value
    elif isinstance(value, (list, tuple)):
        return [_sanitize_attribute(v) for v in value]
    else:
        return str(value)


def trace_tool(
    tool_name: Optional[str] = None,
    record_args: bool = True,
    record_result: bool = True,
) -> Callable[[F], F]:
    """Decorator for tracing tool calls.

    Args:
        tool_name: Name for the tool span (defaults to function name).
        record_args: Whether to record function arguments.
        record_result: Whether to record the result.

    Returns:
        Decorated function with tracing.

    Example:
        @trace_tool()
        def web_search(query: str) -> str:
            ...
    """

    def decorator(func: F) -> F:
        name = tool_name or f"tool.{func.__name__}"

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            attributes: dict[str, Any] = {
                "tool.name": func.__name__,
                "tool.type": "function",
            }

            if record_args:
                # Record positional args with generic names
                for i, arg in enumerate(args):
                    attr_value = str(arg)[:500]  # Truncate long values
                    attributes[f"tool.arg.{i}"] = attr_value

                # Record keyword args
                for key, value in kwargs.items():
                    attr_value = str(value)[:500]
                    attributes[f"tool.kwarg.{key}"] = attr_value

            with trace_span(name, kind=SpanKind.CLIENT, attributes=attributes) as span:
                try:
                    result = func(*args, **kwargs)

                    if record_result and result is not None:
                        result_str = str(result)[:1000]  # Truncate long results
                        span.set_attribute("tool.result_preview", result_str)
                        span.set_attribute("tool.result_length", len(str(result)))

                    span.set_status(Status(StatusCode.OK))
                    return result

                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        return wrapper  # type: ignore

    return decorator


def trace_agent(
    agent_name: Optional[str] = None,
    record_input: bool = True,
    record_output: bool = True,
) -> Callable[[F], F]:
    """Decorator for tracing agent operations.

    Args:
        agent_name: Name for the agent span (defaults to function name).
        record_input: Whether to record the input.
        record_output: Whether to record the output.

    Returns:
        Decorated function with tracing.

    Example:
        @trace_agent("researcher")
        def run_researcher(query: str) -> dict:
            ...
    """

    def decorator(func: F) -> F:
        name = agent_name or f"agent.{func.__name__}"

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            attributes: dict[str, Any] = {
                "agent.name": name.replace("agent.", ""),
                "agent.function": func.__name__,
            }

            if record_input and args:
                # Assume first arg is the main input (query)
                input_preview = str(args[0])[:500]
                attributes["agent.input_preview"] = input_preview

            with trace_span(name, kind=SpanKind.INTERNAL, attributes=attributes) as span:
                try:
                    result = func(*args, **kwargs)

                    if record_output and result is not None:
                        if isinstance(result, dict):
                            span.set_attribute("agent.output_keys", list(result.keys()))
                        result_str = str(result)[:1000]
                        span.set_attribute("agent.output_preview", result_str)

                    span.set_status(Status(StatusCode.OK))
                    return result

                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        return wrapper  # type: ignore

    return decorator


def trace_llm_call(
    model_name: str,
    prompt: str,
    response: Optional[str] = None,
    token_count: Optional[int] = None,
    error: Optional[Exception] = None,
) -> None:
    """Record an LLM call as a span event on the current span.

    This is useful for recording LLM interactions when you don't have
    control over the LLM call itself (e.g., inside ADK).

    Args:
        model_name: Name of the LLM model.
        prompt: The prompt sent to the LLM.
        response: The response from the LLM (if successful).
        token_count: Estimated token count.
        error: Any error that occurred.
    """
    current_span = trace.get_current_span()
    if not current_span.is_recording():
        return

    attributes: dict[str, Any] = {
        "llm.model": model_name,
        "llm.prompt_preview": prompt[:500] if prompt else "",
        "llm.prompt_length": len(prompt) if prompt else 0,
    }

    if response:
        attributes["llm.response_preview"] = response[:500]
        attributes["llm.response_length"] = len(response)

    if token_count:
        attributes["llm.token_count"] = token_count

    if error:
        attributes["llm.error"] = str(error)
        current_span.add_event("llm.call.error", attributes=attributes)
    else:
        current_span.add_event("llm.call", attributes=attributes)


def add_trace_context(
    query: Optional[str] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    turn_number: Optional[int] = None,
) -> None:
    """Add research context to the current span.

    Args:
        query: The research query.
        session_id: The session identifier.
        user_id: The user identifier.
        turn_number: The conversation turn number.
    """
    current_span = trace.get_current_span()
    if not current_span.is_recording():
        return

    if query:
        current_span.set_attribute("research.query", query[:500])
    if session_id:
        current_span.set_attribute("research.session_id", session_id)
    if user_id:
        current_span.set_attribute("research.user_id", user_id)
    if turn_number is not None:
        current_span.set_attribute("research.turn_number", turn_number)


def get_trace_id() -> Optional[str]:
    """Get the current trace ID as a hex string.

    Returns:
        The trace ID or None if no active span.
    """
    current_span = trace.get_current_span()
    span_context = current_span.get_span_context()
    if span_context.is_valid:
        return format(span_context.trace_id, "032x")
    return None


def get_span_id() -> Optional[str]:
    """Get the current span ID as a hex string.

    Returns:
        The span ID or None if no active span.
    """
    current_span = trace.get_current_span()
    span_context = current_span.get_span_context()
    if span_context.is_valid:
        return format(span_context.span_id, "016x")
    return None
