"""Structured Logging Configuration for ResearchCrew

Provides JSON-formatted logging with trace ID correlation for production use.

Usage:
    from utils.logging_config import configure_logging, get_logger

    configure_logging(level="INFO", json_format=True)
    logger = get_logger(__name__)

    logger.info("Processing request", extra={"query": query, "session_id": session_id})
"""

import logging
import sys
from datetime import datetime
from typing import Any, Optional

try:
    from pythonjsonlogger.json import JsonFormatter as jsonlogger

    JSON_LOGGER_AVAILABLE = True
except ImportError:
    try:
        # Fallback for older versions
        from pythonjsonlogger import jsonlogger

        JSON_LOGGER_AVAILABLE = True
    except ImportError:
        JSON_LOGGER_AVAILABLE = False

from utils.tracing import get_trace_id, get_span_id

# Custom log record factory to add trace context
_original_factory = logging.getLogRecordFactory()


def _trace_record_factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
    """Factory that adds trace context to log records."""
    record = _original_factory(*args, **kwargs)

    # Add trace and span IDs if available
    trace_id = get_trace_id()
    span_id = get_span_id()

    record.trace_id = trace_id or "none"  # type: ignore
    record.span_id = span_id or "none"  # type: ignore

    return record


class TraceContextFilter(logging.Filter):
    """Filter that adds trace context to log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Add trace context and always return True."""
        if not hasattr(record, "trace_id"):
            record.trace_id = get_trace_id() or "none"  # type: ignore
        if not hasattr(record, "span_id"):
            record.span_id = get_span_id() or "none"  # type: ignore
        return True


class CustomJsonFormatter(jsonlogger if JSON_LOGGER_AVAILABLE else object):
    """Custom JSON formatter with standard fields."""

    def __init__(self, *args: Any, **kwargs: Any):
        if not JSON_LOGGER_AVAILABLE:
            return
        super().__init__(*args, **kwargs)

    def add_fields(
        self,
        log_record: dict[str, Any],
        record: logging.LogRecord,
        message_dict: dict[str, Any],
    ) -> None:
        """Add custom fields to the log record."""
        super().add_fields(log_record, record, message_dict)

        # Add standard fields
        log_record["timestamp"] = datetime.utcnow().isoformat() + "Z"
        log_record["level"] = record.levelname
        log_record["logger"] = record.name
        log_record["service"] = "researchcrew"

        # Add trace context
        log_record["trace_id"] = getattr(record, "trace_id", "none")
        log_record["span_id"] = getattr(record, "span_id", "none")

        # Add source location for errors
        if record.levelno >= logging.WARNING:
            log_record["source"] = {
                "file": record.pathname,
                "line": record.lineno,
                "function": record.funcName,
            }


class ConsoleFormatter(logging.Formatter):
    """Human-readable console formatter with colors."""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with colors and trace context."""
        color = self.COLORS.get(record.levelname, "")
        reset = self.RESET

        # Get trace context
        trace_id = getattr(record, "trace_id", "none")
        if trace_id and trace_id != "none":
            trace_suffix = f" [trace:{trace_id[:8]}]"
        else:
            trace_suffix = ""

        # Format the message
        formatted = super().format(record)

        return f"{color}{record.levelname:8}{reset} {formatted}{trace_suffix}"


def configure_logging(
    level: str = "INFO",
    json_format: bool = False,
    include_trace: bool = True,
    log_file: Optional[str] = None,
) -> None:
    """Configure logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        json_format: Use JSON format (recommended for production).
        include_trace: Include trace IDs in logs.
        log_file: Optional file to write logs to.
    """
    # Set up the log record factory for trace context
    if include_trace:
        logging.setLogRecordFactory(_trace_record_factory)

    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))

    if json_format and JSON_LOGGER_AVAILABLE:
        # JSON format for production
        formatter = CustomJsonFormatter(
            "%(timestamp)s %(level)s %(name)s %(message)s"
        )
    else:
        # Human-readable format for development
        formatter = ConsoleFormatter(
            "%(asctime)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    console_handler.setFormatter(formatter)

    # Add trace context filter
    if include_trace:
        console_handler.addFilter(TraceContextFilter())

    root_logger.addHandler(console_handler)

    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))

        if JSON_LOGGER_AVAILABLE:
            file_formatter = CustomJsonFormatter(
                "%(timestamp)s %(level)s %(name)s %(message)s"
            )
        else:
            file_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )

        file_handler.setFormatter(file_formatter)

        if include_trace:
            file_handler.addFilter(TraceContextFilter())

        root_logger.addHandler(file_handler)

    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("opentelemetry").setLevel(logging.WARNING)

    logging.info(
        "Logging configured",
        extra={
            "log_level": level,
            "json_format": json_format,
            "include_trace": include_trace,
            "log_file": log_file,
        },
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with trace context support.

    Args:
        name: Logger name (typically __name__).

    Returns:
        Logger instance.
    """
    logger = logging.getLogger(name)

    # Add trace context filter if not already present
    has_filter = any(isinstance(f, TraceContextFilter) for f in logger.filters)
    if not has_filter:
        logger.addFilter(TraceContextFilter())

    return logger


class LogContext:
    """Context manager for adding context to logs within a scope.

    Example:
        with LogContext(session_id="123", user_id="456"):
            logger.info("Processing request")  # Will include session_id and user_id
    """

    _context: dict[str, Any] = {}

    def __init__(self, **kwargs: Any):
        self.updates = kwargs
        self.previous: dict[str, Any] = {}

    def __enter__(self) -> "LogContext":
        # Save previous values and update
        for key, value in self.updates.items():
            self.previous[key] = LogContext._context.get(key)
            LogContext._context[key] = value
        return self

    def __exit__(self, *args: Any) -> None:
        # Restore previous values
        for key, value in self.previous.items():
            if value is None:
                LogContext._context.pop(key, None)
            else:
                LogContext._context[key] = value

    @classmethod
    def get_context(cls) -> dict[str, Any]:
        """Get the current log context."""
        return cls._context.copy()

    @classmethod
    def clear(cls) -> None:
        """Clear all context."""
        cls._context.clear()
