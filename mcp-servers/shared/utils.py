"""Shared utilities for MCP servers.

Common functionality used across all ResearchCrew MCP servers.
"""

import logging
import sys
from typing import Any, Optional


# ============================================================================
# Custom Exceptions
# ============================================================================


class MCPError(Exception):
    """Base exception for MCP server errors."""

    def __init__(
        self,
        message: str,
        error_code: str = "MCP_ERROR",
        details: Optional[dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for MCP response."""
        return {
            "error": True,
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
        }


class ValidationError(MCPError):
    """Raised when input validation fails."""

    def __init__(self, message: str, field: Optional[str] = None):
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            details={"field": field} if field else {},
        )
        self.field = field


class ToolError(MCPError):
    """Raised when a tool execution fails."""

    def __init__(
        self,
        message: str,
        tool_name: str,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            error_code="TOOL_ERROR",
            details={
                "tool_name": tool_name,
                "original_error": str(original_error) if original_error else None,
            },
        )
        self.tool_name = tool_name
        self.original_error = original_error


# ============================================================================
# Response Formatting
# ============================================================================


def format_error_response(
    error: Exception,
    include_traceback: bool = False,
) -> dict[str, Any]:
    """Format an error for MCP response.

    Args:
        error: The exception to format.
        include_traceback: Whether to include traceback info (debug only).

    Returns:
        Formatted error dictionary.
    """
    if isinstance(error, MCPError):
        response = error.to_dict()
    else:
        response = {
            "error": True,
            "error_code": "INTERNAL_ERROR",
            "message": str(error),
            "details": {},
        }

    if include_traceback:
        import traceback

        response["traceback"] = traceback.format_exc()

    return response


def format_success_response(
    data: Any,
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Format a successful response.

    Args:
        data: The response data.
        metadata: Optional metadata to include.

    Returns:
        Formatted success response.
    """
    response = {
        "success": True,
        "data": data,
    }
    if metadata:
        response["metadata"] = metadata
    return response


# ============================================================================
# Validation
# ============================================================================


def validate_required_param(
    params: dict[str, Any],
    param_name: str,
    param_type: Optional[type] = None,
) -> Any:
    """Validate that a required parameter exists and has correct type.

    Args:
        params: Dictionary of parameters.
        param_name: Name of required parameter.
        param_type: Optional type to validate against.

    Returns:
        The validated parameter value.

    Raises:
        ValidationError: If validation fails.
    """
    if param_name not in params:
        raise ValidationError(
            message=f"Missing required parameter: {param_name}",
            field=param_name,
        )

    value = params[param_name]

    if value is None:
        raise ValidationError(
            message=f"Parameter cannot be null: {param_name}",
            field=param_name,
        )

    if param_type is not None and not isinstance(value, param_type):
        raise ValidationError(
            message=f"Parameter '{param_name}' must be {param_type.__name__}, got {type(value).__name__}",
            field=param_name,
        )

    return value


def validate_optional_param(
    params: dict[str, Any],
    param_name: str,
    default: Any = None,
    param_type: Optional[type] = None,
) -> Any:
    """Validate an optional parameter if present.

    Args:
        params: Dictionary of parameters.
        param_name: Name of parameter.
        default: Default value if not present.
        param_type: Optional type to validate against.

    Returns:
        The parameter value or default.

    Raises:
        ValidationError: If type validation fails.
    """
    if param_name not in params:
        return default

    value = params[param_name]

    if value is None:
        return default

    if param_type is not None and not isinstance(value, param_type):
        raise ValidationError(
            message=f"Parameter '{param_name}' must be {param_type.__name__}, got {type(value).__name__}",
            field=param_name,
        )

    return value


# ============================================================================
# Logging
# ============================================================================

_loggers: dict[str, logging.Logger] = {}


def setup_logging(
    name: str,
    level: int = logging.INFO,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """Set up logging for an MCP server.

    Args:
        name: Logger name (usually server name).
        level: Logging level.
        format_string: Optional custom format string.

    Returns:
        Configured logger.
    """
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(level)

        format_string = format_string or "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
        formatter = logging.Formatter(format_string)
        handler.setFormatter(formatter)

        logger.addHandler(handler)

    _loggers[name] = logger
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get an existing logger or create a new one.

    Args:
        name: Logger name.

    Returns:
        Logger instance.
    """
    if name in _loggers:
        return _loggers[name]
    return setup_logging(name)


# ============================================================================
# Text Processing Utilities
# ============================================================================


def truncate_text(text: str, max_length: int = 10000, suffix: str = "...") -> str:
    """Truncate text to a maximum length.

    Args:
        text: Text to truncate.
        max_length: Maximum length.
        suffix: Suffix to add when truncated.

    Returns:
        Truncated text.
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def clean_text(text: str) -> str:
    """Clean text by removing extra whitespace.

    Args:
        text: Text to clean.

    Returns:
        Cleaned text.
    """
    import re

    # Replace multiple whitespace with single space
    text = re.sub(r"\s+", " ", text)
    return text.strip()
