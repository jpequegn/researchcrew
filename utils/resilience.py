"""Resilience Utilities for ResearchCrew

Provides retry logic, error classification, and resilience patterns
for handling transient failures in external calls.

Usage:
    from utils.resilience import (
        retry_with_backoff,
        RetryPolicy,
        classify_error,
        TransientError,
        PermanentError,
    )

    @retry_with_backoff(policy=RetryPolicy.WEB_SEARCH)
    def web_search(query: str) -> str:
        ...
"""

import logging
import random
import time
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Any, Callable, Optional, Type, TypeVar, Union

import tenacity
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_random,
)

from utils.metrics import record_error, record_tool_call
from utils.logging_config import get_logger
from utils.tracing import get_trace_id, trace_span

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


# ============================================================================
# Error Classification
# ============================================================================


class TransientError(Exception):
    """Errors that may succeed on retry.

    Examples: network timeouts, rate limits, 5xx server errors.
    """

    def __init__(
        self,
        message: str,
        original_error: Optional[Exception] = None,
        retry_after: Optional[float] = None,
    ):
        super().__init__(message)
        self.original_error = original_error
        self.retry_after = retry_after


class PermanentError(Exception):
    """Errors that will not succeed on retry.

    Examples: 4xx client errors, invalid input, authentication failures.
    """

    def __init__(
        self,
        message: str,
        original_error: Optional[Exception] = None,
        error_code: Optional[str] = None,
    ):
        super().__init__(message)
        self.original_error = original_error
        self.error_code = error_code


class RateLimitError(TransientError):
    """Specific error for rate limiting."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: Optional[float] = None,
    ):
        super().__init__(message, retry_after=retry_after)


class TimeoutError(TransientError):
    """Specific error for timeouts."""

    def __init__(self, message: str = "Operation timed out"):
        super().__init__(message)


class NetworkError(TransientError):
    """Specific error for network issues."""

    def __init__(self, message: str = "Network error", original_error: Optional[Exception] = None):
        super().__init__(message, original_error=original_error)


class ServerError(TransientError):
    """Specific error for server-side errors (5xx)."""

    def __init__(
        self,
        message: str = "Server error",
        status_code: int = 500,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(message, original_error=original_error)
        self.status_code = status_code


class ClientError(PermanentError):
    """Specific error for client-side errors (4xx)."""

    def __init__(
        self,
        message: str = "Client error",
        status_code: int = 400,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(message, original_error=original_error, error_code=str(status_code))
        self.status_code = status_code


class AuthenticationError(PermanentError):
    """Specific error for authentication failures."""

    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, error_code="AUTH_FAILED")


class ValidationError(PermanentError):
    """Specific error for validation failures."""

    def __init__(self, message: str = "Validation failed", field: Optional[str] = None):
        super().__init__(message, error_code="VALIDATION_ERROR")
        self.field = field


def classify_error(error: Exception) -> Union[TransientError, PermanentError]:
    """Classify an error as transient or permanent.

    Args:
        error: The exception to classify.

    Returns:
        TransientError if the error may succeed on retry,
        PermanentError if the error will not succeed on retry.
    """
    # Already classified
    if isinstance(error, (TransientError, PermanentError)):
        return error

    error_message = str(error).lower()
    error_type = type(error).__name__

    # Check for timeout errors
    if "timeout" in error_message or error_type in ("TimeoutError", "asyncio.TimeoutError"):
        return TimeoutError(str(error))

    # Check for connection errors
    if any(
        term in error_message
        for term in ("connection", "network", "refused", "unreachable", "dns")
    ):
        return NetworkError(str(error), original_error=error)

    # Check for rate limit errors
    if "rate limit" in error_message or "429" in error_message or "too many" in error_message:
        # Try to extract retry-after
        retry_after = None
        if hasattr(error, "retry_after"):
            retry_after = error.retry_after
        return RateLimitError(str(error), retry_after=retry_after)

    # Check for HTTP errors by status code
    status_code = None
    if hasattr(error, "status_code"):
        status_code = error.status_code
    elif hasattr(error, "response") and hasattr(error.response, "status_code"):
        status_code = error.response.status_code

    if status_code is not None:
        if status_code >= 500:
            return ServerError(str(error), status_code=status_code, original_error=error)
        elif status_code == 401 or status_code == 403:
            return AuthenticationError(str(error))
        elif status_code == 429:
            return RateLimitError(str(error))
        elif 400 <= status_code < 500:
            return ClientError(str(error), status_code=status_code, original_error=error)

    # Check for authentication errors
    if any(term in error_message for term in ("auth", "unauthorized", "forbidden", "permission")):
        return AuthenticationError(str(error))

    # Check for validation errors
    if any(term in error_message for term in ("invalid", "validation", "required field")):
        return ValidationError(str(error))

    # Default to transient for unknown errors (safer to retry)
    return TransientError(str(error), original_error=error)


# ============================================================================
# Retry Policies
# ============================================================================


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    min_wait: float = 1.0  # seconds
    max_wait: float = 60.0  # seconds
    multiplier: float = 2.0  # exponential multiplier
    jitter: float = 0.5  # random jitter factor (0-1)
    timeout: Optional[float] = None  # total operation timeout


class RetryPolicy(Enum):
    """Pre-configured retry policies for different operation types."""

    # Web search: moderate retries, reasonable timeouts
    WEB_SEARCH = RetryConfig(
        max_attempts=3,
        min_wait=1.0,
        max_wait=60.0,
        multiplier=2.0,
        jitter=0.5,
        timeout=90.0,
    )

    # URL reading: similar to web search
    READ_URL = RetryConfig(
        max_attempts=3,
        min_wait=1.0,
        max_wait=30.0,
        multiplier=2.0,
        jitter=0.5,
        timeout=60.0,
    )

    # LLM calls: fewer retries, longer backoff (expensive)
    LLM_CALL = RetryConfig(
        max_attempts=2,
        min_wait=2.0,
        max_wait=120.0,
        multiplier=3.0,
        jitter=0.3,
        timeout=180.0,
    )

    # Knowledge base: quick retries, local operation
    KNOWLEDGE_BASE = RetryConfig(
        max_attempts=2,
        min_wait=0.5,
        max_wait=5.0,
        multiplier=2.0,
        jitter=0.2,
        timeout=30.0,
    )

    # Default: balanced policy
    DEFAULT = RetryConfig(
        max_attempts=3,
        min_wait=1.0,
        max_wait=30.0,
        multiplier=2.0,
        jitter=0.5,
        timeout=60.0,
    )

    # Aggressive: more retries for critical operations
    AGGRESSIVE = RetryConfig(
        max_attempts=5,
        min_wait=0.5,
        max_wait=120.0,
        multiplier=2.0,
        jitter=0.5,
        timeout=300.0,
    )

    # Conservative: fewer retries, fail fast
    CONSERVATIVE = RetryConfig(
        max_attempts=2,
        min_wait=1.0,
        max_wait=10.0,
        multiplier=2.0,
        jitter=0.3,
        timeout=30.0,
    )


# ============================================================================
# Retry Statistics
# ============================================================================


@dataclass
class RetryStats:
    """Statistics for a retry operation."""

    operation: str
    attempts: int
    final_status: str  # "success", "failure", "exhausted"
    total_time: float
    last_error: Optional[str] = None
    trace_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "attempts": self.attempts,
            "final_status": self.final_status,
            "total_time": self.total_time,
            "last_error": self.last_error,
            "trace_id": self.trace_id,
        }


# Global stats storage for testing and debugging
_retry_stats: list[RetryStats] = []


def get_retry_stats() -> list[RetryStats]:
    """Get all recorded retry statistics."""
    return _retry_stats.copy()


def clear_retry_stats() -> None:
    """Clear all retry statistics."""
    _retry_stats.clear()


# ============================================================================
# Retry Decorator
# ============================================================================


def _log_retry_attempt(retry_state: tenacity.RetryCallState) -> None:
    """Log retry attempts for debugging."""
    exception = retry_state.outcome.exception() if retry_state.outcome else None
    logger.warning(
        f"Retry attempt {retry_state.attempt_number}",
        extra={
            "attempt": retry_state.attempt_number,
            "error": str(exception) if exception else None,
            "error_type": type(exception).__name__ if exception else None,
            "next_wait": retry_state.next_action.sleep if retry_state.next_action else None,
            "trace_id": get_trace_id(),
        },
    )


def _record_retry_metrics(
    operation: str,
    attempt: int,
    success: bool,
    duration: float,
    error: Optional[Exception] = None,
) -> None:
    """Record retry metrics."""
    record_tool_call(
        tool=f"{operation}_retry",
        success=success,
        duration_seconds=duration,
    )

    if error:
        classified = classify_error(error)
        error_type = "transient" if isinstance(classified, TransientError) else "permanent"
        record_error(
            agent=operation,
            error_type=error_type,
            error_message=str(error)[:200],
        )


def retry_with_backoff(
    policy: Union[RetryPolicy, RetryConfig] = RetryPolicy.DEFAULT,
    operation_name: Optional[str] = None,
    reraise_permanent: bool = True,
) -> Callable[[F], F]:
    """Decorator to add retry logic with exponential backoff.

    Args:
        policy: Retry policy to use (can be a RetryPolicy enum or RetryConfig).
        operation_name: Name for logging/metrics (defaults to function name).
        reraise_permanent: If True, re-raises PermanentErrors immediately without retry.

    Returns:
        Decorated function with retry logic.

    Example:
        @retry_with_backoff(policy=RetryPolicy.WEB_SEARCH)
        def web_search(query: str) -> str:
            ...
    """
    config = policy.value if isinstance(policy, RetryPolicy) else policy

    def decorator(func: F) -> F:
        name = operation_name or func.__name__

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            attempts = 0
            last_error: Optional[Exception] = None

            # Build wait strategy with jitter
            wait_strategy = wait_exponential(
                multiplier=config.multiplier,
                min=config.min_wait,
                max=config.max_wait,
            ) + wait_random(0, config.max_wait * config.jitter)

            # Build retry predicate
            def should_retry(exception: BaseException) -> bool:
                nonlocal last_error
                last_error = exception if isinstance(exception, Exception) else None

                # Classify the error
                classified = classify_error(exception)  # type: ignore

                # Don't retry permanent errors if configured
                if reraise_permanent and isinstance(classified, PermanentError):
                    logger.info(
                        f"Permanent error, not retrying: {type(classified).__name__}",
                        extra={
                            "operation": name,
                            "error_type": type(classified).__name__,
                            "trace_id": get_trace_id(),
                        },
                    )
                    return False

                # Retry transient errors
                return isinstance(classified, TransientError)

            # Create the retry decorator
            retryer = tenacity.Retrying(
                wait=wait_strategy,
                stop=stop_after_attempt(config.max_attempts),
                retry=tenacity.retry_if_exception(should_retry),
                before_sleep=_log_retry_attempt,
                reraise=True,
            )

            # Execute with retry
            with trace_span(
                f"retry.{name}",
                attributes={
                    "retry.policy": policy.name if isinstance(policy, RetryPolicy) else "custom",
                    "retry.max_attempts": config.max_attempts,
                },
            ) as span:
                try:
                    for attempt in retryer:
                        attempts = attempt.retry_state.attempt_number
                        with attempt:
                            result = func(*args, **kwargs)

                            # Record success stats
                            duration = time.time() - start_time
                            stats = RetryStats(
                                operation=name,
                                attempts=attempts,
                                final_status="success",
                                total_time=duration,
                                trace_id=get_trace_id(),
                            )
                            _retry_stats.append(stats)
                            _record_retry_metrics(name, attempts, True, duration)

                            span.set_attribute("retry.attempts", attempts)
                            span.set_attribute("retry.status", "success")

                            logger.debug(
                                f"Operation succeeded after {attempts} attempt(s)",
                                extra={
                                    "operation": name,
                                    "attempts": attempts,
                                    "duration": duration,
                                    "trace_id": get_trace_id(),
                                },
                            )

                            return result

                except RetryError as e:
                    # All retries exhausted (tenacity wrapper)
                    duration = time.time() - start_time
                    stats = RetryStats(
                        operation=name,
                        attempts=attempts,
                        final_status="exhausted",
                        total_time=duration,
                        last_error=str(last_error) if last_error else None,
                        trace_id=get_trace_id(),
                    )
                    _retry_stats.append(stats)
                    _record_retry_metrics(name, attempts, False, duration, last_error)

                    span.set_attribute("retry.attempts", attempts)
                    span.set_attribute("retry.status", "exhausted")
                    span.set_attribute("error", True)

                    logger.error(
                        f"All {attempts} retries exhausted for {name}",
                        extra={
                            "operation": name,
                            "attempts": attempts,
                            "duration": duration,
                            "last_error": str(last_error),
                            "trace_id": get_trace_id(),
                        },
                    )

                    # Re-raise the last error
                    if last_error:
                        raise last_error from e
                    raise

                except Exception as e:
                    # When reraise=True, tenacity re-raises the last exception directly
                    # Check if we've exhausted all attempts
                    duration = time.time() - start_time
                    retries_exhausted = attempts >= config.max_attempts

                    if retries_exhausted:
                        # All retries were used
                        final_status = "exhausted"
                        log_message = f"All {attempts} retries exhausted for {name}"
                    else:
                        # Non-retryable error (permanent or should_retry returned False)
                        final_status = "failure"
                        log_message = f"Non-retryable error after {attempts} attempt(s) for {name}"

                    stats = RetryStats(
                        operation=name,
                        attempts=attempts,
                        final_status=final_status,
                        total_time=duration,
                        last_error=str(e),
                        trace_id=get_trace_id(),
                    )
                    _retry_stats.append(stats)
                    _record_retry_metrics(name, attempts, False, duration, e)

                    span.set_attribute("retry.attempts", attempts)
                    span.set_attribute("retry.status", final_status)
                    span.set_attribute("error", True)

                    logger.error(
                        log_message,
                        extra={
                            "operation": name,
                            "attempts": attempts,
                            "duration": duration,
                            "last_error": str(e),
                            "trace_id": get_trace_id(),
                        },
                    )

                    raise

        return wrapper  # type: ignore

    return decorator


# ============================================================================
# Utility Functions
# ============================================================================


def with_timeout(
    timeout: float,
    operation_name: str = "operation",
) -> Callable[[F], F]:
    """Decorator to add timeout to synchronous operations.

    Args:
        timeout: Maximum time in seconds.
        operation_name: Name for error messages.

    Returns:
        Decorated function with timeout.

    Note: For async functions, use asyncio.timeout instead.
    """
    import signal
    import sys

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Signal-based timeout only works on Unix
            if sys.platform == "win32":
                logger.warning("Timeout decorator not supported on Windows, running without timeout")
                return func(*args, **kwargs)

            def timeout_handler(signum: int, frame: Any) -> None:
                raise TimeoutError(f"{operation_name} timed out after {timeout}s")

            # Set the signal handler
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.setitimer(signal.ITIMER_REAL, timeout)

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                # Restore the old handler
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, old_handler)

        return wrapper  # type: ignore

    return decorator


def is_retriable(error: Exception) -> bool:
    """Check if an error should be retried.

    Args:
        error: The exception to check.

    Returns:
        True if the error is transient and should be retried.
    """
    classified = classify_error(error)
    return isinstance(classified, TransientError)


def get_retry_wait(error: Exception, attempt: int, policy: RetryConfig) -> float:
    """Calculate wait time for a retry attempt.

    Args:
        error: The exception that occurred.
        attempt: Current attempt number (1-based).
        policy: Retry configuration.

    Returns:
        Wait time in seconds.
    """
    # Check for explicit retry-after
    if isinstance(error, TransientError) and error.retry_after:
        return min(error.retry_after, policy.max_wait)

    # Calculate exponential backoff with jitter
    base_wait = min(
        policy.min_wait * (policy.multiplier ** (attempt - 1)),
        policy.max_wait,
    )

    # Add jitter
    jitter = random.uniform(0, base_wait * policy.jitter)

    return base_wait + jitter
