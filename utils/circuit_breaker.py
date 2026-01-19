"""Circuit Breaker Pattern for ResearchCrew

Implements the circuit breaker pattern to prevent cascade failures
when external services are degraded.

Usage:
    from utils.circuit_breaker import (
        CircuitBreaker,
        get_circuit_breaker,
        circuit_breaker_call,
    )

    # Create or get a circuit breaker
    cb = get_circuit_breaker("web_search")

    # Call with circuit breaker protection
    result = cb.call(my_function, arg1, arg2)

    # Or use as decorator
    @circuit_breaker_call("my_service")
    def my_function():
        ...
"""

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from functools import wraps
from typing import Any, Callable, Optional, TypeVar

from utils.metrics import record_error
from utils.logging_config import get_logger
from utils.tracing import get_trace_id, trace_span

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing fast, not calling service
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitOpenError(Exception):
    """Raised when circuit is open and call is rejected."""

    def __init__(
        self,
        circuit_name: str,
        message: Optional[str] = None,
        retry_after: Optional[float] = None,
    ):
        self.circuit_name = circuit_name
        self.retry_after = retry_after
        msg = message or f"Circuit '{circuit_name}' is open"
        super().__init__(msg)


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""

    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 2  # Successes in half-open to close
    recovery_timeout: float = 30.0  # Seconds before trying half-open
    failure_window: float = 60.0  # Window to count failures
    half_open_max_calls: int = 3  # Max calls allowed in half-open


# Pre-configured circuit breaker configs
class CircuitBreakerPreset(Enum):
    """Pre-configured circuit breaker settings."""

    # External APIs - more forgiving
    EXTERNAL_API = CircuitBreakerConfig(
        failure_threshold=5,
        success_threshold=2,
        recovery_timeout=30.0,
        failure_window=60.0,
        half_open_max_calls=3,
    )

    # LLM calls - more conservative (expensive)
    LLM_CALL = CircuitBreakerConfig(
        failure_threshold=3,
        success_threshold=2,
        recovery_timeout=60.0,
        failure_window=120.0,
        half_open_max_calls=2,
    )

    # Local services - quick recovery
    LOCAL_SERVICE = CircuitBreakerConfig(
        failure_threshold=10,
        success_threshold=3,
        recovery_timeout=10.0,
        failure_window=30.0,
        half_open_max_calls=5,
    )

    # Default balanced config
    DEFAULT = CircuitBreakerConfig()


@dataclass
class CircuitBreakerStats:
    """Statistics for a circuit breaker."""

    name: str
    state: CircuitState
    failure_count: int
    success_count: int
    total_calls: int
    total_failures: int
    total_trips: int  # Times circuit opened
    last_failure_time: Optional[datetime]
    last_state_change: Optional[datetime]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "total_calls": self.total_calls,
            "total_failures": self.total_failures,
            "total_trips": self.total_trips,
            "last_failure_time": (
                self.last_failure_time.isoformat() if self.last_failure_time else None
            ),
            "last_state_change": (
                self.last_state_change.isoformat() if self.last_state_change else None
            ),
        }


class CircuitBreaker:
    """Circuit breaker implementation.

    States:
    - CLOSED: Normal operation, calls pass through
    - OPEN: Service is failing, calls are rejected immediately
    - HALF_OPEN: Testing if service recovered

    Transitions:
    - CLOSED -> OPEN: When failure_threshold reached in failure_window
    - OPEN -> HALF_OPEN: After recovery_timeout passes
    - HALF_OPEN -> CLOSED: After success_threshold successes
    - HALF_OPEN -> OPEN: On any failure
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        fallback: Optional[Callable[..., Any]] = None,
        on_state_change: Optional[Callable[[CircuitState, CircuitState], None]] = None,
    ):
        """Initialize circuit breaker.

        Args:
            name: Identifier for this circuit breaker.
            config: Configuration settings.
            fallback: Optional fallback function when circuit is open.
            on_state_change: Callback when state changes.
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.fallback = fallback
        self.on_state_change = on_state_change

        self._state = CircuitState.CLOSED
        self._failure_times: list[float] = []
        self._success_count = 0
        self._half_open_calls = 0
        self._last_failure_time: Optional[datetime] = None
        self._last_state_change: Optional[datetime] = None
        self._opened_at: Optional[float] = None

        # Counters for stats
        self._total_calls = 0
        self._total_failures = 0
        self._total_trips = 0

        # Thread safety
        self._lock = threading.RLock()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            # Check if we should transition from OPEN to HALF_OPEN
            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to(CircuitState.HALF_OPEN)
            return self._state

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to try half-open."""
        if self._opened_at is None:
            return True
        return time.time() - self._opened_at >= self.config.recovery_timeout

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        old_state = self._state
        if old_state == new_state:
            return

        self._state = new_state
        self._last_state_change = datetime.now(timezone.utc)

        if new_state == CircuitState.OPEN:
            self._opened_at = time.time()
            self._total_trips += 1
            logger.warning(
                f"Circuit '{self.name}' OPENED",
                extra={
                    "circuit": self.name,
                    "old_state": old_state.value,
                    "new_state": new_state.value,
                    "failure_count": len(self._failure_times),
                    "trace_id": get_trace_id(),
                },
            )
        elif new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
            self._success_count = 0
            logger.info(
                f"Circuit '{self.name}' entering HALF_OPEN",
                extra={
                    "circuit": self.name,
                    "old_state": old_state.value,
                    "new_state": new_state.value,
                    "trace_id": get_trace_id(),
                },
            )
        elif new_state == CircuitState.CLOSED:
            self._failure_times.clear()
            self._success_count = 0
            self._opened_at = None
            logger.info(
                f"Circuit '{self.name}' CLOSED",
                extra={
                    "circuit": self.name,
                    "old_state": old_state.value,
                    "new_state": new_state.value,
                    "trace_id": get_trace_id(),
                },
            )

        # Call state change callback
        if self.on_state_change:
            try:
                self.on_state_change(old_state, new_state)
            except Exception as e:
                logger.error(f"State change callback failed: {e}")

    def _record_failure(self) -> None:
        """Record a failure."""
        now = time.time()
        self._failure_times.append(now)
        self._last_failure_time = datetime.now(timezone.utc)
        self._total_failures += 1

        # Remove old failures outside the window
        cutoff = now - self.config.failure_window
        self._failure_times = [t for t in self._failure_times if t > cutoff]

        # Check if we should open the circuit
        if len(self._failure_times) >= self.config.failure_threshold:
            self._transition_to(CircuitState.OPEN)

        # Record in metrics
        record_error(
            agent=f"circuit_{self.name}",
            error_type="circuit_failure",
            error_message=f"Circuit {self.name} recorded failure",
        )

    def _record_success(self) -> None:
        """Record a success."""
        self._success_count += 1

        if self._state == CircuitState.HALF_OPEN:
            if self._success_count >= self.config.success_threshold:
                self._transition_to(CircuitState.CLOSED)

    def call(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute a function with circuit breaker protection.

        Args:
            func: Function to call.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Result of the function call.

        Raises:
            CircuitOpenError: If circuit is open and no fallback.
        """
        with self._lock:
            self._total_calls += 1
            current_state = self.state  # This may transition OPEN -> HALF_OPEN

            # Check if circuit allows calls
            if current_state == CircuitState.OPEN:
                retry_after = self.config.recovery_timeout
                if self._opened_at:
                    elapsed = time.time() - self._opened_at
                    retry_after = max(0, self.config.recovery_timeout - elapsed)

                logger.debug(
                    f"Circuit '{self.name}' rejecting call (OPEN)",
                    extra={
                        "circuit": self.name,
                        "retry_after": retry_after,
                        "trace_id": get_trace_id(),
                    },
                )

                if self.fallback:
                    return self.fallback(*args, **kwargs)
                raise CircuitOpenError(self.name, retry_after=retry_after)

            # In half-open, limit concurrent calls
            if current_state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.config.half_open_max_calls:
                    if self.fallback:
                        return self.fallback(*args, **kwargs)
                    raise CircuitOpenError(
                        self.name,
                        message=f"Circuit '{self.name}' half-open limit reached",
                    )
                self._half_open_calls += 1

        # Execute the call (outside lock to not block other calls)
        with trace_span(
            f"circuit.{self.name}",
            attributes={
                "circuit.name": self.name,
                "circuit.state": current_state.value,
            },
        ) as span:
            try:
                result = func(*args, **kwargs)

                with self._lock:
                    self._record_success()
                    span.set_attribute("circuit.success", True)

                return result

            except Exception as e:
                with self._lock:
                    self._record_failure()

                    # In half-open, immediately open on failure
                    if self._state == CircuitState.HALF_OPEN:
                        self._transition_to(CircuitState.OPEN)

                    span.set_attribute("circuit.success", False)
                    span.set_attribute("error", True)

                raise

    def get_stats(self) -> CircuitBreakerStats:
        """Get current statistics."""
        with self._lock:
            return CircuitBreakerStats(
                name=self.name,
                state=self._state,
                failure_count=len(self._failure_times),
                success_count=self._success_count,
                total_calls=self._total_calls,
                total_failures=self._total_failures,
                total_trips=self._total_trips,
                last_failure_time=self._last_failure_time,
                last_state_change=self._last_state_change,
            )

    def reset(self) -> None:
        """Reset the circuit breaker to closed state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_times.clear()
            self._success_count = 0
            self._half_open_calls = 0
            self._opened_at = None
            self._last_state_change = datetime.now(timezone.utc)
            logger.info(f"Circuit '{self.name}' manually reset")

    def force_open(self) -> None:
        """Force the circuit to open (for testing or maintenance)."""
        with self._lock:
            self._transition_to(CircuitState.OPEN)
            logger.info(f"Circuit '{self.name}' manually opened")


# ============================================================================
# Global Circuit Breaker Registry
# ============================================================================

_circuit_breakers: dict[str, CircuitBreaker] = {}
_registry_lock = threading.Lock()


def get_circuit_breaker(
    name: str,
    config: Optional[CircuitBreakerConfig] = None,
    preset: Optional[CircuitBreakerPreset] = None,
    fallback: Optional[Callable[..., Any]] = None,
) -> CircuitBreaker:
    """Get or create a circuit breaker by name.

    Args:
        name: Identifier for the circuit breaker.
        config: Optional custom configuration.
        preset: Optional preset configuration.
        fallback: Optional fallback function.

    Returns:
        CircuitBreaker instance.
    """
    with _registry_lock:
        if name not in _circuit_breakers:
            if preset:
                config = preset.value
            _circuit_breakers[name] = CircuitBreaker(
                name=name,
                config=config,
                fallback=fallback,
            )
        return _circuit_breakers[name]


def reset_circuit_breaker(name: str) -> bool:
    """Reset a specific circuit breaker.

    Args:
        name: Circuit breaker name.

    Returns:
        True if reset, False if not found.
    """
    with _registry_lock:
        if name in _circuit_breakers:
            _circuit_breakers[name].reset()
            return True
        return False


def reset_all_circuit_breakers() -> None:
    """Reset all circuit breakers."""
    with _registry_lock:
        for cb in _circuit_breakers.values():
            cb.reset()


def clear_circuit_breakers() -> None:
    """Remove all circuit breakers from registry."""
    with _registry_lock:
        _circuit_breakers.clear()


def get_all_circuit_stats() -> dict[str, CircuitBreakerStats]:
    """Get statistics for all circuit breakers."""
    with _registry_lock:
        return {name: cb.get_stats() for name, cb in _circuit_breakers.items()}


# ============================================================================
# Decorator
# ============================================================================


def circuit_breaker_call(
    name: str,
    config: Optional[CircuitBreakerConfig] = None,
    preset: Optional[CircuitBreakerPreset] = None,
    fallback: Optional[Callable[..., Any]] = None,
) -> Callable[[F], F]:
    """Decorator to protect a function with a circuit breaker.

    Args:
        name: Circuit breaker name.
        config: Optional custom configuration.
        preset: Optional preset configuration.
        fallback: Optional fallback function.

    Returns:
        Decorated function.

    Example:
        @circuit_breaker_call("external_api")
        def call_external_api():
            ...
    """

    def decorator(func: F) -> F:
        cb = get_circuit_breaker(name, config=config, preset=preset, fallback=fallback)

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return cb.call(func, *args, **kwargs)

        return wrapper  # type: ignore

    return decorator


# ============================================================================
# Monitoring Helpers
# ============================================================================


def get_circuit_breaker_status() -> dict[str, Any]:
    """Get status summary of all circuit breakers.

    Returns:
        Dictionary with circuit breaker status information.
    """
    stats = get_all_circuit_stats()

    open_circuits = [name for name, s in stats.items() if s.state == CircuitState.OPEN]
    half_open_circuits = [
        name for name, s in stats.items() if s.state == CircuitState.HALF_OPEN
    ]

    return {
        "total_circuits": len(stats),
        "open_circuits": open_circuits,
        "half_open_circuits": half_open_circuits,
        "healthy": len(open_circuits) == 0,
        "circuits": {name: s.to_dict() for name, s in stats.items()},
    }


def is_service_healthy(name: str) -> bool:
    """Check if a circuit breaker indicates healthy service.

    Args:
        name: Circuit breaker name.

    Returns:
        True if circuit is closed (healthy).
    """
    with _registry_lock:
        if name not in _circuit_breakers:
            return True  # No circuit = assumed healthy
        return _circuit_breakers[name].state == CircuitState.CLOSED
