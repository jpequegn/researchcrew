"""Base classes for external agent integrations.

Provides the foundation for creating adapters that bridge
ResearchCrew agents with external agent frameworks.
"""

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Generic, TypeVar

from utils.logging_config import get_logger
from utils.metrics import record_error, record_request_duration
from utils.tracing import get_trace_id, trace_span

logger = get_logger(__name__)

T = TypeVar("T")
R = TypeVar("R")


class AdapterError(Exception):
    """Base exception for adapter errors."""

    def __init__(
        self,
        message: str,
        adapter_name: str,
        original_error: Exception | None = None,
    ):
        super().__init__(message)
        self.adapter_name = adapter_name
        self.original_error = original_error

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": True,
            "adapter": self.adapter_name,
            "message": str(self),
            "original_error": str(self.original_error) if self.original_error else None,
        }


class ValidationError(AdapterError):
    """Raised when input/output validation fails."""

    def __init__(self, message: str, adapter_name: str, field: str | None = None):
        super().__init__(message, adapter_name)
        self.field = field


class StateTranslationError(AdapterError):
    """Raised when state translation between frameworks fails."""

    pass


class TimeoutError(AdapterError):
    """Raised when an adapter call times out."""

    pass


@dataclass
class AdapterConfig:
    """Configuration for external agent adapters."""

    # Name identifier for the adapter
    name: str

    # Timeout for adapter calls (seconds)
    timeout: float = 60.0

    # Whether to validate inputs
    validate_input: bool = True

    # Whether to validate outputs
    validate_output: bool = True

    # Maximum retries for failed calls
    max_retries: int = 2

    # Whether to enable tracing
    enable_tracing: bool = True

    # Whether to record metrics
    enable_metrics: bool = True

    # Custom state translation function
    state_translator: Callable[[Any], Any] | None = None

    # Custom result translator function
    result_translator: Callable[[Any], Any] | None = None


@dataclass
class AdapterResult(Generic[T]):
    """Result from an adapter call."""

    # The result value
    value: T

    # Name of the adapter used
    adapter_name: str

    # Execution time in seconds
    execution_time: float

    # Whether the call was successful
    success: bool = True

    # Optional error message if failed
    error_message: str | None = None

    # Source framework
    source_framework: str = ""

    # Target framework
    target_framework: str = ""

    # Metadata from the call
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter_name": self.adapter_name,
            "success": self.success,
            "execution_time": self.execution_time,
            "source_framework": self.source_framework,
            "target_framework": self.target_framework,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }


@dataclass
class AdapterStats:
    """Statistics for an adapter."""

    adapter_name: str
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_time: float = 0.0
    average_time: float = 0.0
    last_call_time: datetime | None = None
    last_error: str | None = None

    def record_call(self, execution_time: float, success: bool, error: str | None = None) -> None:
        """Record a call."""
        self.total_calls += 1
        self.total_time += execution_time
        self.average_time = self.total_time / self.total_calls
        self.last_call_time = datetime.now(UTC)

        if success:
            self.successful_calls += 1
        else:
            self.failed_calls += 1
            self.last_error = error

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter_name": self.adapter_name,
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "success_rate": self.successful_calls / self.total_calls if self.total_calls > 0 else 0,
            "average_time": self.average_time,
            "last_call_time": self.last_call_time.isoformat() if self.last_call_time else None,
            "last_error": self.last_error,
        }


class ExternalAgentAdapter(ABC, Generic[T, R]):
    """Base class for external agent adapters.

    Provides a consistent interface for integrating with external
    agent frameworks.

    Type Parameters:
        T: Input type for the adapter
        R: Output type from the adapter
    """

    def __init__(self, config: AdapterConfig):
        """Initialize the adapter.

        Args:
            config: Adapter configuration.
        """
        self.config = config
        self._stats = AdapterStats(adapter_name=config.name)

    @property
    @abstractmethod
    def source_framework(self) -> str:
        """The source framework name."""
        pass

    @property
    @abstractmethod
    def target_framework(self) -> str:
        """The target framework name."""
        pass

    @abstractmethod
    async def _execute(self, input_data: T) -> R:
        """Execute the adapter call.

        Args:
            input_data: Input data for the external agent.

        Returns:
            Result from the external agent.

        Raises:
            AdapterError: If execution fails.
        """
        pass

    def _translate_state(self, state: Any) -> Any:
        """Translate state between frameworks.

        Override for custom state translation.

        Args:
            state: State to translate.

        Returns:
            Translated state.
        """
        if self.config.state_translator:
            return self.config.state_translator(state)
        return state

    def _translate_result(self, result: Any) -> Any:
        """Translate result from external framework.

        Override for custom result translation.

        Args:
            result: Result to translate.

        Returns:
            Translated result.
        """
        if self.config.result_translator:
            return self.config.result_translator(result)
        return result

    def _validate_input(self, input_data: T) -> None:
        """Validate input data.

        Override for custom validation.

        Args:
            input_data: Data to validate.

        Raises:
            ValidationError: If validation fails.
        """
        if input_data is None:
            raise ValidationError(
                message="Input data cannot be None",
                adapter_name=self.config.name,
                field="input",
            )

    def _validate_output(self, output: R) -> None:
        """Validate output data.

        Override for custom validation.

        Args:
            output: Data to validate.

        Raises:
            ValidationError: If validation fails.
        """
        pass

    async def execute(self, input_data: T) -> AdapterResult[R]:
        """Execute the adapter with full tracing and metrics.

        Args:
            input_data: Input data for the external agent.

        Returns:
            AdapterResult with the result and metadata.

        Raises:
            AdapterError: If execution fails.
        """
        start_time = time.time()
        result: R | None = None
        error_message: str | None = None
        success = True

        # Create span for tracing
        span_name = f"adapter.{self.config.name}"
        attributes = {
            "adapter.name": self.config.name,
            "adapter.source": self.source_framework,
            "adapter.target": self.target_framework,
            "trace_id": get_trace_id(),
        }

        try:
            with trace_span(span_name, attributes=attributes) as span:
                # Validate input
                if self.config.validate_input:
                    self._validate_input(input_data)

                # Translate state
                translated_input = self._translate_state(input_data)

                # Execute with retry
                last_error: Exception | None = None
                for attempt in range(self.config.max_retries + 1):
                    try:
                        result = await self._execute(translated_input)
                        break
                    except Exception as e:
                        last_error = e
                        if attempt < self.config.max_retries:
                            logger.warning(f"Adapter {self.config.name} attempt {attempt + 1} failed: {e}")
                            continue
                        raise

                # Translate result
                result = self._translate_result(result)

                # Validate output
                if self.config.validate_output:
                    self._validate_output(result)

                span.set_attribute("adapter.success", True)

        except AdapterError:
            success = False
            error_message = str(last_error) if "last_error" in dir() and last_error else "Unknown error"
            raise
        except Exception as e:
            success = False
            error_message = str(e)
            logger.error(f"Adapter {self.config.name} failed: {e}")

            # Record error metric
            if self.config.enable_metrics:
                record_error(
                    agent=f"adapter_{self.config.name}",
                    error_type="adapter_error",
                    error_message=str(e),
                )

            raise AdapterError(
                message=str(e),
                adapter_name=self.config.name,
                original_error=e,
            )
        finally:
            execution_time = time.time() - start_time

            # Record stats
            self._stats.record_call(execution_time, success, error_message)

            # Record metrics
            if self.config.enable_metrics:
                record_request_duration(
                    agent=f"adapter_{self.config.name}",
                    duration=execution_time,
                )

        return AdapterResult(
            value=result,
            adapter_name=self.config.name,
            execution_time=execution_time,
            success=success,
            error_message=error_message,
            source_framework=self.source_framework,
            target_framework=self.target_framework,
        )

    def get_stats(self) -> AdapterStats:
        """Get adapter statistics."""
        return self._stats

    def reset_stats(self) -> None:
        """Reset adapter statistics."""
        self._stats = AdapterStats(adapter_name=self.config.name)


# ============================================================================
# Registry for adapters
# ============================================================================

_adapters: dict[str, ExternalAgentAdapter] = {}


def register_adapter(adapter: ExternalAgentAdapter) -> None:
    """Register an adapter globally.

    Args:
        adapter: Adapter to register.
    """
    _adapters[adapter.config.name] = adapter


def get_adapter(name: str) -> ExternalAgentAdapter | None:
    """Get a registered adapter.

    Args:
        name: Adapter name.

    Returns:
        Adapter if registered, None otherwise.
    """
    return _adapters.get(name)


def get_all_adapter_stats() -> dict[str, AdapterStats]:
    """Get statistics for all registered adapters."""
    return {name: adapter.get_stats() for name, adapter in _adapters.items()}


def clear_adapters() -> None:
    """Clear all registered adapters."""
    _adapters.clear()
