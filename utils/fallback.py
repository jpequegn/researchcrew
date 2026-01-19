"""Fallback Strategies for ResearchCrew

Implements graceful degradation with fallback strategies for when
primary services are unavailable.

Usage:
    from utils.fallback import (
        FallbackChain,
        FallbackStrategy,
        get_tool_fallback,
        get_model_fallback,
    )

    # Use pre-configured fallback chain for a tool
    chain = get_tool_fallback("web_search")
    result = chain.execute(query="AI research")

    # Create custom fallback chain
    chain = FallbackChain([
        PrimaryStrategy(),
        CachedStrategy(),
        EmptyStrategy(),
    ])
    result = chain.execute(my_args)
"""

import hashlib
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from functools import wraps
from typing import Any, Callable, Generic, Optional, TypeVar

from utils.logging_config import get_logger
from utils.metrics import record_error
from utils.tracing import get_trace_id, trace_span

logger = get_logger(__name__)

T = TypeVar("T")


class FallbackReason(Enum):
    """Reasons for using a fallback."""

    PRIMARY_FAILED = "primary_failed"
    PRIMARY_TIMEOUT = "primary_timeout"
    CIRCUIT_OPEN = "circuit_open"
    RATE_LIMITED = "rate_limited"
    SERVICE_UNAVAILABLE = "service_unavailable"


class FallbackExhaustedError(Exception):
    """Raised when all fallback strategies have been exhausted."""

    def __init__(
        self,
        message: str = "All fallback strategies exhausted",
        last_error: Optional[Exception] = None,
        attempts: int = 0,
    ):
        super().__init__(message)
        self.last_error = last_error
        self.attempts = attempts


@dataclass
class FallbackResult(Generic[T]):
    """Result from a fallback chain execution."""

    value: T
    strategy_name: str
    is_degraded: bool
    degradation_message: Optional[str] = None
    execution_time: float = 0.0
    fallback_reason: Optional[FallbackReason] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "is_degraded": self.is_degraded,
            "degradation_message": self.degradation_message,
            "execution_time": self.execution_time,
            "fallback_reason": (
                self.fallback_reason.value if self.fallback_reason else None
            ),
        }


@dataclass
class FallbackStats:
    """Statistics for fallback usage."""

    chain_name: str
    total_calls: int = 0
    primary_successes: int = 0
    fallback_uses: int = 0
    exhausted_count: int = 0
    strategy_uses: dict[str, int] = field(default_factory=dict)
    last_fallback_time: Optional[datetime] = None
    last_exhausted_time: Optional[datetime] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "chain_name": self.chain_name,
            "total_calls": self.total_calls,
            "primary_successes": self.primary_successes,
            "fallback_uses": self.fallback_uses,
            "exhausted_count": self.exhausted_count,
            "fallback_rate": (
                self.fallback_uses / self.total_calls if self.total_calls > 0 else 0
            ),
            "strategy_uses": self.strategy_uses,
            "last_fallback_time": (
                self.last_fallback_time.isoformat() if self.last_fallback_time else None
            ),
            "last_exhausted_time": (
                self.last_exhausted_time.isoformat()
                if self.last_exhausted_time
                else None
            ),
        }


class FallbackStrategy(ABC, Generic[T]):
    """Base class for fallback strategies."""

    def __init__(self, name: str, is_degraded: bool = False):
        """Initialize strategy.

        Args:
            name: Strategy identifier.
            is_degraded: Whether this strategy provides degraded functionality.
        """
        self.name = name
        self.is_degraded = is_degraded
        self._degradation_message: Optional[str] = None

    @property
    def degradation_message(self) -> Optional[str]:
        """Message explaining degraded functionality."""
        return self._degradation_message

    @abstractmethod
    def execute(self, *args: Any, **kwargs: Any) -> T:
        """Execute the strategy.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Strategy result.

        Raises:
            Exception: If strategy fails.
        """
        pass

    def can_handle(self, *args: Any, **kwargs: Any) -> bool:
        """Check if this strategy can handle the request.

        Override for conditional strategies.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            True if strategy can handle request.
        """
        return True


class FallbackChain(Generic[T]):
    """Chain of fallback strategies.

    Executes strategies in order until one succeeds.
    Tracks statistics and emits metrics.
    """

    def __init__(
        self,
        name: str,
        strategies: list[FallbackStrategy[T]],
        on_fallback: Optional[Callable[[str, str, Exception], None]] = None,
        on_exhausted: Optional[Callable[[Exception], None]] = None,
    ):
        """Initialize fallback chain.

        Args:
            name: Chain identifier.
            strategies: Ordered list of fallback strategies.
            on_fallback: Callback when falling back (from_strategy, to_strategy, error).
            on_exhausted: Callback when all strategies exhausted.
        """
        if not strategies:
            raise ValueError("At least one strategy is required")

        self.name = name
        self.strategies = strategies
        self.on_fallback = on_fallback
        self.on_exhausted = on_exhausted

        self._stats = FallbackStats(chain_name=name)
        self._lock = threading.Lock()

    def execute(self, *args: Any, **kwargs: Any) -> FallbackResult[T]:
        """Execute the fallback chain.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            FallbackResult with value and metadata.

        Raises:
            FallbackExhaustedError: If all strategies fail.
        """
        with trace_span(
            f"fallback.{self.name}",
            attributes={
                "fallback.chain": self.name,
                "fallback.strategy_count": len(self.strategies),
            },
        ) as span:
            start_time = time.time()
            last_error: Optional[Exception] = None
            fallback_reason: Optional[FallbackReason] = None
            attempts = 0

            with self._lock:
                self._stats.total_calls += 1

            for i, strategy in enumerate(self.strategies):
                # Check if strategy can handle this request
                if not strategy.can_handle(*args, **kwargs):
                    logger.debug(
                        f"Strategy '{strategy.name}' cannot handle request, skipping"
                    )
                    continue

                attempts += 1

                try:
                    with trace_span(
                        f"fallback.strategy.{strategy.name}",
                        attributes={
                            "strategy.name": strategy.name,
                            "strategy.index": i,
                            "strategy.is_degraded": strategy.is_degraded,
                        },
                    ):
                        result = strategy.execute(*args, **kwargs)

                        execution_time = time.time() - start_time

                        # Track statistics
                        with self._lock:
                            if i == 0:
                                self._stats.primary_successes += 1
                            else:
                                self._stats.fallback_uses += 1
                                self._stats.last_fallback_time = datetime.now(
                                    timezone.utc
                                )

                            self._stats.strategy_uses[strategy.name] = (
                                self._stats.strategy_uses.get(strategy.name, 0) + 1
                            )

                        # Log if using fallback
                        if i > 0:
                            logger.warning(
                                f"Fallback chain '{self.name}' using fallback strategy",
                                extra={
                                    "chain": self.name,
                                    "strategy": strategy.name,
                                    "strategy_index": i,
                                    "is_degraded": strategy.is_degraded,
                                    "reason": (
                                        fallback_reason.value if fallback_reason else None
                                    ),
                                    "trace_id": get_trace_id(),
                                },
                            )

                        span.set_attribute("fallback.success", True)
                        span.set_attribute("fallback.strategy_used", strategy.name)
                        span.set_attribute("fallback.is_degraded", strategy.is_degraded)

                        return FallbackResult(
                            value=result,
                            strategy_name=strategy.name,
                            is_degraded=strategy.is_degraded,
                            degradation_message=strategy.degradation_message,
                            execution_time=execution_time,
                            fallback_reason=fallback_reason if i > 0 else None,
                        )

                except Exception as e:
                    last_error = e
                    fallback_reason = self._classify_failure(e)

                    logger.debug(
                        f"Strategy '{strategy.name}' failed: {e}",
                        extra={
                            "chain": self.name,
                            "strategy": strategy.name,
                            "error": str(e),
                            "trace_id": get_trace_id(),
                        },
                    )

                    # Call fallback callback
                    if self.on_fallback and i < len(self.strategies) - 1:
                        try:
                            next_strategy = self.strategies[i + 1].name
                            self.on_fallback(strategy.name, next_strategy, e)
                        except Exception as callback_error:
                            logger.error(f"Fallback callback failed: {callback_error}")

            # All strategies exhausted
            execution_time = time.time() - start_time

            with self._lock:
                self._stats.exhausted_count += 1
                self._stats.last_exhausted_time = datetime.now(timezone.utc)

            logger.error(
                f"Fallback chain '{self.name}' exhausted all strategies",
                extra={
                    "chain": self.name,
                    "attempts": attempts,
                    "last_error": str(last_error),
                    "execution_time": execution_time,
                    "trace_id": get_trace_id(),
                },
            )

            record_error(
                agent=f"fallback_{self.name}",
                error_type="fallback_exhausted",
                error_message=f"All {attempts} fallback strategies exhausted",
            )

            span.set_attribute("fallback.success", False)
            span.set_attribute("fallback.exhausted", True)

            # Call exhausted callback
            if self.on_exhausted and last_error:
                try:
                    self.on_exhausted(last_error)
                except Exception as callback_error:
                    logger.error(f"Exhausted callback failed: {callback_error}")

            raise FallbackExhaustedError(
                message=f"Fallback chain '{self.name}' exhausted all {attempts} strategies",
                last_error=last_error,
                attempts=attempts,
            )

    def _classify_failure(self, error: Exception) -> FallbackReason:
        """Classify the failure reason."""
        error_str = str(error).lower()

        if "timeout" in error_str:
            return FallbackReason.PRIMARY_TIMEOUT
        elif "circuit" in error_str or "open" in error_str:
            return FallbackReason.CIRCUIT_OPEN
        elif "rate" in error_str or "429" in error_str:
            return FallbackReason.RATE_LIMITED
        elif "unavailable" in error_str or "503" in error_str:
            return FallbackReason.SERVICE_UNAVAILABLE
        else:
            return FallbackReason.PRIMARY_FAILED

    def get_stats(self) -> FallbackStats:
        """Get current statistics."""
        with self._lock:
            return FallbackStats(
                chain_name=self._stats.chain_name,
                total_calls=self._stats.total_calls,
                primary_successes=self._stats.primary_successes,
                fallback_uses=self._stats.fallback_uses,
                exhausted_count=self._stats.exhausted_count,
                strategy_uses=dict(self._stats.strategy_uses),
                last_fallback_time=self._stats.last_fallback_time,
                last_exhausted_time=self._stats.last_exhausted_time,
            )

    def reset_stats(self) -> None:
        """Reset statistics."""
        with self._lock:
            self._stats = FallbackStats(chain_name=self.name)


# ============================================================================
# Common Fallback Strategies
# ============================================================================


class CachedFallbackStrategy(FallbackStrategy[T]):
    """Fallback strategy that uses cached results."""

    def __init__(
        self,
        name: str = "cached",
        cache: Optional[dict[str, Any]] = None,
        cache_key_fn: Optional[Callable[..., str]] = None,
        ttl: float = 3600.0,  # 1 hour default
    ):
        super().__init__(name=name, is_degraded=True)
        self._degradation_message = "Using cached results (may be stale)"
        self._cache = cache if cache is not None else {}
        self._cache_times: dict[str, float] = {}
        self._cache_key_fn = cache_key_fn or self._default_cache_key
        self._ttl = ttl
        self._lock = threading.Lock()

    def _default_cache_key(self, *args: Any, **kwargs: Any) -> str:
        """Generate a cache key from arguments."""
        key_parts = [str(arg) for arg in args]
        key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
        key_str = ":".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()

    def cache_result(self, key: str, value: T) -> None:
        """Store a result in the cache."""
        with self._lock:
            self._cache[key] = value
            self._cache_times[key] = time.time()

    def execute(self, *args: Any, **kwargs: Any) -> T:
        """Return cached result if available."""
        key = self._cache_key_fn(*args, **kwargs)

        with self._lock:
            if key not in self._cache:
                raise KeyError(f"No cached result for key: {key}")

            # Check TTL
            cache_time = self._cache_times.get(key, 0)
            if time.time() - cache_time > self._ttl:
                del self._cache[key]
                del self._cache_times[key]
                raise KeyError(f"Cached result expired for key: {key}")

            return self._cache[key]

    def can_handle(self, *args: Any, **kwargs: Any) -> bool:
        """Check if we have a cached result."""
        key = self._cache_key_fn(*args, **kwargs)
        with self._lock:
            if key not in self._cache:
                return False
            cache_time = self._cache_times.get(key, 0)
            return time.time() - cache_time <= self._ttl


class EmptyFallbackStrategy(FallbackStrategy[T]):
    """Fallback strategy that returns an empty/default result."""

    def __init__(
        self,
        name: str = "empty",
        default_value: T = None,  # type: ignore
        message: str = "Service unavailable, returning empty result",
    ):
        super().__init__(name=name, is_degraded=True)
        self._default_value = default_value
        self._degradation_message = message

    def execute(self, *args: Any, **kwargs: Any) -> T:
        """Return the default/empty value."""
        return self._default_value


class LambdaFallbackStrategy(FallbackStrategy[T]):
    """Fallback strategy using a lambda/function."""

    def __init__(
        self,
        name: str,
        func: Callable[..., T],
        is_degraded: bool = False,
        degradation_message: Optional[str] = None,
        can_handle_fn: Optional[Callable[..., bool]] = None,
    ):
        super().__init__(name=name, is_degraded=is_degraded)
        self._func = func
        self._degradation_message = degradation_message
        self._can_handle_fn = can_handle_fn

    def execute(self, *args: Any, **kwargs: Any) -> T:
        """Execute the wrapped function."""
        return self._func(*args, **kwargs)

    def can_handle(self, *args: Any, **kwargs: Any) -> bool:
        if self._can_handle_fn:
            return self._can_handle_fn(*args, **kwargs)
        return True


# ============================================================================
# Model Fallback Chain
# ============================================================================


@dataclass
class ModelConfig:
    """Configuration for a model in the fallback chain."""

    name: str
    model_id: str
    max_tokens: int = 8192
    temperature: float = 0.7
    is_degraded: bool = False
    degradation_message: Optional[str] = None


# Default model fallback chain
DEFAULT_MODEL_CHAIN = [
    ModelConfig(
        name="primary",
        model_id="gemini-2.0-flash",
        max_tokens=8192,
        is_degraded=False,
    ),
    ModelConfig(
        name="fallback_1",
        model_id="gemini-1.5-flash",
        max_tokens=8192,
        is_degraded=True,
        degradation_message="Using fallback model (may have reduced capabilities)",
    ),
    ModelConfig(
        name="fallback_2",
        model_id="gemini-1.5-flash-8b",
        max_tokens=4096,
        is_degraded=True,
        degradation_message="Using reduced functionality mode",
    ),
]


class ModelFallbackChain:
    """Fallback chain specifically for model calls."""

    def __init__(
        self,
        models: Optional[list[ModelConfig]] = None,
        on_fallback: Optional[Callable[[str, str, Exception], None]] = None,
    ):
        """Initialize model fallback chain.

        Args:
            models: List of model configurations in order of preference.
            on_fallback: Callback when falling back between models.
        """
        self.models = models or DEFAULT_MODEL_CHAIN
        self.on_fallback = on_fallback

        self._stats = FallbackStats(chain_name="model_fallback")
        self._lock = threading.Lock()

    def call(
        self,
        call_fn: Callable[[ModelConfig], T],
        *args: Any,
        **kwargs: Any,
    ) -> FallbackResult[T]:
        """Call with model fallback.

        Args:
            call_fn: Function that takes ModelConfig and makes the call.
            *args: Additional arguments (passed to callbacks).
            **kwargs: Additional keyword arguments.

        Returns:
            FallbackResult with the response.

        Raises:
            FallbackExhaustedError: If all models fail.
        """
        with trace_span(
            "model_fallback",
            attributes={
                "fallback.model_count": len(self.models),
            },
        ) as span:
            start_time = time.time()
            last_error: Optional[Exception] = None

            with self._lock:
                self._stats.total_calls += 1

            for i, model in enumerate(self.models):
                try:
                    with trace_span(
                        f"model.{model.name}",
                        attributes={
                            "model.name": model.name,
                            "model.id": model.model_id,
                            "model.index": i,
                        },
                    ):
                        result = call_fn(model)

                        execution_time = time.time() - start_time

                        with self._lock:
                            if i == 0:
                                self._stats.primary_successes += 1
                            else:
                                self._stats.fallback_uses += 1
                                self._stats.last_fallback_time = datetime.now(
                                    timezone.utc
                                )

                            self._stats.strategy_uses[model.name] = (
                                self._stats.strategy_uses.get(model.name, 0) + 1
                            )

                        if i > 0:
                            logger.warning(
                                f"Model fallback: using {model.name}",
                                extra={
                                    "model": model.model_id,
                                    "model_index": i,
                                    "is_degraded": model.is_degraded,
                                    "trace_id": get_trace_id(),
                                },
                            )

                        span.set_attribute("model.used", model.model_id)
                        span.set_attribute("model.is_degraded", model.is_degraded)

                        return FallbackResult(
                            value=result,
                            strategy_name=model.name,
                            is_degraded=model.is_degraded,
                            degradation_message=model.degradation_message,
                            execution_time=execution_time,
                            fallback_reason=(
                                FallbackReason.PRIMARY_FAILED if i > 0 else None
                            ),
                        )

                except Exception as e:
                    last_error = e
                    logger.debug(
                        f"Model '{model.name}' failed: {e}",
                        extra={
                            "model": model.model_id,
                            "error": str(e),
                            "trace_id": get_trace_id(),
                        },
                    )

                    if self.on_fallback and i < len(self.models) - 1:
                        try:
                            next_model = self.models[i + 1].name
                            self.on_fallback(model.name, next_model, e)
                        except Exception as callback_error:
                            logger.error(f"Model fallback callback failed: {callback_error}")

            with self._lock:
                self._stats.exhausted_count += 1
                self._stats.last_exhausted_time = datetime.now(timezone.utc)

            logger.error(
                "Model fallback chain exhausted",
                extra={
                    "attempts": len(self.models),
                    "last_error": str(last_error),
                    "trace_id": get_trace_id(),
                },
            )

            span.set_attribute("fallback.exhausted", True)

            raise FallbackExhaustedError(
                message=f"All {len(self.models)} models failed",
                last_error=last_error,
                attempts=len(self.models),
            )

    def get_stats(self) -> FallbackStats:
        """Get current statistics."""
        with self._lock:
            return FallbackStats(
                chain_name=self._stats.chain_name,
                total_calls=self._stats.total_calls,
                primary_successes=self._stats.primary_successes,
                fallback_uses=self._stats.fallback_uses,
                exhausted_count=self._stats.exhausted_count,
                strategy_uses=dict(self._stats.strategy_uses),
                last_fallback_time=self._stats.last_fallback_time,
                last_exhausted_time=self._stats.last_exhausted_time,
            )


# ============================================================================
# Tool Fallback Factories
# ============================================================================


def create_search_fallback(
    primary_fn: Callable[..., Any],
    knowledge_base_fn: Optional[Callable[..., Any]] = None,
) -> FallbackChain:
    """Create a fallback chain for web search.

    Hierarchy:
    1. Primary search API
    2. Cached search results
    3. Knowledge base search (if available)
    4. Empty result with warning

    Args:
        primary_fn: Primary search function.
        knowledge_base_fn: Optional knowledge base search function.

    Returns:
        Configured FallbackChain.
    """
    strategies: list[FallbackStrategy] = [
        LambdaFallbackStrategy(
            name="primary_search",
            func=primary_fn,
            is_degraded=False,
        ),
        CachedFallbackStrategy(
            name="cached_search",
            ttl=3600.0,  # 1 hour cache
        ),
    ]

    if knowledge_base_fn:
        strategies.append(
            LambdaFallbackStrategy(
                name="knowledge_base",
                func=knowledge_base_fn,
                is_degraded=True,
                degradation_message="Using knowledge base instead of live search",
            )
        )

    strategies.append(
        EmptyFallbackStrategy(
            name="empty_search",
            default_value={"results": [], "query": ""},
            message="Search service unavailable",
        )
    )

    return FallbackChain(name="web_search", strategies=strategies)


def create_url_reader_fallback(
    primary_fn: Callable[..., Any],
    knowledge_base_fn: Optional[Callable[..., Any]] = None,
) -> FallbackChain:
    """Create a fallback chain for URL reading.

    Hierarchy:
    1. Primary URL reader
    2. Cached URL content
    3. Knowledge base summary (if available)
    4. Empty result with warning

    Args:
        primary_fn: Primary URL reader function.
        knowledge_base_fn: Optional function to get URL summary from KB.

    Returns:
        Configured FallbackChain.
    """
    strategies: list[FallbackStrategy] = [
        LambdaFallbackStrategy(
            name="primary_reader",
            func=primary_fn,
            is_degraded=False,
        ),
        CachedFallbackStrategy(
            name="cached_content",
            ttl=86400.0,  # 24 hour cache for URLs
        ),
    ]

    if knowledge_base_fn:
        strategies.append(
            LambdaFallbackStrategy(
                name="knowledge_base_summary",
                func=knowledge_base_fn,
                is_degraded=True,
                degradation_message="Using cached summary from knowledge base",
            )
        )

    strategies.append(
        EmptyFallbackStrategy(
            name="empty_content",
            default_value={"content": "", "url": "", "error": "Content unavailable"},
            message="URL content unavailable",
        )
    )

    return FallbackChain(name="url_reader", strategies=strategies)


def create_knowledge_search_fallback(
    primary_fn: Callable[..., Any],
) -> FallbackChain:
    """Create a fallback chain for knowledge base search.

    Hierarchy:
    1. Primary semantic search
    2. Keyword search fallback
    3. Empty result with warning

    Args:
        primary_fn: Primary knowledge search function.

    Returns:
        Configured FallbackChain.
    """

    def keyword_search(*args: Any, **kwargs: Any) -> Any:
        """Simple keyword search as fallback."""
        # This would be implemented by the caller
        raise NotImplementedError("Keyword search not configured")

    strategies: list[FallbackStrategy] = [
        LambdaFallbackStrategy(
            name="semantic_search",
            func=primary_fn,
            is_degraded=False,
        ),
        LambdaFallbackStrategy(
            name="keyword_search",
            func=keyword_search,
            is_degraded=True,
            degradation_message="Using keyword search (less accurate)",
        ),
        EmptyFallbackStrategy(
            name="empty_results",
            default_value={"results": [], "query": ""},
            message="Knowledge base search unavailable",
        ),
    ]

    return FallbackChain(name="knowledge_search", strategies=strategies)


# ============================================================================
# Global Fallback Registry
# ============================================================================

_fallback_chains: dict[str, FallbackChain] = {}
_model_fallback: Optional[ModelFallbackChain] = None
_registry_lock = threading.Lock()


def register_fallback_chain(name: str, chain: FallbackChain) -> None:
    """Register a fallback chain globally.

    Args:
        name: Chain identifier.
        chain: FallbackChain instance.
    """
    with _registry_lock:
        _fallback_chains[name] = chain


def get_fallback_chain(name: str) -> Optional[FallbackChain]:
    """Get a registered fallback chain.

    Args:
        name: Chain identifier.

    Returns:
        FallbackChain if registered, None otherwise.
    """
    with _registry_lock:
        return _fallback_chains.get(name)


def get_model_fallback() -> ModelFallbackChain:
    """Get the global model fallback chain."""
    global _model_fallback
    with _registry_lock:
        if _model_fallback is None:
            _model_fallback = ModelFallbackChain()
        return _model_fallback


def set_model_fallback(chain: ModelFallbackChain) -> None:
    """Set the global model fallback chain."""
    global _model_fallback
    with _registry_lock:
        _model_fallback = chain


def get_all_fallback_stats() -> dict[str, FallbackStats]:
    """Get statistics for all registered fallback chains."""
    with _registry_lock:
        stats = {name: chain.get_stats() for name, chain in _fallback_chains.items()}
        if _model_fallback:
            stats["model_fallback"] = _model_fallback.get_stats()
        return stats


def reset_fallback_stats() -> None:
    """Reset statistics for all fallback chains."""
    with _registry_lock:
        for chain in _fallback_chains.values():
            chain.reset_stats()
        if _model_fallback:
            _model_fallback._stats = FallbackStats(chain_name="model_fallback")


def clear_fallback_registry() -> None:
    """Clear all registered fallback chains."""
    global _model_fallback
    with _registry_lock:
        _fallback_chains.clear()
        _model_fallback = None


# ============================================================================
# Decorator
# ============================================================================


def with_fallback(
    chain_name: str,
    chain: Optional[FallbackChain] = None,
) -> Callable:
    """Decorator to add fallback behavior to a function.

    The decorated function becomes the primary strategy in the chain.

    Args:
        chain_name: Name for the fallback chain.
        chain: Optional existing chain (function becomes first strategy).

    Returns:
        Decorator function.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., FallbackResult[T]]:
        # Create a new chain with the function as primary
        primary = LambdaFallbackStrategy(
            name="primary",
            func=func,
            is_degraded=False,
        )

        if chain:
            # Prepend function to existing chain
            strategies = [primary] + chain.strategies
            fallback_chain = FallbackChain(
                name=chain_name,
                strategies=strategies,
                on_fallback=chain.on_fallback,
                on_exhausted=chain.on_exhausted,
            )
        else:
            # Just wrap function with empty fallback
            empty = EmptyFallbackStrategy(
                name="empty",
                default_value=None,
                message="Service unavailable",
            )
            fallback_chain = FallbackChain(
                name=chain_name,
                strategies=[primary, empty],
            )

        # Register the chain
        register_fallback_chain(chain_name, fallback_chain)

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> FallbackResult[T]:
            return fallback_chain.execute(*args, **kwargs)

        return wrapper

    return decorator
