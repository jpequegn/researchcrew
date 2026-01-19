"""Tests for Fallback Strategies

Tests the fallback chain and strategy implementations for ResearchCrew.
"""

import time
import pytest
from unittest.mock import Mock, patch


class TestFallbackChain:
    """Tests for FallbackChain class."""

    def setup_method(self):
        """Reset state before each test."""
        from utils.fallback import clear_fallback_registry
        from utils import reset_tracing, init_tracing, reset_metrics, init_metrics

        clear_fallback_registry()
        reset_tracing()
        reset_metrics()
        init_tracing(exporter_type="none")
        init_metrics()

    def teardown_method(self):
        """Reset state after each test."""
        from utils.fallback import clear_fallback_registry
        from utils import reset_tracing, reset_metrics

        clear_fallback_registry()
        reset_tracing()
        reset_metrics()

    def test_primary_success(self):
        """Test that primary strategy is used when successful."""
        from utils.fallback import (
            FallbackChain,
            LambdaFallbackStrategy,
        )

        primary = LambdaFallbackStrategy(
            name="primary",
            func=lambda: "primary_result",
            is_degraded=False,
        )
        fallback = LambdaFallbackStrategy(
            name="fallback",
            func=lambda: "fallback_result",
            is_degraded=True,
        )

        chain = FallbackChain(name="test", strategies=[primary, fallback])
        result = chain.execute()

        assert result.value == "primary_result"
        assert result.strategy_name == "primary"
        assert result.is_degraded is False
        assert result.fallback_reason is None

    def test_fallback_on_primary_failure(self):
        """Test that fallback is used when primary fails."""
        from utils.fallback import (
            FallbackChain,
            LambdaFallbackStrategy,
            FallbackReason,
        )

        def failing_primary():
            raise ValueError("Primary failed")

        primary = LambdaFallbackStrategy(
            name="primary",
            func=failing_primary,
            is_degraded=False,
        )
        fallback = LambdaFallbackStrategy(
            name="fallback",
            func=lambda: "fallback_result",
            is_degraded=True,
            degradation_message="Using fallback",
        )

        chain = FallbackChain(name="test", strategies=[primary, fallback])
        result = chain.execute()

        assert result.value == "fallback_result"
        assert result.strategy_name == "fallback"
        assert result.is_degraded is True
        assert result.degradation_message == "Using fallback"
        assert result.fallback_reason == FallbackReason.PRIMARY_FAILED

    def test_exhausted_error(self):
        """Test that FallbackExhaustedError is raised when all strategies fail."""
        from utils.fallback import (
            FallbackChain,
            LambdaFallbackStrategy,
            FallbackExhaustedError,
        )

        def always_fails():
            raise ValueError("Always fails")

        primary = LambdaFallbackStrategy(
            name="primary",
            func=always_fails,
            is_degraded=False,
        )
        fallback = LambdaFallbackStrategy(
            name="fallback",
            func=always_fails,
            is_degraded=True,
        )

        chain = FallbackChain(name="test", strategies=[primary, fallback])

        with pytest.raises(FallbackExhaustedError) as exc_info:
            chain.execute()

        assert exc_info.value.attempts == 2
        assert exc_info.value.last_error is not None

    def test_stats_tracking(self):
        """Test that statistics are tracked correctly."""
        from utils.fallback import (
            FallbackChain,
            LambdaFallbackStrategy,
        )

        call_count = 0

        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("First call fails")
            return "success"

        primary = LambdaFallbackStrategy(
            name="primary",
            func=flaky,
            is_degraded=False,
        )
        fallback = LambdaFallbackStrategy(
            name="fallback",
            func=lambda: "fallback_result",
            is_degraded=True,
        )

        chain = FallbackChain(name="test", strategies=[primary, fallback])

        # First call - primary fails, fallback succeeds
        result1 = chain.execute()
        assert result1.strategy_name == "fallback"

        # Second call - primary succeeds
        result2 = chain.execute()
        assert result2.strategy_name == "primary"

        stats = chain.get_stats()
        assert stats.total_calls == 2
        assert stats.primary_successes == 1
        assert stats.fallback_uses == 1
        assert stats.strategy_uses["primary"] == 1
        assert stats.strategy_uses["fallback"] == 1

    def test_on_fallback_callback(self):
        """Test that on_fallback callback is called."""
        from utils.fallback import (
            FallbackChain,
            LambdaFallbackStrategy,
        )

        callback_called = False
        callback_args = {}

        def on_fallback(from_strategy, to_strategy, error):
            nonlocal callback_called, callback_args
            callback_called = True
            callback_args = {
                "from": from_strategy,
                "to": to_strategy,
                "error": error,
            }

        primary = LambdaFallbackStrategy(
            name="primary",
            func=lambda: (_ for _ in ()).throw(ValueError("Fail")),
            is_degraded=False,
        )
        fallback = LambdaFallbackStrategy(
            name="fallback",
            func=lambda: "result",
            is_degraded=True,
        )

        chain = FallbackChain(
            name="test",
            strategies=[primary, fallback],
            on_fallback=on_fallback,
        )
        chain.execute()

        assert callback_called
        assert callback_args["from"] == "primary"
        assert callback_args["to"] == "fallback"

    def test_on_exhausted_callback(self):
        """Test that on_exhausted callback is called."""
        from utils.fallback import (
            FallbackChain,
            LambdaFallbackStrategy,
            FallbackExhaustedError,
        )

        callback_called = False

        def on_exhausted(error):
            nonlocal callback_called
            callback_called = True

        def always_fails():
            raise ValueError("Fails")

        primary = LambdaFallbackStrategy(
            name="primary",
            func=always_fails,
            is_degraded=False,
        )

        chain = FallbackChain(
            name="test",
            strategies=[primary],
            on_exhausted=on_exhausted,
        )

        with pytest.raises(FallbackExhaustedError):
            chain.execute()

        assert callback_called

    def test_requires_at_least_one_strategy(self):
        """Test that chain requires at least one strategy."""
        from utils.fallback import FallbackChain

        with pytest.raises(ValueError):
            FallbackChain(name="test", strategies=[])

    def test_args_passed_to_strategy(self):
        """Test that arguments are passed to strategies."""
        from utils.fallback import (
            FallbackChain,
            LambdaFallbackStrategy,
        )

        def strategy_func(a, b, c=None):
            return f"{a}-{b}-{c}"

        primary = LambdaFallbackStrategy(
            name="primary",
            func=strategy_func,
            is_degraded=False,
        )

        chain = FallbackChain(name="test", strategies=[primary])
        result = chain.execute("x", "y", c="z")

        assert result.value == "x-y-z"


class TestCachedFallbackStrategy:
    """Tests for CachedFallbackStrategy."""

    def test_returns_cached_value(self):
        """Test that cached value is returned."""
        from utils.fallback import CachedFallbackStrategy

        strategy = CachedFallbackStrategy(name="cache")
        strategy.cache_result("key1", "cached_value")

        # Manually set up the cache key function to return "key1"
        strategy._cache_key_fn = lambda *args, **kwargs: "key1"

        result = strategy.execute()
        assert result == "cached_value"

    def test_raises_on_cache_miss(self):
        """Test that KeyError is raised on cache miss."""
        from utils.fallback import CachedFallbackStrategy

        strategy = CachedFallbackStrategy(name="cache")

        with pytest.raises(KeyError):
            strategy.execute("nonexistent")

    def test_cache_expiration(self):
        """Test that cached values expire."""
        from utils.fallback import CachedFallbackStrategy

        strategy = CachedFallbackStrategy(name="cache", ttl=0.1)
        strategy.cache_result("key1", "cached_value")
        strategy._cache_key_fn = lambda *args, **kwargs: "key1"

        # Should work immediately
        assert strategy.can_handle() is True

        # Wait for expiration
        time.sleep(0.15)

        # Should be expired
        assert strategy.can_handle() is False

    def test_can_handle_returns_false_on_miss(self):
        """Test can_handle returns False when not cached."""
        from utils.fallback import CachedFallbackStrategy

        strategy = CachedFallbackStrategy(name="cache")

        assert strategy.can_handle("missing_key") is False

    def test_custom_cache_key_function(self):
        """Test custom cache key function."""
        from utils.fallback import CachedFallbackStrategy

        def custom_key(*args, **kwargs):
            return f"custom_{args[0]}"

        strategy = CachedFallbackStrategy(name="cache", cache_key_fn=custom_key)
        strategy.cache_result("custom_test", "value")

        result = strategy.execute("test")
        assert result == "value"


class TestEmptyFallbackStrategy:
    """Tests for EmptyFallbackStrategy."""

    def test_returns_default_value(self):
        """Test that default value is returned."""
        from utils.fallback import EmptyFallbackStrategy

        strategy = EmptyFallbackStrategy(
            name="empty",
            default_value={"results": []},
            message="Service unavailable",
        )

        result = strategy.execute()
        assert result == {"results": []}

    def test_is_degraded(self):
        """Test that strategy is marked as degraded."""
        from utils.fallback import EmptyFallbackStrategy

        strategy = EmptyFallbackStrategy(name="empty")
        assert strategy.is_degraded is True

    def test_degradation_message(self):
        """Test degradation message."""
        from utils.fallback import EmptyFallbackStrategy

        strategy = EmptyFallbackStrategy(
            name="empty",
            message="Custom message",
        )
        assert strategy.degradation_message == "Custom message"


class TestLambdaFallbackStrategy:
    """Tests for LambdaFallbackStrategy."""

    def test_executes_function(self):
        """Test that function is executed."""
        from utils.fallback import LambdaFallbackStrategy

        strategy = LambdaFallbackStrategy(
            name="lambda",
            func=lambda x, y: x + y,
        )

        result = strategy.execute(2, 3)
        assert result == 5

    def test_custom_can_handle(self):
        """Test custom can_handle function."""
        from utils.fallback import LambdaFallbackStrategy

        strategy = LambdaFallbackStrategy(
            name="lambda",
            func=lambda x: x * 2,
            can_handle_fn=lambda x: x > 0,
        )

        assert strategy.can_handle(5) is True
        assert strategy.can_handle(-1) is False

    def test_degradation_settings(self):
        """Test degradation settings."""
        from utils.fallback import LambdaFallbackStrategy

        strategy = LambdaFallbackStrategy(
            name="lambda",
            func=lambda: "result",
            is_degraded=True,
            degradation_message="Degraded mode",
        )

        assert strategy.is_degraded is True
        assert strategy.degradation_message == "Degraded mode"


class TestModelFallbackChain:
    """Tests for ModelFallbackChain."""

    def setup_method(self):
        """Reset state before each test."""
        from utils.fallback import clear_fallback_registry
        from utils import reset_tracing, init_tracing, reset_metrics, init_metrics

        clear_fallback_registry()
        reset_tracing()
        reset_metrics()
        init_tracing(exporter_type="none")
        init_metrics()

    def teardown_method(self):
        """Reset state after each test."""
        from utils.fallback import clear_fallback_registry
        from utils import reset_tracing, reset_metrics

        clear_fallback_registry()
        reset_tracing()
        reset_metrics()

    def test_primary_model_success(self):
        """Test that primary model is used when successful."""
        from utils.fallback import ModelFallbackChain, ModelConfig

        models = [
            ModelConfig(name="primary", model_id="model-1", is_degraded=False),
            ModelConfig(name="fallback", model_id="model-2", is_degraded=True),
        ]

        chain = ModelFallbackChain(models=models)

        def call_fn(model):
            return f"result_from_{model.model_id}"

        result = chain.call(call_fn)

        assert result.value == "result_from_model-1"
        assert result.strategy_name == "primary"
        assert result.is_degraded is False

    def test_fallback_model_on_failure(self):
        """Test that fallback model is used when primary fails."""
        from utils.fallback import ModelFallbackChain, ModelConfig

        models = [
            ModelConfig(name="primary", model_id="model-1", is_degraded=False),
            ModelConfig(
                name="fallback",
                model_id="model-2",
                is_degraded=True,
                degradation_message="Using fallback",
            ),
        ]

        chain = ModelFallbackChain(models=models)

        call_count = 0

        def call_fn(model):
            nonlocal call_count
            call_count += 1
            if model.model_id == "model-1":
                raise ValueError("Primary failed")
            return f"result_from_{model.model_id}"

        result = chain.call(call_fn)

        assert result.value == "result_from_model-2"
        assert result.strategy_name == "fallback"
        assert result.is_degraded is True
        assert result.degradation_message == "Using fallback"
        assert call_count == 2

    def test_all_models_fail(self):
        """Test FallbackExhaustedError when all models fail."""
        from utils.fallback import (
            ModelFallbackChain,
            ModelConfig,
            FallbackExhaustedError,
        )

        models = [
            ModelConfig(name="primary", model_id="model-1"),
            ModelConfig(name="fallback", model_id="model-2"),
        ]

        chain = ModelFallbackChain(models=models)

        def call_fn(model):
            raise ValueError(f"Model {model.model_id} failed")

        with pytest.raises(FallbackExhaustedError) as exc_info:
            chain.call(call_fn)

        assert exc_info.value.attempts == 2

    def test_stats_tracking(self):
        """Test that model stats are tracked."""
        from utils.fallback import ModelFallbackChain, ModelConfig

        models = [
            ModelConfig(name="primary", model_id="model-1"),
            ModelConfig(name="fallback", model_id="model-2"),
        ]

        chain = ModelFallbackChain(models=models)

        call_count = 0

        def call_fn(model):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("First fails")
            return "success"

        # First call - primary fails, fallback succeeds
        chain.call(call_fn)

        # Second call - primary succeeds
        chain.call(call_fn)

        stats = chain.get_stats()
        assert stats.total_calls == 2
        assert stats.primary_successes == 1
        assert stats.fallback_uses == 1


class TestFallbackRegistry:
    """Tests for global fallback registry."""

    def setup_method(self):
        """Reset state before each test."""
        from utils.fallback import clear_fallback_registry
        from utils import reset_tracing, init_tracing, reset_metrics, init_metrics

        clear_fallback_registry()
        reset_tracing()
        reset_metrics()
        init_tracing(exporter_type="none")
        init_metrics()

    def teardown_method(self):
        """Reset state after each test."""
        from utils.fallback import clear_fallback_registry
        from utils import reset_tracing, reset_metrics

        clear_fallback_registry()
        reset_tracing()
        reset_metrics()

    def test_register_and_get_chain(self):
        """Test registering and getting a fallback chain."""
        from utils.fallback import (
            FallbackChain,
            LambdaFallbackStrategy,
            register_fallback_chain,
            get_fallback_chain,
        )

        primary = LambdaFallbackStrategy(
            name="primary",
            func=lambda: "result",
        )
        chain = FallbackChain(name="test_chain", strategies=[primary])

        register_fallback_chain("test_chain", chain)

        retrieved = get_fallback_chain("test_chain")
        assert retrieved is chain

    def test_get_nonexistent_chain(self):
        """Test getting nonexistent chain returns None."""
        from utils.fallback import get_fallback_chain

        assert get_fallback_chain("nonexistent") is None

    def test_get_all_stats(self):
        """Test getting all fallback stats."""
        from utils.fallback import (
            FallbackChain,
            LambdaFallbackStrategy,
            register_fallback_chain,
            get_all_fallback_stats,
        )

        primary = LambdaFallbackStrategy(name="primary", func=lambda: "result")
        chain1 = FallbackChain(name="chain1", strategies=[primary])
        chain2 = FallbackChain(name="chain2", strategies=[primary])

        register_fallback_chain("chain1", chain1)
        register_fallback_chain("chain2", chain2)

        chain1.execute()
        chain2.execute()

        stats = get_all_fallback_stats()
        assert "chain1" in stats
        assert "chain2" in stats
        assert stats["chain1"].total_calls == 1
        assert stats["chain2"].total_calls == 1

    def test_reset_stats(self):
        """Test resetting fallback stats."""
        from utils.fallback import (
            FallbackChain,
            LambdaFallbackStrategy,
            register_fallback_chain,
            reset_fallback_stats,
        )

        primary = LambdaFallbackStrategy(name="primary", func=lambda: "result")
        chain = FallbackChain(name="chain", strategies=[primary])
        register_fallback_chain("chain", chain)

        chain.execute()
        assert chain.get_stats().total_calls == 1

        reset_fallback_stats()
        assert chain.get_stats().total_calls == 0

    def test_clear_registry(self):
        """Test clearing the fallback registry."""
        from utils.fallback import (
            FallbackChain,
            LambdaFallbackStrategy,
            register_fallback_chain,
            get_fallback_chain,
            clear_fallback_registry,
        )

        primary = LambdaFallbackStrategy(name="primary", func=lambda: "result")
        chain = FallbackChain(name="chain", strategies=[primary])
        register_fallback_chain("chain", chain)

        assert get_fallback_chain("chain") is not None

        clear_fallback_registry()
        assert get_fallback_chain("chain") is None


class TestWithFallbackDecorator:
    """Tests for with_fallback decorator."""

    def setup_method(self):
        """Reset state before each test."""
        from utils.fallback import clear_fallback_registry
        from utils import reset_tracing, init_tracing, reset_metrics, init_metrics

        clear_fallback_registry()
        reset_tracing()
        reset_metrics()
        init_tracing(exporter_type="none")
        init_metrics()

    def teardown_method(self):
        """Reset state after each test."""
        from utils.fallback import clear_fallback_registry
        from utils import reset_tracing, reset_metrics

        clear_fallback_registry()
        reset_tracing()
        reset_metrics()

    def test_decorator_wraps_function(self):
        """Test that decorator wraps function."""
        from utils.fallback import with_fallback, get_fallback_chain

        @with_fallback("test_func")
        def my_function(x):
            return x * 2

        result = my_function(5)

        assert result.value == 10
        assert result.strategy_name == "primary"
        assert result.is_degraded is False

        # Chain should be registered
        assert get_fallback_chain("test_func") is not None

    def test_decorator_uses_fallback_on_failure(self):
        """Test that decorator falls back on failure."""
        from utils.fallback import with_fallback, EmptyFallbackStrategy, FallbackChain

        fallback_chain = FallbackChain(
            name="existing",
            strategies=[
                EmptyFallbackStrategy(
                    name="empty",
                    default_value="fallback_value",
                )
            ],
        )

        call_count = 0

        @with_fallback("test_func", chain=fallback_chain)
        def failing_function():
            nonlocal call_count
            call_count += 1
            raise ValueError("Always fails")

        result = failing_function()

        assert result.value == "fallback_value"
        assert result.is_degraded is True
        assert call_count == 1  # Primary was attempted


class TestFallbackFactories:
    """Tests for fallback factory functions."""

    def setup_method(self):
        """Reset state before each test."""
        from utils.fallback import clear_fallback_registry
        from utils import reset_tracing, init_tracing, reset_metrics, init_metrics

        clear_fallback_registry()
        reset_tracing()
        reset_metrics()
        init_tracing(exporter_type="none")
        init_metrics()

    def teardown_method(self):
        """Reset state after each test."""
        from utils.fallback import clear_fallback_registry
        from utils import reset_tracing, reset_metrics

        clear_fallback_registry()
        reset_tracing()
        reset_metrics()

    def test_create_search_fallback(self):
        """Test creating search fallback chain."""
        from utils.fallback import create_search_fallback

        def primary_search(query):
            return {"results": [f"Result for: {query}"], "query": query}

        chain = create_search_fallback(primary_fn=primary_search)

        result = chain.execute(query="test")
        assert result.value["query"] == "test"
        assert len(result.value["results"]) == 1

    def test_create_search_fallback_with_kb(self):
        """Test search fallback with knowledge base."""
        from utils.fallback import create_search_fallback

        def failing_primary(query):
            raise ValueError("Search failed")

        def kb_search(query):
            return {"results": [f"KB result for: {query}"], "query": query}

        chain = create_search_fallback(
            primary_fn=failing_primary,
            knowledge_base_fn=kb_search,
        )

        result = chain.execute(query="test")
        assert result.strategy_name == "knowledge_base"
        assert result.is_degraded is True

    def test_create_url_reader_fallback(self):
        """Test creating URL reader fallback chain."""
        from utils.fallback import create_url_reader_fallback

        def primary_reader(url):
            return {"content": f"Content from {url}", "url": url}

        chain = create_url_reader_fallback(primary_fn=primary_reader)

        result = chain.execute(url="https://example.com")
        assert result.value["url"] == "https://example.com"

    def test_create_knowledge_search_fallback(self):
        """Test creating knowledge search fallback chain."""
        from utils.fallback import create_knowledge_search_fallback

        def primary_search(query):
            return {"results": [query], "query": query}

        chain = create_knowledge_search_fallback(primary_fn=primary_search)

        result = chain.execute(query="test")
        assert result.value["query"] == "test"


class TestFallbackResult:
    """Tests for FallbackResult dataclass."""

    def test_to_dict(self):
        """Test FallbackResult serialization."""
        from utils.fallback import FallbackResult, FallbackReason

        result = FallbackResult(
            value="test",
            strategy_name="primary",
            is_degraded=False,
            degradation_message=None,
            execution_time=0.5,
            fallback_reason=None,
        )

        d = result.to_dict()
        assert d["strategy_name"] == "primary"
        assert d["is_degraded"] is False
        assert d["execution_time"] == 0.5
        assert d["fallback_reason"] is None

    def test_to_dict_with_reason(self):
        """Test FallbackResult serialization with reason."""
        from utils.fallback import FallbackResult, FallbackReason

        result = FallbackResult(
            value="test",
            strategy_name="fallback",
            is_degraded=True,
            degradation_message="Using fallback",
            execution_time=0.5,
            fallback_reason=FallbackReason.PRIMARY_TIMEOUT,
        )

        d = result.to_dict()
        assert d["fallback_reason"] == "primary_timeout"
        assert d["degradation_message"] == "Using fallback"


class TestFallbackStats:
    """Tests for FallbackStats dataclass."""

    def test_to_dict(self):
        """Test FallbackStats serialization."""
        from utils.fallback import FallbackStats
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        stats = FallbackStats(
            chain_name="test",
            total_calls=10,
            primary_successes=8,
            fallback_uses=2,
            exhausted_count=0,
            strategy_uses={"primary": 8, "fallback": 2},
            last_fallback_time=now,
            last_exhausted_time=None,
        )

        d = stats.to_dict()
        assert d["chain_name"] == "test"
        assert d["total_calls"] == 10
        assert d["fallback_rate"] == 0.2
        assert d["strategy_uses"]["primary"] == 8


class TestFallbackReason:
    """Tests for FallbackReason classification."""

    def setup_method(self):
        """Reset state before each test."""
        from utils.fallback import clear_fallback_registry
        from utils import reset_tracing, init_tracing, reset_metrics, init_metrics

        clear_fallback_registry()
        reset_tracing()
        reset_metrics()
        init_tracing(exporter_type="none")
        init_metrics()

    def teardown_method(self):
        """Reset state after each test."""
        from utils.fallback import clear_fallback_registry
        from utils import reset_tracing, reset_metrics

        clear_fallback_registry()
        reset_tracing()
        reset_metrics()

    def test_timeout_classification(self):
        """Test timeout error classification."""
        from utils.fallback import (
            FallbackChain,
            LambdaFallbackStrategy,
            FallbackReason,
        )

        def timeout_func():
            raise TimeoutError("Connection timeout")

        primary = LambdaFallbackStrategy(name="primary", func=timeout_func)
        fallback = LambdaFallbackStrategy(name="fallback", func=lambda: "result")

        chain = FallbackChain(name="test", strategies=[primary, fallback])
        result = chain.execute()

        assert result.fallback_reason == FallbackReason.PRIMARY_TIMEOUT

    def test_rate_limit_classification(self):
        """Test rate limit error classification."""
        from utils.fallback import (
            FallbackChain,
            LambdaFallbackStrategy,
            FallbackReason,
        )

        def rate_limited():
            raise Exception("Rate limit exceeded (429)")

        primary = LambdaFallbackStrategy(name="primary", func=rate_limited)
        fallback = LambdaFallbackStrategy(name="fallback", func=lambda: "result")

        chain = FallbackChain(name="test", strategies=[primary, fallback])
        result = chain.execute()

        assert result.fallback_reason == FallbackReason.RATE_LIMITED

    def test_unavailable_classification(self):
        """Test service unavailable error classification."""
        from utils.fallback import (
            FallbackChain,
            LambdaFallbackStrategy,
            FallbackReason,
        )

        def unavailable():
            raise Exception("Service unavailable (503)")

        primary = LambdaFallbackStrategy(name="primary", func=unavailable)
        fallback = LambdaFallbackStrategy(name="fallback", func=lambda: "result")

        chain = FallbackChain(name="test", strategies=[primary, fallback])
        result = chain.execute()

        assert result.fallback_reason == FallbackReason.SERVICE_UNAVAILABLE


class TestDefaultModelChain:
    """Tests for default model chain configuration."""

    def test_default_chain_exists(self):
        """Test that default model chain is configured."""
        from utils.fallback import DEFAULT_MODEL_CHAIN

        assert len(DEFAULT_MODEL_CHAIN) >= 2

    def test_primary_model_not_degraded(self):
        """Test that primary model is not marked as degraded."""
        from utils.fallback import DEFAULT_MODEL_CHAIN

        primary = DEFAULT_MODEL_CHAIN[0]
        assert primary.is_degraded is False

    def test_fallback_models_are_degraded(self):
        """Test that fallback models are marked as degraded."""
        from utils.fallback import DEFAULT_MODEL_CHAIN

        for model in DEFAULT_MODEL_CHAIN[1:]:
            assert model.is_degraded is True
