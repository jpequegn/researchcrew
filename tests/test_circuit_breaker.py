"""Tests for Circuit Breaker Pattern

Tests the circuit breaker module for ResearchCrew.
"""

import time
from datetime import UTC

import pytest


class TestCircuitBreakerStates:
    """Tests for circuit breaker state transitions."""

    def setup_method(self):
        """Reset circuit breakers before each test."""
        from utils import init_metrics, init_tracing, reset_metrics, reset_tracing
        from utils.circuit_breaker import clear_circuit_breakers

        clear_circuit_breakers()
        reset_tracing()
        reset_metrics()
        init_tracing(exporter_type="none")
        init_metrics()

    def teardown_method(self):
        """Clean up after each test."""
        from utils import reset_metrics, reset_tracing
        from utils.circuit_breaker import clear_circuit_breakers

        clear_circuit_breakers()
        reset_tracing()
        reset_metrics()

    def test_initial_state_is_closed(self):
        """Test that circuit starts in closed state."""
        from utils.circuit_breaker import CircuitBreaker, CircuitState

        cb = CircuitBreaker("test")
        assert cb.state == CircuitState.CLOSED

    def test_transitions_to_open_on_failures(self):
        """Test circuit opens after failure threshold."""
        from utils.circuit_breaker import (
            CircuitBreaker,
            CircuitBreakerConfig,
            CircuitState,
        )

        config = CircuitBreakerConfig(
            failure_threshold=3,
            failure_window=60.0,
        )
        cb = CircuitBreaker("test", config=config)

        # Simulate failures
        for _ in range(3):
            try:
                cb.call(lambda: (_ for _ in ()).throw(Exception("fail")))
            except Exception:
                pass

        assert cb.state == CircuitState.OPEN

    def test_rejects_calls_when_open(self):
        """Test that open circuit rejects calls."""
        from utils.circuit_breaker import (
            CircuitBreaker,
            CircuitBreakerConfig,
            CircuitOpenError,
            CircuitState,
        )

        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=60.0,
        )
        cb = CircuitBreaker("test", config=config)

        # Open the circuit
        for _ in range(2):
            try:
                cb.call(lambda: (_ for _ in ()).throw(Exception("fail")))
            except Exception:
                pass

        assert cb.state == CircuitState.OPEN

        # New call should be rejected
        with pytest.raises(CircuitOpenError):
            cb.call(lambda: "should not run")

    def test_transitions_to_half_open_after_timeout(self):
        """Test circuit transitions to half-open after recovery timeout."""
        from utils.circuit_breaker import (
            CircuitBreaker,
            CircuitBreakerConfig,
            CircuitState,
        )

        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=0.1,  # 100ms for fast testing
        )
        cb = CircuitBreaker("test", config=config)

        # Open the circuit
        for _ in range(2):
            try:
                cb.call(lambda: (_ for _ in ()).throw(Exception("fail")))
            except Exception:
                pass

        assert cb.state == CircuitState.OPEN

        # Wait for recovery timeout
        time.sleep(0.15)

        # State should now be half-open (checked on access)
        assert cb.state == CircuitState.HALF_OPEN

    def test_closes_after_successes_in_half_open(self):
        """Test circuit closes after success threshold in half-open."""
        from utils.circuit_breaker import (
            CircuitBreaker,
            CircuitBreakerConfig,
            CircuitState,
        )

        config = CircuitBreakerConfig(
            failure_threshold=2,
            success_threshold=2,
            recovery_timeout=0.05,
        )
        cb = CircuitBreaker("test", config=config)

        # Open the circuit
        for _ in range(2):
            try:
                cb.call(lambda: (_ for _ in ()).throw(Exception("fail")))
            except Exception:
                pass

        time.sleep(0.1)  # Wait for half-open

        assert cb.state == CircuitState.HALF_OPEN

        # Successful calls should close the circuit
        cb.call(lambda: "success")
        cb.call(lambda: "success")

        assert cb.state == CircuitState.CLOSED

    def test_reopens_on_failure_in_half_open(self):
        """Test circuit reopens on failure in half-open state."""
        from utils.circuit_breaker import (
            CircuitBreaker,
            CircuitBreakerConfig,
            CircuitState,
        )

        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=0.05,
        )
        cb = CircuitBreaker("test", config=config)

        # Open the circuit
        for _ in range(2):
            try:
                cb.call(lambda: (_ for _ in ()).throw(Exception("fail")))
            except Exception:
                pass

        time.sleep(0.1)  # Wait for half-open

        assert cb.state == CircuitState.HALF_OPEN

        # Failure should reopen
        try:
            cb.call(lambda: (_ for _ in ()).throw(Exception("fail")))
        except Exception:
            pass

        assert cb.state == CircuitState.OPEN


class TestCircuitBreakerConfig:
    """Tests for circuit breaker configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        from utils.circuit_breaker import CircuitBreakerConfig

        config = CircuitBreakerConfig()
        assert config.failure_threshold == 5
        assert config.success_threshold == 2
        assert config.recovery_timeout == 30.0
        assert config.failure_window == 60.0

    def test_preset_configs(self):
        """Test preset configurations."""
        from utils.circuit_breaker import CircuitBreakerPreset

        for preset in CircuitBreakerPreset:
            config = preset.value
            assert config.failure_threshold > 0
            assert config.success_threshold > 0
            assert config.recovery_timeout > 0

    def test_external_api_preset(self):
        """Test external API preset."""
        from utils.circuit_breaker import CircuitBreakerPreset

        config = CircuitBreakerPreset.EXTERNAL_API.value
        assert config.failure_threshold == 5
        assert config.recovery_timeout == 30.0

    def test_llm_call_preset(self):
        """Test LLM call preset (more conservative)."""
        from utils.circuit_breaker import CircuitBreakerPreset

        config = CircuitBreakerPreset.LLM_CALL.value
        assert config.failure_threshold == 3
        assert config.recovery_timeout == 60.0


class TestCircuitBreakerFallback:
    """Tests for circuit breaker fallback behavior."""

    def setup_method(self):
        """Reset circuit breakers before each test."""
        from utils import init_metrics, init_tracing, reset_metrics, reset_tracing
        from utils.circuit_breaker import clear_circuit_breakers

        clear_circuit_breakers()
        reset_tracing()
        reset_metrics()
        init_tracing(exporter_type="none")
        init_metrics()

    def teardown_method(self):
        """Clean up after each test."""
        from utils import reset_metrics, reset_tracing
        from utils.circuit_breaker import clear_circuit_breakers

        clear_circuit_breakers()
        reset_tracing()
        reset_metrics()

    def test_fallback_called_when_open(self):
        """Test fallback is called when circuit is open."""
        from utils.circuit_breaker import CircuitBreaker, CircuitBreakerConfig

        fallback_called = False

        def fallback():
            nonlocal fallback_called
            fallback_called = True
            return "fallback_result"

        config = CircuitBreakerConfig(failure_threshold=2)
        cb = CircuitBreaker("test", config=config, fallback=fallback)

        # Open the circuit
        for _ in range(2):
            try:
                cb.call(lambda: (_ for _ in ()).throw(Exception("fail")))
            except Exception:
                pass

        # Call with open circuit should use fallback
        result = cb.call(lambda: "should not run")

        assert fallback_called
        assert result == "fallback_result"

    def test_fallback_receives_args(self):
        """Test fallback receives the original arguments."""
        from utils.circuit_breaker import CircuitBreaker, CircuitBreakerConfig

        received_args = []

        def fallback(*args, **kwargs):
            received_args.append((args, kwargs))
            return "fallback"

        config = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker("test", config=config, fallback=fallback)

        # Open the circuit
        try:
            cb.call(lambda: (_ for _ in ()).throw(Exception("fail")))
        except Exception:
            pass

        # Call with args
        cb.call(lambda x, y: None, "arg1", y="kwarg1")

        assert received_args[0] == (("arg1",), {"y": "kwarg1"})


class TestCircuitBreakerStats:
    """Tests for circuit breaker statistics."""

    def setup_method(self):
        """Reset circuit breakers before each test."""
        from utils import init_metrics, init_tracing, reset_metrics, reset_tracing
        from utils.circuit_breaker import clear_circuit_breakers

        clear_circuit_breakers()
        reset_tracing()
        reset_metrics()
        init_tracing(exporter_type="none")
        init_metrics()

    def teardown_method(self):
        """Clean up after each test."""
        from utils import reset_metrics, reset_tracing
        from utils.circuit_breaker import clear_circuit_breakers

        clear_circuit_breakers()
        reset_tracing()
        reset_metrics()

    def test_stats_tracking(self):
        """Test that statistics are tracked correctly."""
        from utils.circuit_breaker import CircuitBreaker, CircuitState

        cb = CircuitBreaker("test")

        # Make some calls
        cb.call(lambda: "success")
        cb.call(lambda: "success")
        try:
            cb.call(lambda: (_ for _ in ()).throw(Exception("fail")))
        except Exception:
            pass

        stats = cb.get_stats()
        assert stats.name == "test"
        assert stats.total_calls == 3
        assert stats.total_failures == 1
        assert stats.state == CircuitState.CLOSED

    def test_stats_to_dict(self):
        """Test stats serialization."""
        from datetime import datetime

        from utils.circuit_breaker import CircuitBreakerStats, CircuitState

        stats = CircuitBreakerStats(
            name="test",
            state=CircuitState.CLOSED,
            failure_count=2,
            success_count=5,
            total_calls=10,
            total_failures=3,
            total_trips=1,
            last_failure_time=datetime.now(UTC),
            last_state_change=datetime.now(UTC),
        )

        d = stats.to_dict()
        assert d["name"] == "test"
        assert d["state"] == "closed"
        assert d["failure_count"] == 2
        assert d["total_trips"] == 1

    def test_trip_counter(self):
        """Test that trips are counted correctly."""
        from utils.circuit_breaker import CircuitBreaker, CircuitBreakerConfig

        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=0.05,
            success_threshold=1,
        )
        cb = CircuitBreaker("test", config=config)

        # First trip
        for _ in range(2):
            try:
                cb.call(lambda: (_ for _ in ()).throw(Exception("fail")))
            except Exception:
                pass

        time.sleep(0.1)  # Wait for half-open
        cb.call(lambda: "success")  # Close

        # Second trip
        for _ in range(2):
            try:
                cb.call(lambda: (_ for _ in ()).throw(Exception("fail")))
            except Exception:
                pass

        stats = cb.get_stats()
        assert stats.total_trips == 2


class TestCircuitBreakerRegistry:
    """Tests for circuit breaker registry."""

    def setup_method(self):
        """Reset circuit breakers before each test."""
        from utils import init_metrics, init_tracing, reset_metrics, reset_tracing
        from utils.circuit_breaker import clear_circuit_breakers

        clear_circuit_breakers()
        reset_tracing()
        reset_metrics()
        init_tracing(exporter_type="none")
        init_metrics()

    def teardown_method(self):
        """Clean up after each test."""
        from utils import reset_metrics, reset_tracing
        from utils.circuit_breaker import clear_circuit_breakers

        clear_circuit_breakers()
        reset_tracing()
        reset_metrics()

    def test_get_or_create(self):
        """Test get_circuit_breaker creates if not exists."""
        from utils.circuit_breaker import get_circuit_breaker

        cb1 = get_circuit_breaker("test")
        cb2 = get_circuit_breaker("test")

        assert cb1 is cb2

    def test_get_with_preset(self):
        """Test get_circuit_breaker with preset."""
        from utils.circuit_breaker import CircuitBreakerPreset, get_circuit_breaker

        cb = get_circuit_breaker("test", preset=CircuitBreakerPreset.LLM_CALL)
        assert cb.config.failure_threshold == 3

    def test_reset_single_circuit(self):
        """Test resetting a single circuit breaker."""
        from utils.circuit_breaker import (
            CircuitBreakerConfig,
            CircuitState,
            get_circuit_breaker,
            reset_circuit_breaker,
        )

        config = CircuitBreakerConfig(failure_threshold=2)
        cb = get_circuit_breaker("test", config=config)

        # Open the circuit
        for _ in range(2):
            try:
                cb.call(lambda: (_ for _ in ()).throw(Exception("fail")))
            except Exception:
                pass

        assert cb.state == CircuitState.OPEN

        # Reset
        reset_circuit_breaker("test")
        assert cb.state == CircuitState.CLOSED

    def test_reset_all_circuits(self):
        """Test resetting all circuit breakers."""
        from utils.circuit_breaker import (
            CircuitBreakerConfig,
            CircuitState,
            get_circuit_breaker,
            reset_all_circuit_breakers,
        )

        config = CircuitBreakerConfig(failure_threshold=1)

        cb1 = get_circuit_breaker("test1", config=config)
        cb2 = get_circuit_breaker("test2", config=config)

        # Open both circuits
        try:
            cb1.call(lambda: (_ for _ in ()).throw(Exception("fail")))
        except Exception:
            pass
        try:
            cb2.call(lambda: (_ for _ in ()).throw(Exception("fail")))
        except Exception:
            pass

        reset_all_circuit_breakers()

        assert cb1.state == CircuitState.CLOSED
        assert cb2.state == CircuitState.CLOSED

    def test_get_all_stats(self):
        """Test getting stats for all circuits."""
        from utils.circuit_breaker import get_all_circuit_stats, get_circuit_breaker

        get_circuit_breaker("test1")
        get_circuit_breaker("test2")

        stats = get_all_circuit_stats()
        assert "test1" in stats
        assert "test2" in stats


class TestCircuitBreakerDecorator:
    """Tests for circuit_breaker_call decorator."""

    def setup_method(self):
        """Reset circuit breakers before each test."""
        from utils import init_metrics, init_tracing, reset_metrics, reset_tracing
        from utils.circuit_breaker import clear_circuit_breakers

        clear_circuit_breakers()
        reset_tracing()
        reset_metrics()
        init_tracing(exporter_type="none")
        init_metrics()

    def teardown_method(self):
        """Clean up after each test."""
        from utils import reset_metrics, reset_tracing
        from utils.circuit_breaker import clear_circuit_breakers

        clear_circuit_breakers()
        reset_tracing()
        reset_metrics()

    def test_decorator_basic(self):
        """Test basic decorator usage."""
        from utils.circuit_breaker import circuit_breaker_call

        @circuit_breaker_call("test_decorator")
        def my_function():
            return "success"

        result = my_function()
        assert result == "success"

    def test_decorator_with_failure(self):
        """Test decorator with failing function."""
        from utils.circuit_breaker import (
            CircuitBreakerConfig,
            CircuitState,
            circuit_breaker_call,
            get_circuit_breaker,
        )

        config = CircuitBreakerConfig(failure_threshold=2)

        @circuit_breaker_call("test_fail", config=config)
        def failing_function():
            raise ValueError("Always fails")

        # Call until circuit opens
        for _ in range(2):
            try:
                failing_function()
            except ValueError:
                pass

        cb = get_circuit_breaker("test_fail")
        assert cb.state == CircuitState.OPEN


class TestCircuitBreakerMonitoring:
    """Tests for circuit breaker monitoring helpers."""

    def setup_method(self):
        """Reset circuit breakers before each test."""
        from utils import init_metrics, init_tracing, reset_metrics, reset_tracing
        from utils.circuit_breaker import clear_circuit_breakers

        clear_circuit_breakers()
        reset_tracing()
        reset_metrics()
        init_tracing(exporter_type="none")
        init_metrics()

    def teardown_method(self):
        """Clean up after each test."""
        from utils import reset_metrics, reset_tracing
        from utils.circuit_breaker import clear_circuit_breakers

        clear_circuit_breakers()
        reset_tracing()
        reset_metrics()

    def test_get_status_healthy(self):
        """Test status when all circuits healthy."""
        from utils.circuit_breaker import (
            get_circuit_breaker,
            get_circuit_breaker_status,
        )

        get_circuit_breaker("test1")
        get_circuit_breaker("test2")

        status = get_circuit_breaker_status()
        assert status["healthy"] is True
        assert len(status["open_circuits"]) == 0

    def test_get_status_unhealthy(self):
        """Test status when circuits are open."""
        from utils.circuit_breaker import (
            CircuitBreakerConfig,
            get_circuit_breaker,
            get_circuit_breaker_status,
        )

        config = CircuitBreakerConfig(failure_threshold=1)
        cb = get_circuit_breaker("test", config=config)

        try:
            cb.call(lambda: (_ for _ in ()).throw(Exception("fail")))
        except Exception:
            pass

        status = get_circuit_breaker_status()
        assert status["healthy"] is False
        assert "test" in status["open_circuits"]

    def test_is_service_healthy(self):
        """Test is_service_healthy helper."""
        from utils.circuit_breaker import (
            CircuitBreakerConfig,
            get_circuit_breaker,
            is_service_healthy,
        )

        # Non-existent circuit is considered healthy
        assert is_service_healthy("nonexistent") is True

        config = CircuitBreakerConfig(failure_threshold=1)
        cb = get_circuit_breaker("test", config=config)

        assert is_service_healthy("test") is True

        # Open the circuit
        try:
            cb.call(lambda: (_ for _ in ()).throw(Exception("fail")))
        except Exception:
            pass

        assert is_service_healthy("test") is False


class TestCircuitBreakerReset:
    """Tests for manual circuit breaker control."""

    def setup_method(self):
        """Reset circuit breakers before each test."""
        from utils import init_metrics, init_tracing, reset_metrics, reset_tracing
        from utils.circuit_breaker import clear_circuit_breakers

        clear_circuit_breakers()
        reset_tracing()
        reset_metrics()
        init_tracing(exporter_type="none")
        init_metrics()

    def teardown_method(self):
        """Clean up after each test."""
        from utils import reset_metrics, reset_tracing
        from utils.circuit_breaker import clear_circuit_breakers

        clear_circuit_breakers()
        reset_tracing()
        reset_metrics()

    def test_manual_reset(self):
        """Test manually resetting a circuit."""
        from utils.circuit_breaker import (
            CircuitBreaker,
            CircuitBreakerConfig,
            CircuitState,
        )

        config = CircuitBreakerConfig(failure_threshold=2)
        cb = CircuitBreaker("test", config=config)

        # Open the circuit
        for _ in range(2):
            try:
                cb.call(lambda: (_ for _ in ()).throw(Exception("fail")))
            except Exception:
                pass

        assert cb.state == CircuitState.OPEN

        cb.reset()
        assert cb.state == CircuitState.CLOSED

    def test_force_open(self):
        """Test forcing a circuit open."""
        from utils.circuit_breaker import CircuitBreaker, CircuitState

        cb = CircuitBreaker("test")
        assert cb.state == CircuitState.CLOSED

        cb.force_open()
        assert cb.state == CircuitState.OPEN


class TestCircuitBreakerStateChange:
    """Tests for state change callbacks."""

    def setup_method(self):
        """Reset circuit breakers before each test."""
        from utils import init_metrics, init_tracing, reset_metrics, reset_tracing
        from utils.circuit_breaker import clear_circuit_breakers

        clear_circuit_breakers()
        reset_tracing()
        reset_metrics()
        init_tracing(exporter_type="none")
        init_metrics()

    def teardown_method(self):
        """Clean up after each test."""
        from utils import reset_metrics, reset_tracing
        from utils.circuit_breaker import clear_circuit_breakers

        clear_circuit_breakers()
        reset_tracing()
        reset_metrics()

    def test_state_change_callback(self):
        """Test state change callback is called."""
        from utils.circuit_breaker import (
            CircuitBreaker,
            CircuitBreakerConfig,
            CircuitState,
        )

        state_changes = []

        def on_change(old_state, new_state):
            state_changes.append((old_state, new_state))

        config = CircuitBreakerConfig(failure_threshold=2)
        cb = CircuitBreaker("test", config=config, on_state_change=on_change)

        # Open the circuit
        for _ in range(2):
            try:
                cb.call(lambda: (_ for _ in ()).throw(Exception("fail")))
            except Exception:
                pass

        assert len(state_changes) == 1
        assert state_changes[0] == (CircuitState.CLOSED, CircuitState.OPEN)
