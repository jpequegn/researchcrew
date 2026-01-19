"""Tests for Resilience Utilities

Tests the retry logic and error classification for ResearchCrew.
"""

from unittest.mock import Mock

import pytest


class TestErrorClassification:
    """Tests for error classification."""

    def test_classify_timeout_error(self):
        """Test classification of timeout errors."""
        from utils.resilience import TransientError, classify_error

        error = Exception("Connection timeout after 30s")
        classified = classify_error(error)
        assert isinstance(classified, TransientError)

    def test_classify_connection_error(self):
        """Test classification of connection errors."""
        from utils.resilience import NetworkError, classify_error

        error = Exception("Connection refused")
        classified = classify_error(error)
        assert isinstance(classified, NetworkError)

    def test_classify_rate_limit_error(self):
        """Test classification of rate limit errors."""
        from utils.resilience import RateLimitError, classify_error

        error = Exception("Rate limit exceeded, retry after 60s")
        classified = classify_error(error)
        assert isinstance(classified, RateLimitError)

    def test_classify_server_error(self):
        """Test classification of server errors (5xx)."""
        from utils.resilience import ServerError, classify_error

        # Create mock HTTP error with status code
        error = Mock()
        error.status_code = 500
        error.__str__ = lambda self: "Internal Server Error"

        classified = classify_error(error)
        assert isinstance(classified, ServerError)
        assert classified.status_code == 500

    def test_classify_client_error(self):
        """Test classification of client errors (4xx)."""
        from utils.resilience import ClientError, classify_error

        # Create mock HTTP error with status code
        error = Mock()
        error.status_code = 400
        error.__str__ = lambda self: "Bad Request"

        classified = classify_error(error)
        assert isinstance(classified, ClientError)
        assert classified.status_code == 400

    def test_classify_auth_error(self):
        """Test classification of authentication errors."""
        from utils.resilience import AuthenticationError, classify_error

        error = Exception("Unauthorized: invalid API key")
        classified = classify_error(error)
        assert isinstance(classified, AuthenticationError)

    def test_classify_validation_error(self):
        """Test classification of validation errors."""
        from utils.resilience import ValidationError, classify_error

        error = Exception("Invalid input: required field missing")
        classified = classify_error(error)
        assert isinstance(classified, ValidationError)

    def test_classify_already_classified(self):
        """Test that already classified errors are returned as-is."""
        from utils.resilience import PermanentError, TransientError, classify_error

        transient = TransientError("Already transient")
        permanent = PermanentError("Already permanent")

        assert classify_error(transient) is transient
        assert classify_error(permanent) is permanent

    def test_classify_unknown_error(self):
        """Test that unknown errors default to transient."""
        from utils.resilience import TransientError, classify_error

        error = Exception("Some unknown error")
        classified = classify_error(error)
        # Unknown errors should be classified as transient for safety
        assert isinstance(classified, TransientError)


class TestTransientError:
    """Tests for TransientError class."""

    def test_basic_creation(self):
        """Test basic TransientError creation."""
        from utils.resilience import TransientError

        error = TransientError("Test error")
        assert str(error) == "Test error"
        assert error.original_error is None
        assert error.retry_after is None

    def test_with_original_error(self):
        """Test TransientError with original error."""
        from utils.resilience import TransientError

        original = ValueError("Original")
        error = TransientError("Wrapped", original_error=original)
        assert error.original_error is original

    def test_with_retry_after(self):
        """Test TransientError with retry_after."""
        from utils.resilience import TransientError

        error = TransientError("Rate limited", retry_after=60.0)
        assert error.retry_after == 60.0


class TestPermanentError:
    """Tests for PermanentError class."""

    def test_basic_creation(self):
        """Test basic PermanentError creation."""
        from utils.resilience import PermanentError

        error = PermanentError("Test error")
        assert str(error) == "Test error"
        assert error.error_code is None

    def test_with_error_code(self):
        """Test PermanentError with error code."""
        from utils.resilience import PermanentError

        error = PermanentError("Auth failed", error_code="AUTH_FAILED")
        assert error.error_code == "AUTH_FAILED"


class TestRetryPolicy:
    """Tests for RetryPolicy enum."""

    def test_web_search_policy(self):
        """Test web search retry policy."""
        from utils.resilience import RetryPolicy

        config = RetryPolicy.WEB_SEARCH.value
        assert config.max_attempts == 3
        assert config.max_wait == 60.0

    def test_llm_call_policy(self):
        """Test LLM call retry policy."""
        from utils.resilience import RetryPolicy

        config = RetryPolicy.LLM_CALL.value
        assert config.max_attempts == 2
        assert config.max_wait == 120.0

    def test_all_policies_have_valid_config(self):
        """Test all policies have valid configuration."""
        from utils.resilience import RetryPolicy

        for policy in RetryPolicy:
            config = policy.value
            assert config.max_attempts > 0
            assert config.min_wait >= 0
            assert config.max_wait > config.min_wait
            assert config.multiplier > 0
            assert 0 <= config.jitter <= 1


class TestRetryWithBackoff:
    """Tests for retry_with_backoff decorator."""

    def setup_method(self):
        """Reset state before each test."""
        from utils import init_metrics, init_tracing, reset_metrics, reset_tracing
        from utils.resilience import clear_retry_stats

        clear_retry_stats()
        reset_tracing()
        reset_metrics()
        init_tracing(exporter_type="none")
        init_metrics()

    def teardown_method(self):
        """Reset state after each test."""
        from utils import reset_metrics, reset_tracing
        from utils.resilience import clear_retry_stats

        clear_retry_stats()
        reset_tracing()
        reset_metrics()

    def test_success_no_retry(self):
        """Test successful call without retry."""
        from utils.resilience import RetryPolicy, get_retry_stats, retry_with_backoff

        call_count = 0

        @retry_with_backoff(policy=RetryPolicy.DEFAULT)
        def successful_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = successful_func()

        assert result == "success"
        assert call_count == 1

        stats = get_retry_stats()
        assert len(stats) == 1
        assert stats[0].attempts == 1
        assert stats[0].final_status == "success"

    def test_retry_on_transient_error(self):
        """Test retry on transient error."""
        from utils.resilience import (
            RetryConfig,
            TransientError,
            get_retry_stats,
            retry_with_backoff,
        )

        call_count = 0

        # Custom config for fast testing
        fast_config = RetryConfig(
            max_attempts=3,
            min_wait=0.01,
            max_wait=0.1,
            multiplier=2.0,
            jitter=0.0,
        )

        @retry_with_backoff(policy=fast_config)
        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise TransientError("Transient failure")
            return "success"

        result = flaky_func()

        assert result == "success"
        assert call_count == 2

        stats = get_retry_stats()
        assert len(stats) == 1
        assert stats[0].attempts == 2
        assert stats[0].final_status == "success"

    def test_no_retry_on_permanent_error(self):
        """Test that permanent errors are not retried."""
        from utils.resilience import (
            PermanentError,
            RetryConfig,
            get_retry_stats,
            retry_with_backoff,
        )

        call_count = 0

        fast_config = RetryConfig(
            max_attempts=3,
            min_wait=0.01,
            max_wait=0.1,
            multiplier=2.0,
            jitter=0.0,
        )

        @retry_with_backoff(policy=fast_config, reraise_permanent=True)
        def permanent_fail():
            nonlocal call_count
            call_count += 1
            raise PermanentError("Permanent failure")

        with pytest.raises(PermanentError):
            permanent_fail()

        # Should only be called once (no retry)
        assert call_count == 1

        stats = get_retry_stats()
        assert len(stats) == 1
        assert stats[0].final_status == "failure"

    def test_exhausted_retries(self):
        """Test behavior when all retries are exhausted."""
        from utils.resilience import (
            RetryConfig,
            TransientError,
            get_retry_stats,
            retry_with_backoff,
        )

        call_count = 0

        fast_config = RetryConfig(
            max_attempts=3,
            min_wait=0.01,
            max_wait=0.1,
            multiplier=2.0,
            jitter=0.0,
        )

        @retry_with_backoff(policy=fast_config)
        def always_fails():
            nonlocal call_count
            call_count += 1
            raise TransientError("Always fails")

        with pytest.raises(TransientError):
            always_fails()

        assert call_count == 3

        stats = get_retry_stats()
        assert len(stats) == 1
        assert stats[0].final_status == "exhausted"
        assert stats[0].attempts == 3

    def test_custom_operation_name(self):
        """Test custom operation name in stats."""
        from utils.resilience import (
            RetryConfig,
            get_retry_stats,
            retry_with_backoff,
        )

        fast_config = RetryConfig(
            max_attempts=2,
            min_wait=0.01,
            max_wait=0.1,
            multiplier=2.0,
            jitter=0.0,
        )

        @retry_with_backoff(policy=fast_config, operation_name="custom_op")
        def my_func():
            return "done"

        my_func()

        stats = get_retry_stats()
        assert stats[0].operation == "custom_op"


class TestRetryStats:
    """Tests for RetryStats."""

    def test_to_dict(self):
        """Test RetryStats serialization."""
        from utils.resilience import RetryStats

        stats = RetryStats(
            operation="test_op",
            attempts=3,
            final_status="success",
            total_time=1.5,
            last_error=None,
            trace_id="abc123",
        )

        d = stats.to_dict()
        assert d["operation"] == "test_op"
        assert d["attempts"] == 3
        assert d["final_status"] == "success"
        assert d["total_time"] == 1.5
        assert d["trace_id"] == "abc123"


class TestIsRetriable:
    """Tests for is_retriable function."""

    def test_transient_is_retriable(self):
        """Test that transient errors are retriable."""
        from utils.resilience import TransientError, is_retriable

        error = TransientError("Transient")
        assert is_retriable(error) is True

    def test_permanent_not_retriable(self):
        """Test that permanent errors are not retriable."""
        from utils.resilience import PermanentError, is_retriable

        error = PermanentError("Permanent")
        assert is_retriable(error) is False

    def test_timeout_is_retriable(self):
        """Test that timeout errors are retriable."""
        from utils.resilience import is_retriable

        error = Exception("Connection timeout")
        assert is_retriable(error) is True


class TestGetRetryWait:
    """Tests for get_retry_wait function."""

    def test_exponential_backoff(self):
        """Test exponential backoff calculation."""
        from utils.resilience import RetryConfig, TransientError, get_retry_wait

        config = RetryConfig(
            max_attempts=3,
            min_wait=1.0,
            max_wait=60.0,
            multiplier=2.0,
            jitter=0.0,  # No jitter for deterministic test
        )

        error = TransientError("Test")

        wait1 = get_retry_wait(error, 1, config)
        wait2 = get_retry_wait(error, 2, config)
        wait3 = get_retry_wait(error, 3, config)

        assert wait1 == 1.0  # min_wait
        assert wait2 == 2.0  # min_wait * multiplier
        assert wait3 == 4.0  # min_wait * multiplier^2

    def test_respects_max_wait(self):
        """Test that wait time respects max_wait."""
        from utils.resilience import RetryConfig, TransientError, get_retry_wait

        config = RetryConfig(
            max_attempts=10,
            min_wait=1.0,
            max_wait=5.0,
            multiplier=10.0,
            jitter=0.0,
        )

        error = TransientError("Test")

        wait = get_retry_wait(error, 3, config)
        assert wait <= 5.0  # Should be clamped to max_wait

    def test_uses_retry_after_header(self):
        """Test that retry_after from error is respected."""
        from utils.resilience import RetryConfig, TransientError, get_retry_wait

        config = RetryConfig(
            max_attempts=3,
            min_wait=1.0,
            max_wait=60.0,
            multiplier=2.0,
            jitter=0.0,
        )

        error = TransientError("Rate limited", retry_after=30.0)

        wait = get_retry_wait(error, 1, config)
        assert wait == 30.0


class TestSpecificErrors:
    """Tests for specific error classes."""

    def test_rate_limit_error(self):
        """Test RateLimitError."""
        from utils.resilience import RateLimitError

        error = RateLimitError(retry_after=60.0)
        assert error.retry_after == 60.0
        assert "Rate limit" in str(error)

    def test_network_error(self):
        """Test NetworkError."""
        from utils.resilience import NetworkError

        original = ConnectionError("Connection refused")
        error = NetworkError("Network failed", original_error=original)
        assert error.original_error is original

    def test_server_error(self):
        """Test ServerError."""
        from utils.resilience import ServerError

        error = ServerError(status_code=503)
        assert error.status_code == 503

    def test_client_error(self):
        """Test ClientError."""
        from utils.resilience import ClientError

        error = ClientError(status_code=404)
        assert error.status_code == 404

    def test_authentication_error(self):
        """Test AuthenticationError."""
        from utils.resilience import AuthenticationError

        error = AuthenticationError("Invalid token")
        assert error.error_code == "AUTH_FAILED"

    def test_validation_error(self):
        """Test ValidationError."""
        from utils.resilience import ValidationError

        error = ValidationError("Missing field", field="name")
        assert error.field == "name"
        assert error.error_code == "VALIDATION_ERROR"


class TestClearRetryStats:
    """Tests for clear_retry_stats function."""

    def setup_method(self):
        """Reset state before each test."""
        from utils import init_metrics, init_tracing, reset_metrics, reset_tracing
        from utils.resilience import clear_retry_stats

        clear_retry_stats()
        reset_tracing()
        reset_metrics()
        init_tracing(exporter_type="none")
        init_metrics()

    def teardown_method(self):
        """Reset state after each test."""
        from utils import reset_metrics, reset_tracing
        from utils.resilience import clear_retry_stats

        clear_retry_stats()
        reset_tracing()
        reset_metrics()

    def test_clear_stats(self):
        """Test clearing retry stats."""
        from utils.resilience import (
            RetryConfig,
            clear_retry_stats,
            get_retry_stats,
            retry_with_backoff,
        )

        fast_config = RetryConfig(
            max_attempts=2,
            min_wait=0.01,
            max_wait=0.1,
            multiplier=2.0,
            jitter=0.0,
        )

        @retry_with_backoff(policy=fast_config)
        def my_func():
            return "done"

        my_func()
        assert len(get_retry_stats()) == 1

        clear_retry_stats()
        assert len(get_retry_stats()) == 0


class TestIntegrationWithMetrics:
    """Integration tests with metrics system."""

    def setup_method(self):
        """Reset state before each test."""
        from utils import init_metrics, init_tracing, reset_metrics, reset_tracing
        from utils.resilience import clear_retry_stats

        clear_retry_stats()
        reset_tracing()
        reset_metrics()
        init_tracing(exporter_type="none")
        init_metrics()

    def teardown_method(self):
        """Reset state after each test."""
        from utils import reset_metrics, reset_tracing
        from utils.resilience import clear_retry_stats

        clear_retry_stats()
        reset_tracing()
        reset_metrics()

    def test_retry_records_metrics(self):
        """Test that retries are recorded in metrics."""
        from utils import get_metrics_text
        from utils.resilience import (
            RetryConfig,
            TransientError,
            retry_with_backoff,
        )

        call_count = 0

        fast_config = RetryConfig(
            max_attempts=3,
            min_wait=0.01,
            max_wait=0.1,
            multiplier=2.0,
            jitter=0.0,
        )

        @retry_with_backoff(policy=fast_config, operation_name="test_retry")
        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise TransientError("Transient")
            return "success"

        flaky_func()

        metrics = get_metrics_text()
        assert "tool_calls_total" in metrics
