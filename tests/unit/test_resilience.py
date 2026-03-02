"""
Tests for resilience patterns (retry, circuit breaker).

Constitution A6: Fail-Closed Semantics - These patterns ensure
cascading failures don't take down the system.
"""

from datetime import datetime, timedelta

from aperion_switchboard.core.resilience import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitOpenError,
    CircuitState,
    RetryConfig,
    get_all_circuit_stats,
    get_circuit_breaker,
    reset_circuit_breakers,
)


class TestRetryConfig:
    """Tests for RetryConfig exponential backoff."""

    def test_default_values(self):
        """Verify sensible defaults."""
        config = RetryConfig()

        assert config.max_attempts == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert 429 in config.retryable_status_codes
        assert 503 in config.retryable_status_codes

    def test_exponential_backoff_without_jitter(self):
        """Verify exponential growth of delay."""
        config = RetryConfig(base_delay=1.0, jitter_factor=0.0)

        assert config.calculate_delay(0) == 1.0
        assert config.calculate_delay(1) == 2.0
        assert config.calculate_delay(2) == 4.0
        assert config.calculate_delay(3) == 8.0

    def test_max_delay_cap(self):
        """Verify delay is capped at max_delay."""
        config = RetryConfig(base_delay=1.0, max_delay=10.0, jitter_factor=0.0)

        # 2^10 = 1024, but should be capped at 10
        assert config.calculate_delay(10) == 10.0
        assert config.calculate_delay(100) == 10.0

    def test_retry_after_takes_precedence(self):
        """Verify Retry-After header value is used when provided."""
        config = RetryConfig()

        # Should use retry_after regardless of attempt number
        assert config.calculate_delay(0, retry_after=5.0) == 5.0
        assert config.calculate_delay(10, retry_after=120.0) == 120.0

    def test_jitter_adds_randomness(self):
        """Verify jitter adds some randomness."""
        config = RetryConfig(base_delay=1.0, jitter_factor=0.5)

        # Get multiple samples
        delays = [config.calculate_delay(0) for _ in range(100)]

        # Base delay is 1.0, jitter factor 0.5 means max jitter is 0.5
        # So delays should be between 1.0 and 1.5
        assert all(1.0 <= d <= 1.5 for d in delays)

        # Delays should not all be the same (randomness)
        assert len(set(delays)) > 1


class TestCircuitBreaker:
    """Tests for CircuitBreaker state machine."""

    def setup_method(self):
        """Reset circuit breakers before each test."""
        reset_circuit_breakers()

    def test_starts_closed(self):
        """New circuit breaker should be closed."""
        cb = CircuitBreaker(name="test")

        assert cb.state == CircuitState.CLOSED
        assert cb.can_execute() is True
        assert cb.failure_count == 0

    def test_stays_closed_under_threshold(self):
        """Circuit stays closed if failures < threshold."""
        cb = CircuitBreaker(
            name="test",
            config=CircuitBreakerConfig(failure_threshold=5)
        )

        for _ in range(4):
            cb.record_failure()

        assert cb.state == CircuitState.CLOSED
        assert cb.can_execute() is True

    def test_opens_at_threshold(self):
        """Circuit opens when failures reach threshold."""
        cb = CircuitBreaker(
            name="test",
            config=CircuitBreakerConfig(failure_threshold=3)
        )

        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED

        cb.record_failure()  # 3rd failure
        assert cb.state == CircuitState.OPEN
        assert cb.can_execute() is False

    def test_success_resets_failure_count(self):
        """Success in closed state resets failure count."""
        cb = CircuitBreaker(
            name="test",
            config=CircuitBreakerConfig(failure_threshold=5)
        )

        cb.record_failure()
        cb.record_failure()
        assert cb.failure_count == 2

        cb.record_success()
        assert cb.failure_count == 0
        assert cb.state == CircuitState.CLOSED

    def test_transitions_to_half_open_after_timeout(self):
        """Circuit moves to half-open after timeout expires."""
        cb = CircuitBreaker(
            name="test",
            config=CircuitBreakerConfig(
                failure_threshold=1,
                timeout_seconds=0.01  # Very short for testing
            )
        )

        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        # Simulate time passing
        cb.last_failure_time = datetime.now() - timedelta(seconds=1)

        assert cb.can_execute() is True
        assert cb.state == CircuitState.HALF_OPEN

    def test_half_open_success_closes_circuit(self):
        """Successes in half-open state close the circuit."""
        cb = CircuitBreaker(
            name="test",
            config=CircuitBreakerConfig(
                failure_threshold=1,
                success_threshold=2,
                timeout_seconds=0
            )
        )

        # Open the circuit
        cb.record_failure()
        cb.last_failure_time = datetime.now() - timedelta(seconds=1)
        cb.can_execute()  # Trigger transition to half-open

        assert cb.state == CircuitState.HALF_OPEN

        cb.record_success()
        assert cb.state == CircuitState.HALF_OPEN

        cb.record_success()  # 2nd success
        assert cb.state == CircuitState.CLOSED

    def test_half_open_failure_reopens_circuit(self):
        """Any failure in half-open state reopens the circuit."""
        cb = CircuitBreaker(
            name="test",
            config=CircuitBreakerConfig(
                failure_threshold=1,
                timeout_seconds=0
            )
        )

        # Open the circuit
        cb.record_failure()
        cb.last_failure_time = datetime.now() - timedelta(seconds=1)
        cb.can_execute()  # Transition to half-open

        assert cb.state == CircuitState.HALF_OPEN

        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_half_open_limits_calls(self):
        """Half-open state limits number of probe calls."""
        cb = CircuitBreaker(
            name="test",
            config=CircuitBreakerConfig(
                failure_threshold=1,
                timeout_seconds=0,
                half_open_max_calls=2
            )
        )

        # Open and transition to half-open
        cb.record_failure()
        cb.last_failure_time = datetime.now() - timedelta(seconds=1)

        # First call transitions to half-open and increments counter
        assert cb.can_execute() is True  # half_open_calls = 1
        assert cb.state == CircuitState.HALF_OPEN

        # Second call should also be allowed
        assert cb.can_execute() is True  # half_open_calls = 2

        # Third call exceeds limit (2 >= 2)
        assert cb.can_execute() is False

    def test_reset_returns_to_initial_state(self):
        """Reset clears all state."""
        cb = CircuitBreaker(
            name="test",
            config=CircuitBreakerConfig(failure_threshold=3)
        )

        cb.record_failure()
        cb.record_failure()
        cb.record_failure()  # Now at threshold
        assert cb.state == CircuitState.OPEN

        cb.reset()

        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0
        assert cb.success_count == 0
        assert cb.last_failure_time is None

    def test_get_stats(self):
        """get_stats returns useful information."""
        cb = CircuitBreaker(name="test-provider")
        cb.record_failure()

        stats = cb.get_stats()

        assert stats["name"] == "test-provider"
        assert stats["state"] == "closed"
        assert stats["failure_count"] == 1
        assert stats["last_failure_time"] is not None


class TestCircuitBreakerRegistry:
    """Tests for global circuit breaker registry."""

    def setup_method(self):
        reset_circuit_breakers()

    def test_get_creates_new_breaker(self):
        """get_circuit_breaker creates breaker on first access."""
        cb = get_circuit_breaker("openai")

        assert cb is not None
        assert cb.name == "openai"
        assert cb.state == CircuitState.CLOSED

    def test_get_returns_same_instance(self):
        """get_circuit_breaker returns same instance for same name."""
        cb1 = get_circuit_breaker("gemini")
        cb2 = get_circuit_breaker("gemini")

        assert cb1 is cb2

    def test_different_names_get_different_breakers(self):
        """Different provider names get independent breakers."""
        cb_openai = get_circuit_breaker("openai")
        cb_gemini = get_circuit_breaker("gemini")

        cb_openai.record_failure()
        cb_openai.record_failure()
        cb_openai.record_failure()
        cb_openai.record_failure()
        cb_openai.record_failure()

        assert cb_openai.state == CircuitState.OPEN
        assert cb_gemini.state == CircuitState.CLOSED

    def test_reset_clears_all(self):
        """reset_circuit_breakers clears the registry."""
        get_circuit_breaker("openai")
        get_circuit_breaker("gemini")

        reset_circuit_breakers()

        # Should get new instances after reset
        cb = get_circuit_breaker("openai")
        assert cb.failure_count == 0

    def test_get_all_stats(self):
        """get_all_circuit_stats returns all breaker stats."""
        cb1 = get_circuit_breaker("openai")
        get_circuit_breaker("gemini")

        cb1.record_failure()

        stats = get_all_circuit_stats()

        assert "openai" in stats
        assert "gemini" in stats
        assert stats["openai"]["failure_count"] == 1
        assert stats["gemini"]["failure_count"] == 0


class TestCircuitOpenError:
    """Tests for CircuitOpenError exception."""

    def test_includes_provider_name(self):
        """Exception includes provider name in message and attribute."""
        err = CircuitOpenError("openai")

        assert "openai" in str(err)
        assert err.provider == "openai"

    def test_is_catchable(self):
        """CircuitOpenError can be caught and handled."""
        try:
            raise CircuitOpenError("gemini")
        except CircuitOpenError as e:
            assert e.provider == "gemini"
