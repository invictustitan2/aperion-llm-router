"""
Resilience patterns for The Switchboard.

Implements retry, circuit breaker, and timeout patterns
for robust provider communication.
"""

import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    jitter_factor: float = 0.25
    retryable_status_codes: frozenset[int] = field(
        default_factory=lambda: frozenset({429, 500, 502, 503, 504})
    )

    def calculate_delay(
        self, attempt: int, retry_after: float | None = None
    ) -> float:
        """Calculate delay with exponential backoff and jitter."""
        if retry_after is not None:
            return retry_after

        delay = min(self.base_delay * (2**attempt), self.max_delay)
        jitter = random.uniform(0, self.jitter_factor * delay)
        return delay + jitter


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    failure_threshold: int = 5
    success_threshold: int = 2
    timeout_seconds: float = 60.0
    half_open_max_calls: int = 3


@dataclass
class CircuitBreaker:
    """
    Per-provider circuit breaker.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Provider is failing, reject requests immediately
    - HALF_OPEN: Testing recovery with limited requests

    Transitions:
    - CLOSED → OPEN: When failure_threshold failures occur
    - OPEN → HALF_OPEN: After timeout_seconds pass
    - HALF_OPEN → CLOSED: When success_threshold successes occur
    - HALF_OPEN → OPEN: On any failure
    """

    name: str
    config: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: datetime | None = None
    half_open_calls: int = 0
    _last_state_change: float = field(default_factory=time.monotonic)

    def can_execute(self) -> bool:
        """Check if a request can proceed."""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            if self.last_failure_time:
                elapsed = datetime.now() - self.last_failure_time
                if elapsed > timedelta(seconds=self.config.timeout_seconds):
                    self._transition_to_half_open()
                    # Fall through to HALF_OPEN logic to count this call
                else:
                    return False
            else:
                return False

        # HALF_OPEN: Allow limited probes
        if self.half_open_calls < self.config.half_open_max_calls:
            self.half_open_calls += 1
            return True
        return False

    def record_success(self) -> None:
        """Record a successful call."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self._transition_to_closed()
        else:
            # In CLOSED state, success resets failure count
            self.failure_count = 0

    def record_failure(self) -> None:
        """Record a failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.state == CircuitState.HALF_OPEN:
            self._transition_to_open()
        elif self.failure_count >= self.config.failure_threshold:
            self._transition_to_open()

    def reset(self) -> None:
        """Reset circuit breaker to initial state."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.half_open_calls = 0
        self._last_state_change = time.monotonic()

    def get_stats(self) -> dict:
        """Get circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": (
                self.last_failure_time.isoformat() if self.last_failure_time else None
            ),
            "half_open_calls": self.half_open_calls,
        }

    def _transition_to_open(self) -> None:
        logger.warning(
            f"Circuit breaker {self.name}: OPEN (failures={self.failure_count})"
        )
        self.state = CircuitState.OPEN
        self.half_open_calls = 0
        self.success_count = 0
        self._last_state_change = time.monotonic()

    def _transition_to_half_open(self) -> None:
        logger.info(f"Circuit breaker {self.name}: HALF_OPEN (probing)")
        self.state = CircuitState.HALF_OPEN
        self.half_open_calls = 0
        self.success_count = 0
        self._last_state_change = time.monotonic()

    def _transition_to_closed(self) -> None:
        logger.info(f"Circuit breaker {self.name}: CLOSED (recovered)")
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self._last_state_change = time.monotonic()


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""

    def __init__(self, provider: str):
        super().__init__(f"Circuit breaker open for provider: {provider}")
        self.provider = provider


# Global circuit breakers registry
_circuit_breakers: dict[str, CircuitBreaker] = {}


def get_circuit_breaker(provider_name: str) -> CircuitBreaker:
    """Get or create circuit breaker for a provider."""
    if provider_name not in _circuit_breakers:
        _circuit_breakers[provider_name] = CircuitBreaker(name=provider_name)
    return _circuit_breakers[provider_name]


def reset_circuit_breakers() -> None:
    """Reset all circuit breakers (for testing)."""
    _circuit_breakers.clear()


def get_all_circuit_stats() -> dict[str, dict]:
    """Get stats for all circuit breakers."""
    return {name: cb.get_stats() for name, cb in _circuit_breakers.items()}
