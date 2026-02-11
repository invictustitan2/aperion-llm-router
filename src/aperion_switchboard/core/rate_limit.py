"""
Rate Limiting for The Switchboard.

Implements token bucket algorithm with:
- Per-key rate limits (API key specific quotas)
- Global rate limit fallback
- Configurable burst allowance
- Thread-safe in-memory storage

Constitution compliance:
- Protects against DoS (abuse prevention)
- Ensures fair resource allocation
- Logs rate limit events for observability
"""

import time
from dataclasses import dataclass, field
from threading import Lock
from typing import NamedTuple

import structlog

logger = structlog.get_logger(__name__)


class RateLimitResult(NamedTuple):
    """Result of a rate limit check."""

    allowed: bool
    remaining: int
    reset_at: float
    limit: int
    retry_after: float | None = None


@dataclass
class TokenBucket:
    """
    Token bucket rate limiter.

    Allows bursts up to capacity, refills at rate tokens per second.
    """

    capacity: int  # Max tokens (burst limit)
    refill_rate: float  # Tokens per second
    tokens: float = field(default=0.0, init=False)
    last_update: float = field(default_factory=time.monotonic, init=False)
    lock: Lock = field(default_factory=Lock, init=False)

    def __post_init__(self):
        self.tokens = float(self.capacity)

    def _refill(self, now: float) -> None:
        """Refill tokens based on elapsed time."""
        elapsed = now - self.last_update
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_update = now

    def consume(self, tokens: int = 1) -> RateLimitResult:
        """
        Attempt to consume tokens.

        Args:
            tokens: Number of tokens to consume (default 1 for request-based)

        Returns:
            RateLimitResult with allowed status and metadata
        """
        with self.lock:
            now = time.monotonic()
            self._refill(now)

            if self.tokens >= tokens:
                self.tokens -= tokens
                return RateLimitResult(
                    allowed=True,
                    remaining=int(self.tokens),
                    reset_at=now + (self.capacity - self.tokens) / self.refill_rate,
                    limit=self.capacity,
                )
            else:
                # Calculate time until enough tokens available
                tokens_needed = tokens - self.tokens
                retry_after = tokens_needed / self.refill_rate
                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_at=now + retry_after,
                    limit=self.capacity,
                    retry_after=retry_after,
                )


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""

    # Requests per minute
    requests_per_minute: int = 60

    # Burst allowance (max requests at once)
    burst_size: int = 10

    # Enable per-key limits
    per_key_enabled: bool = True

    # Per-key limits (if different from global)
    per_key_rpm: int | None = None
    per_key_burst: int | None = None

    @property
    def refill_rate(self) -> float:
        """Tokens per second."""
        return self.requests_per_minute / 60.0

    @property
    def per_key_refill_rate(self) -> float:
        """Per-key tokens per second."""
        rpm = self.per_key_rpm or self.requests_per_minute
        return rpm / 60.0


class RateLimiter:
    """
    In-memory rate limiter with per-key and global limits.

    Uses token bucket algorithm for smooth rate limiting with burst allowance.
    """

    def __init__(self, config: RateLimitConfig | None = None):
        self.config = config or RateLimitConfig()
        self._global_bucket = TokenBucket(
            capacity=self.config.burst_size,
            refill_rate=self.config.refill_rate,
        )
        self._key_buckets: dict[str, TokenBucket] = {}
        self._lock = Lock()

    def _get_key_bucket(self, key: str) -> TokenBucket:
        """Get or create bucket for API key."""
        with self._lock:
            if key not in self._key_buckets:
                burst = self.config.per_key_burst or self.config.burst_size
                self._key_buckets[key] = TokenBucket(
                    capacity=burst,
                    refill_rate=self.config.per_key_refill_rate,
                )
            return self._key_buckets[key]

    def check(self, key: str | None = None, tokens: int = 1) -> RateLimitResult:
        """
        Check if request is allowed.

        Args:
            key: API key for per-key limiting (None for anonymous)
            tokens: Number of tokens to consume

        Returns:
            RateLimitResult with allowed status
        """
        # Check per-key limit first if enabled and key provided
        if self.config.per_key_enabled and key:
            bucket = self._get_key_bucket(key)
            result = bucket.consume(tokens)
            if not result.allowed:
                logger.warning(
                    "rate_limit_exceeded",
                    key=key[:8] + "..." if len(key) > 8 else key,
                    remaining=result.remaining,
                    retry_after=result.retry_after,
                    limit_type="per_key",
                )
                return result

        # Check global limit
        result = self._global_bucket.consume(tokens)
        if not result.allowed:
            logger.warning(
                "rate_limit_exceeded",
                key=key[:8] + "..." if key and len(key) > 8 else key,
                remaining=result.remaining,
                retry_after=result.retry_after,
                limit_type="global",
            )

        return result

    def get_stats(self) -> dict:
        """Get rate limiter statistics."""
        return {
            "global": {
                "remaining": int(self._global_bucket.tokens),
                "capacity": self._global_bucket.capacity,
            },
            "per_key_count": len(self._key_buckets),
            "config": {
                "requests_per_minute": self.config.requests_per_minute,
                "burst_size": self.config.burst_size,
                "per_key_enabled": self.config.per_key_enabled,
            },
        }

    def reset(self) -> None:
        """Reset all buckets (for testing)."""
        with self._lock:
            self._global_bucket = TokenBucket(
                capacity=self.config.burst_size,
                refill_rate=self.config.refill_rate,
            )
            self._key_buckets.clear()


# Global rate limiter instance
_rate_limiter: RateLimiter | None = None


def get_rate_limiter() -> RateLimiter:
    """Get the global rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter


def set_rate_limiter(limiter: RateLimiter) -> None:
    """Set the global rate limiter (for testing/configuration)."""
    global _rate_limiter
    _rate_limiter = limiter
