"""
Tests for rate limiting.

Validates:
- Token bucket algorithm behavior
- Per-key rate limiting
- Global rate limiting
- Rate limit headers in responses
- 429 response on limit exceeded
"""

import time

import pytest
from fastapi.testclient import TestClient

from aperion_switchboard.core.rate_limit import (
    RateLimitConfig,
    RateLimiter,
    TokenBucket,
)
from aperion_switchboard.service.app import create_app


class TestTokenBucket:
    """Tests for TokenBucket algorithm."""

    def test_initial_tokens_at_capacity(self):
        """Bucket starts with full capacity."""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        assert bucket.tokens == 10

    def test_consume_reduces_tokens(self):
        """Consuming tokens reduces available count."""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        result = bucket.consume(3)
        assert result.allowed is True
        assert result.remaining == 7

    def test_consume_respects_capacity(self):
        """Cannot consume more tokens than available."""
        bucket = TokenBucket(capacity=5, refill_rate=1.0)
        
        # Consume all tokens
        for _ in range(5):
            result = bucket.consume(1)
            assert result.allowed is True
        
        # Next request should fail
        result = bucket.consume(1)
        assert result.allowed is False
        assert result.remaining == 0
        assert result.retry_after is not None
        assert result.retry_after > 0

    def test_refill_over_time(self):
        """Tokens refill based on elapsed time."""
        bucket = TokenBucket(capacity=10, refill_rate=10.0)  # 10 tokens/second
        
        # Consume all
        for _ in range(10):
            bucket.consume(1)
        
        assert bucket.tokens < 1
        
        # Wait a bit and check refill
        time.sleep(0.2)  # Should refill ~2 tokens
        result = bucket.consume(1)
        assert result.allowed is True

    def test_refill_caps_at_capacity(self):
        """Tokens don't exceed capacity after refill."""
        bucket = TokenBucket(capacity=5, refill_rate=100.0)  # Fast refill
        
        # Consume one
        bucket.consume(1)
        time.sleep(0.1)  # Would refill 10 tokens, but capped at 5
        
        result = bucket.consume(1)
        assert result.remaining <= 4  # At most capacity - 1


class TestRateLimiter:
    """Tests for RateLimiter with per-key and global limits."""

    def test_allows_requests_within_limit(self):
        """Requests within rate limit are allowed."""
        config = RateLimitConfig(requests_per_minute=60, burst_size=10)
        limiter = RateLimiter(config)
        
        for _ in range(10):  # Within burst
            result = limiter.check(key="test-key")
            assert result.allowed is True

    def test_blocks_requests_exceeding_burst(self):
        """Requests exceeding burst limit are blocked."""
        config = RateLimitConfig(requests_per_minute=60, burst_size=5)
        limiter = RateLimiter(config)
        
        # Exhaust burst
        for _ in range(5):
            result = limiter.check(key="test-key")
            assert result.allowed is True
        
        # Next should be blocked
        result = limiter.check(key="test-key")
        assert result.allowed is False

    def test_per_key_isolation(self):
        """Different keys have independent limits."""
        config = RateLimitConfig(
            requests_per_minute=600,
            burst_size=10,  # Higher global burst
            per_key_burst=3,  # Lower per-key burst
        )
        limiter = RateLimiter(config)
        
        # Exhaust key1's limit
        for _ in range(3):
            limiter.check(key="key1")
        
        # key1 should be blocked
        result = limiter.check(key="key1")
        assert result.allowed is False
        
        # key2 should still have capacity
        result = limiter.check(key="key2")
        assert result.allowed is True

    def test_global_limit_applies_to_all(self):
        """Global limit applies across all keys."""
        config = RateLimitConfig(
            requests_per_minute=60,
            burst_size=5,  # Global burst
            per_key_burst=10,  # Per-key burst higher
        )
        limiter = RateLimiter(config)
        
        # Multiple keys, but global limit should kick in
        for i in range(5):
            result = limiter.check(key=f"key{i}")
            assert result.allowed is True
        
        # Global limit exhausted
        result = limiter.check(key="key999")
        assert result.allowed is False

    def test_anonymous_requests_use_global(self):
        """Requests without key use global limit only."""
        config = RateLimitConfig(burst_size=3)
        limiter = RateLimiter(config)
        
        for _ in range(3):
            result = limiter.check(key=None)
            assert result.allowed is True
        
        result = limiter.check(key=None)
        assert result.allowed is False

    def test_reset_clears_all_limits(self):
        """Reset clears all buckets."""
        config = RateLimitConfig(burst_size=2)
        limiter = RateLimiter(config)
        
        # Exhaust limits
        limiter.check(key="test")
        limiter.check(key="test")
        result = limiter.check(key="test")
        assert result.allowed is False
        
        # Reset
        limiter.reset()
        
        # Should be allowed again
        result = limiter.check(key="test")
        assert result.allowed is True

    def test_get_stats_returns_info(self):
        """Stats endpoint returns useful information."""
        config = RateLimitConfig(requests_per_minute=100, burst_size=10)
        limiter = RateLimiter(config)
        
        limiter.check(key="key1")
        limiter.check(key="key2")
        
        stats = limiter.get_stats()
        assert stats["per_key_count"] == 2
        assert stats["config"]["requests_per_minute"] == 100


class TestRateLimitMiddleware:
    """Tests for rate limit middleware integration."""

    @pytest.fixture
    def limited_client(self):
        """Create client with aggressive rate limiting."""
        from aperion_switchboard.core.rate_limit import set_rate_limiter
        
        config = RateLimitConfig(requests_per_minute=60, burst_size=3)
        limiter = RateLimiter(config)
        set_rate_limiter(limiter)
        
        app = create_app()
        client = TestClient(app)
        
        yield client
        
        # Reset after test
        limiter.reset()

    def test_rate_limit_headers_present(self, limited_client):
        """Rate limit headers are included in response."""
        response = limited_client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "test"}]}
        )
        
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Reset" in response.headers

    def test_rate_limit_exceeded_returns_429(self, limited_client):
        """Exceeding rate limit returns 429."""
        # Make requests up to burst limit
        # Note: We may get 200 (success) or 503 (no provider) - both count as "not rate limited"
        for _ in range(3):
            response = limited_client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "test"}]}
            )
            assert response.status_code in (200, 503)  # Not rate limited
        
        # Next request should be rate limited (429 takes precedence)
        response = limited_client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "test"}]}
        )
        
        assert response.status_code == 429
        assert "Retry-After" in response.headers
        
        data = response.json()
        assert "error" in data
        assert "rate limit" in data["error"]["message"].lower()

    def test_health_endpoint_bypasses_rate_limit(self, limited_client):
        """Health endpoints are not rate limited."""
        # Exhaust rate limit
        for _ in range(5):
            limited_client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "test"}]}
            )
        
        # Health should still work
        response = limited_client.get("/health")
        assert response.status_code == 200

    def test_circuits_endpoint_bypasses_rate_limit(self, limited_client):
        """Circuits endpoint is not rate limited."""
        # Exhaust rate limit
        for _ in range(5):
            limited_client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "test"}]}
            )
        
        # Circuits should still work
        response = limited_client.get("/circuits")
        assert response.status_code == 200
