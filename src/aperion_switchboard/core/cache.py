"""
Response Caching for The Switchboard.

Provides LRU cache with TTL for LLM responses to:
- Reduce duplicate API calls for identical prompts
- Lower latency for repeated queries
- Reduce costs by avoiding redundant provider calls

Features:
- In-memory LRU cache (no external dependencies)
- TTL-based expiration
- Cache key based on prompt + model + parameters
- Prometheus metrics for hit/miss rates
- Thread-safe implementation
"""

import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class CacheEntry:
    """A cached response with metadata."""

    value: dict[str, Any]
    created_at: float
    expires_at: float
    hit_count: int = 0

    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        return time.time() > self.expires_at

    @property
    def age_seconds(self) -> float:
        """Age of the cache entry in seconds."""
        return time.time() - self.created_at


@dataclass
class CacheConfig:
    """Configuration for the response cache."""

    # Maximum number of entries in cache
    max_size: int = 1000

    # Default TTL in seconds (5 minutes)
    default_ttl_seconds: float = 300.0

    # Enable/disable caching
    enabled: bool = True

    # Minimum prompt length to cache (avoid caching very short prompts)
    min_prompt_length: int = 10


@dataclass
class CacheStats:
    """Cache statistics."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0
    size: int = 0
    max_size: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "expirations": self.expirations,
            "size": self.size,
            "max_size": self.max_size,
            "hit_rate": round(self.hit_rate, 4),
        }


class ResponseCache:
    """
    LRU cache with TTL for LLM responses.

    Thread-safe implementation using OrderedDict for LRU eviction.
    """

    def __init__(self, config: CacheConfig | None = None):
        self.config = config or CacheConfig()
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = RLock()
        self._stats = CacheStats(max_size=self.config.max_size)

    def _generate_key(
        self,
        prompt: str,
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate a cache key from request parameters.

        Uses SHA256 hash of normalized parameters for consistent keys.
        """
        # Build canonical representation
        key_data = {
            "prompt": prompt.strip(),
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Add any additional parameters that affect output
        for k in sorted(kwargs.keys()):
            if k not in ("correlation_id", "request_id", "stream"):
                key_data[k] = kwargs[k]

        # Generate hash
        key_json = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_json.encode()).hexdigest()[:32]

    def get(
        self,
        prompt: str,
        model: str,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        """
        Get cached response if available and not expired.

        Args:
            prompt: The user prompt
            model: Model identifier
            **kwargs: Additional parameters used in key generation

        Returns:
            Cached response dict or None if miss
        """
        if not self.config.enabled:
            return None

        if len(prompt) < self.config.min_prompt_length:
            return None

        key = self._generate_key(prompt, model, **kwargs)

        with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._stats.misses += 1
                return None

            if entry.is_expired:
                # Remove expired entry
                del self._cache[key]
                self._stats.expirations += 1
                self._stats.misses += 1
                self._stats.size = len(self._cache)
                return None

            # Cache hit - move to end (most recently used)
            self._cache.move_to_end(key)
            entry.hit_count += 1
            self._stats.hits += 1

            logger.debug(
                "cache_hit",
                key=key[:8],
                age_seconds=round(entry.age_seconds, 2),
                hit_count=entry.hit_count,
            )

            return entry.value

    def set(
        self,
        prompt: str,
        model: str,
        response: dict[str, Any],
        ttl_seconds: float | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Cache a response.

        Args:
            prompt: The user prompt
            model: Model identifier
            response: The response to cache
            ttl_seconds: Optional TTL override
            **kwargs: Additional parameters used in key generation
        """
        if not self.config.enabled:
            return

        if len(prompt) < self.config.min_prompt_length:
            return

        key = self._generate_key(prompt, model, **kwargs)
        ttl = ttl_seconds or self.config.default_ttl_seconds
        now = time.time()

        entry = CacheEntry(
            value=response,
            created_at=now,
            expires_at=now + ttl,
        )

        with self._lock:
            # Evict oldest entries if at capacity
            while len(self._cache) >= self.config.max_size:
                self._cache.popitem(last=False)
                self._stats.evictions += 1

            self._cache[key] = entry
            self._stats.size = len(self._cache)

            logger.debug(
                "cache_set",
                key=key[:8],
                ttl_seconds=ttl,
                size=self._stats.size,
            )

    def invalidate(self, prompt: str, model: str, **kwargs: Any) -> bool:
        """
        Invalidate a specific cache entry.

        Returns True if entry was found and removed.
        """
        key = self._generate_key(prompt, model, **kwargs)

        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._stats.size = len(self._cache)
                return True
            return False

    def clear(self) -> int:
        """
        Clear all cache entries.

        Returns number of entries cleared.
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._stats.size = 0
            return count

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            self._stats.size = len(self._cache)
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                expirations=self._stats.expirations,
                size=self._stats.size,
                max_size=self._stats.max_size,
            )

    def cleanup_expired(self) -> int:
        """
        Remove all expired entries.

        Returns number of entries removed.
        """
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items() if entry.is_expired
            ]

            for key in expired_keys:
                del self._cache[key]
                self._stats.expirations += 1

            self._stats.size = len(self._cache)
            return len(expired_keys)


# Global cache instance
_response_cache: ResponseCache | None = None


def get_response_cache() -> ResponseCache:
    """Get the global response cache instance."""
    global _response_cache
    if _response_cache is None:
        _response_cache = ResponseCache()
    return _response_cache


def set_response_cache(cache: ResponseCache) -> None:
    """Set the global response cache (for testing/configuration)."""
    global _response_cache
    _response_cache = cache
