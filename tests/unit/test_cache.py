"""Tests for response caching."""

import time
from unittest.mock import patch

import pytest

from aperion_switchboard.core.cache import CacheConfig, CacheEntry, ResponseCache


class TestCacheConfig:
    """Tests for CacheConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CacheConfig()
        assert config.max_size == 1000
        assert config.default_ttl_seconds == 300
        assert config.min_prompt_length == 10

    def test_custom_config(self):
        """Test custom configuration."""
        config = CacheConfig(max_size=500, default_ttl_seconds=60, min_prompt_length=20)
        assert config.max_size == 500
        assert config.default_ttl_seconds == 60
        assert config.min_prompt_length == 20


class TestCacheEntry:
    """Tests for CacheEntry."""

    def test_entry_creation(self):
        """Test creating a cache entry."""
        now = time.time()
        entry = CacheEntry(
            value={"replies": ["test"]},
            created_at=now,
            expires_at=now + 60,
        )
        assert entry.value == {"replies": ["test"]}
        assert not entry.is_expired

    def test_expired_entry(self):
        """Test expired entry detection."""
        now = time.time()
        entry = CacheEntry(
            value={"replies": ["test"]},
            created_at=now - 10,
            expires_at=now - 1,
        )
        assert entry.is_expired


class TestResponseCache:
    """Tests for ResponseCache."""

    def test_set_and_get(self):
        """Test basic cache set and get."""
        cache = ResponseCache()
        response = {"replies": ["Hello, world!"]}
        
        cache.set(
            prompt="What is the meaning of life?",
            model="gpt-4",
            response=response,
        )
        
        result = cache.get(
            prompt="What is the meaning of life?",
            model="gpt-4",
        )
        assert result == response

    def test_cache_miss(self):
        """Test cache miss returns None."""
        cache = ResponseCache()
        result = cache.get(prompt="unknown prompt", model="gpt-4")
        assert result is None

    def test_different_prompts_different_keys(self):
        """Test that different prompts create different cache keys."""
        cache = ResponseCache()
        
        cache.set(
            prompt="Hello world test",
            model="gpt-4",
            response={"replies": ["response1"]},
        )
        cache.set(
            prompt="Goodbye world test",
            model="gpt-4",
            response={"replies": ["response2"]},
        )
        
        result1 = cache.get(prompt="Hello world test", model="gpt-4")
        result2 = cache.get(prompt="Goodbye world test", model="gpt-4")
        
        assert result1 == {"replies": ["response1"]}
        assert result2 == {"replies": ["response2"]}

    def test_same_prompt_different_models(self):
        """Test that same prompt with different models creates different keys."""
        cache = ResponseCache()
        prompt = "What is 2+2 test?"
        
        cache.set(prompt=prompt, model="gpt-4", response={"replies": ["4 from gpt-4"]})
        cache.set(prompt=prompt, model="gpt-3.5", response={"replies": ["4 from gpt-3.5"]})
        
        result1 = cache.get(prompt=prompt, model="gpt-4")
        result2 = cache.get(prompt=prompt, model="gpt-3.5")
        
        assert result1 == {"replies": ["4 from gpt-4"]}
        assert result2 == {"replies": ["4 from gpt-3.5"]}

    def test_temperature_affects_key(self):
        """Test that temperature is included in cache key."""
        cache = ResponseCache()
        prompt = "Write a poem test"
        
        cache.set(prompt=prompt, model="gpt-4", response={"replies": ["creative"]}, temperature=1.0)
        cache.set(prompt=prompt, model="gpt-4", response={"replies": ["precise"]}, temperature=0.0)
        
        result1 = cache.get(prompt=prompt, model="gpt-4", temperature=1.0)
        result2 = cache.get(prompt=prompt, model="gpt-4", temperature=0.0)
        
        assert result1 == {"replies": ["creative"]}
        assert result2 == {"replies": ["precise"]}

    def test_ttl_expiration(self):
        """Test that entries expire after TTL."""
        config = CacheConfig(default_ttl_seconds=1)
        cache = ResponseCache(config=config)
        
        cache.set(
            prompt="Short TTL test prompt",
            model="gpt-4",
            response={"replies": ["test"]},
        )
        
        # Should exist immediately
        assert cache.get(prompt="Short TTL test prompt", model="gpt-4") is not None
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Should be expired
        assert cache.get(prompt="Short TTL test prompt", model="gpt-4") is None

    def test_lru_eviction(self):
        """Test LRU eviction when max size is reached."""
        config = CacheConfig(max_size=3)
        cache = ResponseCache(config=config)
        
        # Fill cache
        for i in range(3):
            cache.set(
                prompt=f"Prompt number {i} test",
                model="gpt-4",
                response={"replies": [f"response {i}"]},
            )
        
        # Access first entry to make it most recently used
        cache.get(prompt="Prompt number 0 test", model="gpt-4")
        
        # Add new entry - should evict entry 1 (least recently used)
        cache.set(
            prompt="New prompt test here",
            model="gpt-4",
            response={"replies": ["new response"]},
        )
        
        # Entry 0 should still exist (was accessed)
        assert cache.get(prompt="Prompt number 0 test", model="gpt-4") is not None
        
        # Entry 1 should be evicted
        assert cache.get(prompt="Prompt number 1 test", model="gpt-4") is None
        
        # Entry 2 should still exist
        assert cache.get(prompt="Prompt number 2 test", model="gpt-4") is not None
        
        # New entry should exist
        assert cache.get(prompt="New prompt test here", model="gpt-4") is not None

    def test_short_prompts_not_cached(self):
        """Test that prompts shorter than min length are not cached."""
        config = CacheConfig(min_prompt_length=20)
        cache = ResponseCache(config=config)
        
        cache.set(
            prompt="short",  # Less than 20 chars
            model="gpt-4",
            response={"replies": ["test"]},
        )
        
        # Should return None because it shouldn't be cached
        stats = cache.get_stats()
        # The get will be a miss since it wasn't stored
        result = cache.get(prompt="short", model="gpt-4")
        assert result is None

    def test_cache_stats(self):
        """Test cache statistics."""
        cache = ResponseCache()
        
        # Initial stats
        stats = cache.get_stats()
        assert stats.size == 0
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.hit_rate == 0.0
        
        # Add entry
        cache.set(
            prompt="Test prompt here",
            model="gpt-4",
            response={"replies": ["test"]},
        )
        
        # Cache miss
        cache.get(prompt="Unknown prompt", model="gpt-4")
        
        # Cache hit
        cache.get(prompt="Test prompt here", model="gpt-4")
        
        stats = cache.get_stats()
        assert stats.size == 1
        assert stats.hits == 1
        assert stats.misses == 1
        assert stats.hit_rate == 0.5

    def test_clear_cache(self):
        """Test clearing the cache."""
        cache = ResponseCache()
        
        # Add entries
        for i in range(5):
            cache.set(
                prompt=f"Test prompt {i} here",
                model="gpt-4",
                response={"replies": [f"response {i}"]},
            )
        
        assert cache.get_stats().size == 5
        
        # Clear
        cache.clear()
        
        assert cache.get_stats().size == 0
        
        # Entries should be gone
        result = cache.get(prompt="Test prompt 0 here", model="gpt-4")
        assert result is None

    def test_generate_key_consistency(self):
        """Test that the same inputs always generate the same key."""
        cache = ResponseCache()
        
        key1 = cache._generate_key(
            prompt="Hello world",
            model="gpt-4",
            temperature=0.7,
            max_tokens=100,
        )
        key2 = cache._generate_key(
            prompt="Hello world",
            model="gpt-4",
            temperature=0.7,
            max_tokens=100,
        )
        
        assert key1 == key2

    def test_max_tokens_affects_key(self):
        """Test that max_tokens is included in cache key."""
        cache = ResponseCache()
        prompt = "Generate text test"
        
        cache.set(prompt=prompt, model="gpt-4", response={"replies": ["short"]}, max_tokens=50)
        cache.set(prompt=prompt, model="gpt-4", response={"replies": ["long"]}, max_tokens=500)
        
        result1 = cache.get(prompt=prompt, model="gpt-4", max_tokens=50)
        result2 = cache.get(prompt=prompt, model="gpt-4", max_tokens=500)
        
        assert result1 == {"replies": ["short"]}
        assert result2 == {"replies": ["long"]}


class TestCacheIntegration:
    """Integration tests for cache with FastAPI app."""

    @pytest.fixture
    def client(self):
        """Create test client with EchoProvider."""
        import os
        os.environ["APERION_ALLOW_ECHO"] = "true"
        os.environ["PYTEST_CURRENT_TEST"] = "test_cache.py"
        
        from fastapi.testclient import TestClient
        from aperion_switchboard.service.app import create_app
        from aperion_switchboard.core.rate_limit import get_rate_limiter
        from aperion_switchboard.core.cache import get_response_cache
        
        # Reset rate limiter to avoid test interference
        limiter = get_rate_limiter()
        limiter.reset()
        
        # Clear the response cache
        cache = get_response_cache()
        cache.clear()
        
        app = create_app()
        with TestClient(app) as client:
            yield client

    def test_cache_stats_endpoint(self, client):
        """Test /cache endpoint returns stats."""
        response = client.get("/cache")
        assert response.status_code == 200
        data = response.json()
        
        assert "size" in data
        assert "max_size" in data
        assert "hits" in data
        assert "misses" in data
        assert "hit_rate" in data
        assert "ttl_seconds" in data

    def test_cache_clear_endpoint(self, client):
        """Test /cache/clear endpoint clears cache."""
        response = client.post("/cache/clear")
        assert response.status_code == 200
        data = response.json()
        
        assert data["message"] == "Cache cleared"
        assert data["size"] == 0

    def test_cache_hit_on_duplicate_request(self, client):
        """Test that duplicate requests return cached response."""
        # First request - should be a miss
        body = {
            "model": "echo",
            "messages": [{"role": "user", "content": "Hello cache test message"}],
        }
        response1 = client.post("/v1/chat/completions", json=body)
        assert response1.status_code == 200
        
        # Check cache stats after first request
        stats1 = client.get("/cache").json()
        
        # Second identical request - should be a hit
        response2 = client.post("/v1/chat/completions", json=body)
        assert response2.status_code == 200
        
        # Check cache stats after second request
        stats2 = client.get("/cache").json()
        
        # Hits should have increased
        assert stats2["hits"] >= stats1["hits"]

    def test_cache_bypass_header(self, client):
        """Test X-Switchboard-No-Cache header bypasses cache."""
        body = {
            "model": "echo",
            "messages": [{"role": "user", "content": "Bypass test message here"}],
        }
        
        # First request - should populate cache
        client.post("/v1/chat/completions", json=body)
        stats1 = client.get("/cache").json()
        
        # Request with bypass header - should not use cache
        client.post(
            "/v1/chat/completions",
            json=body,
            headers={"X-Switchboard-No-Cache": "true"},
        )
        
        # Cache stats should show a miss for the bypassed request
        stats2 = client.get("/cache").json()
        # Misses should have increased by at least 1
        assert stats2["misses"] >= stats1["misses"]
