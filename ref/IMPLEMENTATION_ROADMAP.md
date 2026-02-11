# Implementation Roadmap: P0 Fixes

> Ready-to-implement code for critical gaps

## Phase 1: Async HTTP Client (ISSUE-001)

### Step 1: Add Resilience Module

Create `src/switchboard/core/resilience.py`:

```python
"""
Resilience patterns for The Switchboard.

Implements retry, circuit breaker, and timeout patterns
for robust provider communication.
"""

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    jitter_factor: float = 0.25
    retryable_status_codes: frozenset[int] = frozenset({429, 500, 502, 503, 504})

    def calculate_delay(self, attempt: int, retry_after: float | None = None) -> float:
        """Calculate delay with exponential backoff and jitter."""
        if retry_after is not None:
            return retry_after
        
        delay = min(self.base_delay * (2 ** attempt), self.max_delay)
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
    """Per-provider circuit breaker."""
    name: str
    config: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: datetime | None = None
    half_open_calls: int = 0

    def can_execute(self) -> bool:
        """Check if a request can proceed."""
        if self.state == CircuitState.CLOSED:
            return True
        
        if self.state == CircuitState.OPEN:
            # Check if timeout has passed
            if self.last_failure_time:
                elapsed = datetime.now() - self.last_failure_time
                if elapsed > timedelta(seconds=self.config.timeout_seconds):
                    self._transition_to_half_open()
                    return True
            return False
        
        # HALF_OPEN: Allow limited probes
        return self.half_open_calls < self.config.half_open_max_calls

    def record_success(self) -> None:
        """Record a successful call."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self._transition_to_closed()
        else:
            self.failure_count = 0

    def record_failure(self) -> None:
        """Record a failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.state == CircuitState.HALF_OPEN:
            self._transition_to_open()
        elif self.failure_count >= self.config.failure_threshold:
            self._transition_to_open()

    def _transition_to_open(self) -> None:
        logger.warning(f"Circuit breaker {self.name}: OPEN (failures={self.failure_count})")
        self.state = CircuitState.OPEN
        self.half_open_calls = 0
        self.success_count = 0

    def _transition_to_half_open(self) -> None:
        logger.info(f"Circuit breaker {self.name}: HALF_OPEN (probing)")
        self.state = CircuitState.HALF_OPEN
        self.half_open_calls = 0
        self.success_count = 0

    def _transition_to_closed(self) -> None:
        logger.info(f"Circuit breaker {self.name}: CLOSED (recovered)")
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0


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
```


### Step 2: Update Base Provider with Async + Retry

Replace `src/switchboard/providers/base.py`:

```python
"""
Base Provider - Async HTTP client with resilience patterns.
"""

import asyncio
import logging
import os
import time
import uuid
from abc import abstractmethod
from collections.abc import AsyncIterator, Mapping
from typing import Any

import httpx

from ..core.protocol import (
    LLMClient,
    ProviderError,
    ProviderHealth,
    ProviderRateLimitError,
)
from ..core.resilience import (
    CircuitBreaker,
    CircuitOpenError,
    RetryConfig,
    get_circuit_breaker,
)

logger = logging.getLogger(__name__)


class BaseProvider(LLMClient):
    """
    Base class for LLM providers with async HTTP and resilience.
    """

    ENV_PREFIX: str = "LLM"
    DEFAULT_BASE_URL: str = ""
    DEFAULT_MODEL: str = ""
    DEFAULT_TIMEOUT: int = 30

    # Shared async client (set during app lifespan)
    _shared_async_client: httpx.AsyncClient | None = None

    def __init__(self) -> None:
        prefix = self.ENV_PREFIX
        self._api_key: str = os.environ.get(f"{prefix}_API_KEY", "")
        self._base_url: str = os.environ.get(
            f"{prefix}_BASE_URL", self.DEFAULT_BASE_URL
        )
        self._model: str = os.environ.get(f"{prefix}_MODEL", self.DEFAULT_MODEL)
        self._timeout: int = int(
            os.environ.get(f"{prefix}_TIMEOUT", str(self.DEFAULT_TIMEOUT))
        )
        self._retry_config = RetryConfig()
        self._circuit_breaker: CircuitBreaker | None = None

    @classmethod
    def set_shared_client(cls, client: httpx.AsyncClient) -> None:
        """Set shared async client (call during app startup)."""
        cls._shared_async_client = client

    @classmethod
    async def close_shared_client(cls) -> None:
        """Close shared client (call during app shutdown)."""
        if cls._shared_async_client:
            await cls._shared_async_client.aclose()
            cls._shared_async_client = None

    def _get_circuit_breaker(self) -> CircuitBreaker:
        """Get circuit breaker for this provider."""
        if self._circuit_breaker is None:
            self._circuit_breaker = get_circuit_breaker(self.name)
        return self._circuit_breaker

    def _get_async_client(self) -> httpx.AsyncClient:
        """Get async HTTP client."""
        if self._shared_async_client is None:
            # Fallback: create per-provider client (not recommended)
            limits = httpx.Limits(max_connections=100, max_keepalive_connections=20)
            return httpx.AsyncClient(limits=limits, timeout=self._timeout)
        return self._shared_async_client

    @abstractmethod
    def _build_headers(self, correlation_id: str | None = None) -> dict[str, str]:
        ...

    @abstractmethod
    def _build_url(self) -> str:
        ...

    @abstractmethod
    def _build_payload(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        ...

    @abstractmethod
    def _parse_response(self, data: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
        ...

    async def chat_async(self, prompt: str, **kwargs: Any) -> Mapping[str, Any]:
        """
        Async chat with retry and circuit breaker.
        """
        request_id = f"llm_{uuid.uuid4().hex[:12]}"
        correlation_id = kwargs.pop("correlation_id", None)
        circuit = self._get_circuit_breaker()

        # Check circuit breaker
        if not circuit.can_execute():
            raise CircuitOpenError(self.name)

        url = self._build_url()
        headers = self._build_headers(correlation_id)
        payload = self._build_payload(prompt, **kwargs)
        client = self._get_async_client()

        last_error: Exception | None = None
        
        for attempt in range(self._retry_config.max_attempts):
            start = time.monotonic()
            
            try:
                response = await client.post(url, headers=headers, json=payload)
                
                # Check for retryable status codes
                if response.status_code in self._retry_config.retryable_status_codes:
                    retry_after = response.headers.get("Retry-After")
                    delay = self._retry_config.calculate_delay(
                        attempt, 
                        float(retry_after) if retry_after else None
                    )
                    
                    if response.status_code == 429:
                        logger.warning(
                            f"Rate limited by {self.name}, waiting {delay:.1f}s",
                            extra={"attempt": attempt + 1, "request_id": request_id}
                        )
                    
                    if attempt < self._retry_config.max_attempts - 1:
                        await asyncio.sleep(delay)
                        continue
                    else:
                        raise ProviderRateLimitError(self.name, retry_after=delay)

                response.raise_for_status()
                data = response.json()
                elapsed = time.monotonic() - start

                replies, usage = self._parse_response(data)
                
                # Record success
                circuit.record_success()

                return {
                    "replies": replies,
                    "provider": self.name,
                    "model": self._model,
                    "usage": usage,
                    "processing_time": elapsed,
                    "request_id": request_id,
                }

            except httpx.HTTPStatusError as e:
                last_error = e
                circuit.record_failure()
                
                # Non-retryable client errors
                if 400 <= e.response.status_code < 500 and e.response.status_code != 429:
                    raise ProviderError(
                        f"Client error from {self.name}: {e.response.status_code}",
                        provider=self.name,
                        status_code=e.response.status_code,
                        recoverable=False,
                    ) from e

            except (httpx.ConnectError, httpx.TimeoutException) as e:
                last_error = e
                circuit.record_failure()
                
                if attempt < self._retry_config.max_attempts - 1:
                    delay = self._retry_config.calculate_delay(attempt)
                    logger.warning(
                        f"Connection error to {self.name}, retrying in {delay:.1f}s",
                        extra={"attempt": attempt + 1, "error": str(e)}
                    )
                    await asyncio.sleep(delay)

        # All retries exhausted
        raise ProviderError(
            f"Provider {self.name} failed after {self._retry_config.max_attempts} attempts: {last_error}",
            provider=self.name,
            recoverable=True,
        )

    def chat(self, prompt: str, **kwargs: Any) -> Mapping[str, Any]:
        """Sync wrapper for backwards compatibility."""
        return asyncio.get_event_loop().run_until_complete(
            self.chat_async(prompt, **kwargs)
        )

    async def async_generate(self, prompt: str, **kwargs: Any) -> Mapping[str, Any]:
        """Async generation - now truly async."""
        return await self.chat_async(prompt, **kwargs)

    # ... rest of methods
```


### Step 3: Update App Lifespan

In `src/switchboard/service/app.py`, update lifespan:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - startup and shutdown."""
    global _providers, _router

    configure_structlog()
    logger.info("Starting The Switchboard...")

    # Create shared async client with proper limits
    limits = httpx.Limits(max_connections=100, max_keepalive_connections=20)
    async_client = httpx.AsyncClient(limits=limits, timeout=30.0)
    
    # Set shared client for all providers
    from ..providers.base import BaseProvider
    BaseProvider.set_shared_client(async_client)

    # Initialize providers
    _providers = _init_providers()

    # ... fail-closed check ...

    _router = _init_router(_providers)

    yield

    # Shutdown
    logger.info("Shutting down The Switchboard...")
    await BaseProvider.close_shared_client()
    for provider in _providers.values():
        if hasattr(provider, "close"):
            provider.close()
```


### Step 4: Update Endpoint to Use Async

```python
async def chat_completions(
    request: Request,
    body: ChatCompletionRequest,
    x_aperion_task_type: str | None = Header(default=None, alias="X-Aperion-Task-Type"),
) -> ChatCompletionResponse:
    # ... setup ...

    try:
        provider, decision = _router.get_provider(task_type, fallback=True)

        prompt = body.get_prompt()
        kwargs = body.to_provider_kwargs()
        kwargs["correlation_id"] = correlation_id

        if system_prompt := body.get_system_prompt():
            kwargs["persona"] = system_prompt

        # Use async method
        result = await provider.chat_async(prompt, **kwargs)
        
        # ... rest unchanged ...
```


---

## Phase 2: Streaming Support

Add to `src/switchboard/service/app.py`:

```python
from fastapi.responses import StreamingResponse

async def stream_chat_completion(
    provider: LLMClient,
    prompt: str,
    decision: RoutingDecision,
    request_id: str,
    **kwargs: Any,
):
    """Generate SSE stream from provider."""
    created = int(time.time())
    
    try:
        async for chunk in provider.stream_generate(prompt, **kwargs):
            if chunk.get("done"):
                # Final chunk
                yield "data: [DONE]\n\n"
                return
            
            content = chunk.get("chunk", "")
            if content:
                sse_data = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": kwargs.get("model", "gpt-4.1-mini"),
                    "choices": [{
                        "index": 0,
                        "delta": {"content": content},
                        "finish_reason": None,
                    }],
                }
                yield f"data: {json.dumps(sse_data)}\n\n"
                
    except Exception as e:
        error_data = {
            "error": {
                "message": str(e),
                "type": "stream_error",
            }
        }
        yield f"data: {json.dumps(error_data)}\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest, ...):
    # ... setup ...
    
    if body.stream:
        return StreamingResponse(
            stream_chat_completion(provider, prompt, decision, request_id, **kwargs),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Correlation-ID": correlation_id,
            },
        )
    
    # ... non-streaming path ...
```


---

## Testing the Changes

Add `tests/unit/test_resilience.py`:

```python
"""Tests for resilience patterns."""

import pytest
from switchboard.core.resilience import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    RetryConfig,
)


class TestRetryConfig:
    def test_exponential_backoff(self):
        config = RetryConfig(base_delay=1.0, jitter_factor=0.0)
        
        assert config.calculate_delay(0) == 1.0
        assert config.calculate_delay(1) == 2.0
        assert config.calculate_delay(2) == 4.0
        assert config.calculate_delay(3) == 8.0

    def test_max_delay_cap(self):
        config = RetryConfig(base_delay=1.0, max_delay=10.0, jitter_factor=0.0)
        
        assert config.calculate_delay(10) == 10.0  # Capped

    def test_retry_after_takes_precedence(self):
        config = RetryConfig()
        
        assert config.calculate_delay(0, retry_after=5.0) == 5.0


class TestCircuitBreaker:
    def test_starts_closed(self):
        cb = CircuitBreaker(name="test")
        assert cb.state == CircuitState.CLOSED
        assert cb.can_execute() is True

    def test_opens_after_threshold_failures(self):
        cb = CircuitBreaker(
            name="test",
            config=CircuitBreakerConfig(failure_threshold=3)
        )
        
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED
        
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.can_execute() is False

    def test_success_resets_failure_count(self):
        cb = CircuitBreaker(
            name="test",
            config=CircuitBreakerConfig(failure_threshold=3)
        )
        
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        
        assert cb.failure_count == 0
        assert cb.state == CircuitState.CLOSED
```


---

## Verification Checklist

After implementing:

- [ ] All existing tests still pass
- [ ] New resilience tests pass
- [ ] Manual test: Provider returns 429 → request retries
- [ ] Manual test: Provider down → circuit opens → fast failure
- [ ] Manual test: Streaming endpoint works
- [ ] Load test: 100 concurrent requests don't block
