"""
Base Provider - Common functionality for all providers.

Implements shared logic for HTTP clients, error handling, telemetry,
resilience patterns (retry + circuit breaker), and the LLMClient protocol.
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
    Base class for LLM providers with async HTTP and resilience patterns.

    Features:
    - Shared async HTTP client (connection pooling)
    - Retry with exponential backoff for transient errors
    - Circuit breaker for cascading failure protection
    - Correlation ID propagation for telemetry

    Subclasses must implement:
    - name property
    - is_configured property
    - _build_headers() method
    - _build_url() method
    - _build_payload() method
    - _parse_response() method
    """

    # Override in subclasses
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
        self._sync_client: httpx.Client | None = None
        self._retry_config = RetryConfig()
        self._circuit_breaker: CircuitBreaker | None = None

    @classmethod
    def set_shared_client(cls, client: httpx.AsyncClient) -> None:
        """Set shared async client (call during app startup)."""
        cls._shared_async_client = client

    @classmethod
    async def close_shared_client(cls) -> None:
        """Close shared async client (call during app shutdown)."""
        if cls._shared_async_client:
            await cls._shared_async_client.aclose()
            cls._shared_async_client = None

    def _get_sync_client(self) -> httpx.Client:
        """Get or create sync HTTP client (legacy/testing)."""
        if self._sync_client is None:
            self._sync_client = httpx.Client(timeout=self._timeout)
        return self._sync_client

    def _get_async_client(self) -> httpx.AsyncClient:
        """Get async HTTP client (shared or per-instance)."""
        if self._shared_async_client is not None:
            return self._shared_async_client
        # Fallback: create per-instance client (not recommended for production)
        limits = httpx.Limits(max_connections=100, max_keepalive_connections=20)
        return httpx.AsyncClient(limits=limits, timeout=self._timeout)

    def _get_circuit_breaker(self) -> CircuitBreaker:
        """Get circuit breaker for this provider."""
        if self._circuit_breaker is None:
            self._circuit_breaker = get_circuit_breaker(self.name)
        return self._circuit_breaker

    def _generate_request_id(self) -> str:
        """Generate unique request ID for tracing."""
        return f"llm_{uuid.uuid4().hex[:12]}"

    @abstractmethod
    def _build_headers(self, correlation_id: str | None = None) -> dict[str, str]:
        """Build HTTP headers for the request."""
        ...

    @abstractmethod
    def _build_url(self) -> str:
        """Build the API endpoint URL."""
        ...

    @abstractmethod
    def _build_payload(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        """Build the request payload."""
        ...

    @abstractmethod
    def _parse_response(self, data: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
        """
        Parse provider response.

        Returns:
            Tuple of (replies list, usage dict)
        """
        ...

    def chat(self, prompt: str, **kwargs: Any) -> Mapping[str, Any]:
        """
        Synchronous chat - uses sync client with basic error handling.

        For production use, prefer chat_async() which includes retry
        and circuit breaker patterns.
        """
        request_id = self._generate_request_id()
        correlation_id = kwargs.pop("correlation_id", None)
        start = time.monotonic()

        try:
            url = self._build_url()
            headers = self._build_headers(correlation_id)
            payload = self._build_payload(prompt, **kwargs)

            client = self._get_sync_client()
            response = client.post(url, headers=headers, json=payload)

            # Handle rate limiting
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                raise ProviderRateLimitError(
                    self.name,
                    retry_after=float(retry_after) if retry_after else None,
                )

            response.raise_for_status()
            data = response.json()
            elapsed = time.monotonic() - start

            replies, usage = self._parse_response(data)

            return {
                "replies": replies,
                "provider": self.name,
                "model": self._model,
                "usage": usage,
                "processing_time": elapsed,
                "request_id": request_id,
            }

        except ProviderRateLimitError:
            raise
        except httpx.HTTPStatusError as e:
            raise ProviderError(
                f"HTTP error from {self.name}: {e.response.status_code}",
                provider=self.name,
                status_code=e.response.status_code,
                recoverable=e.response.status_code >= 500,
            ) from e
        except Exception as e:
            logger.error(
                f"Provider {self.name} error: {e}",
                extra={"request_id": request_id, "error": str(e)},
            )
            raise ProviderError(
                f"Provider {self.name} failed: {e}",
                provider=self.name,
                recoverable=True,
            ) from e

    async def chat_async(self, prompt: str, **kwargs: Any) -> Mapping[str, Any]:
        """
        Async chat with retry and circuit breaker patterns.

        Features:
        - Exponential backoff with jitter for retries
        - Circuit breaker for cascading failure protection
        - Respects Retry-After headers from rate limiting
        - Correlation ID propagation

        Args:
            prompt: Input text to process
            **kwargs: Provider-specific parameters

        Returns:
            Structured mapping with replies, provider, usage, etc.

        Raises:
            CircuitOpenError: If circuit breaker is open
            ProviderError: If all retries exhausted
            ProviderRateLimitError: If rate limited after all retries
        """
        request_id = self._generate_request_id()
        correlation_id = kwargs.pop("correlation_id", None)
        circuit = self._get_circuit_breaker()

        # Check circuit breaker before any attempt
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

                # Handle retryable status codes (429, 5xx)
                if response.status_code in self._retry_config.retryable_status_codes:
                    retry_after = response.headers.get("Retry-After")
                    delay = self._retry_config.calculate_delay(
                        attempt, float(retry_after) if retry_after else None
                    )

                    if response.status_code == 429:
                        logger.warning(
                            f"Rate limited by {self.name}, attempt {attempt + 1}, "
                            f"waiting {delay:.1f}s",
                            extra={"request_id": request_id, "attempt": attempt + 1},
                        )

                    if attempt < self._retry_config.max_attempts - 1:
                        await asyncio.sleep(delay)
                        continue
                    else:
                        circuit.record_failure()
                        if response.status_code == 429:
                            raise ProviderRateLimitError(
                                self.name, retry_after=delay
                            )
                        raise ProviderError(
                            f"Server error from {self.name}: {response.status_code}",
                            provider=self.name,
                            status_code=response.status_code,
                            recoverable=True,
                        )

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

                # Non-retryable client errors (4xx except 429)
                if (
                    400 <= e.response.status_code < 500
                    and e.response.status_code != 429
                ):
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
                        f"Connection error to {self.name}, attempt {attempt + 1}, "
                        f"retrying in {delay:.1f}s",
                        extra={"request_id": request_id, "error": str(e)},
                    )
                    await asyncio.sleep(delay)
                    continue

            except Exception as e:
                last_error = e
                circuit.record_failure()
                logger.error(
                    f"Unexpected error from {self.name}: {e}",
                    extra={"request_id": request_id, "error": str(e)},
                )

        # All retries exhausted
        raise ProviderError(
            f"Provider {self.name} failed after {self._retry_config.max_attempts} "
            f"attempts: {last_error}",
            provider=self.name,
            recoverable=True,
        )

    def complete(self, prompt: str, **kwargs: Any) -> str:
        """Generate raw text completion, delegating to chat()."""
        result = self.chat(prompt, **kwargs)
        replies = result.get("replies", [])
        return replies[0] if replies else ""

    async def async_generate(self, prompt: str, **kwargs: Any) -> Mapping[str, Any]:
        """Async generation - now truly async with retry."""
        return await self.chat_async(prompt, **kwargs)

    def stream_generate(
        self, prompt: str, **kwargs: Any
    ) -> AsyncIterator[Mapping[str, Any]]:
        """Streaming generation - must be implemented by subclasses."""
        raise NotImplementedError(
            f"Streaming not implemented for {self.name} provider"
        )

    def health_check(self) -> ProviderHealth:
        """Check provider health via lightweight API call."""
        if not self.is_configured:
            return ProviderHealth.UNHEALTHY

        # Check circuit breaker state
        circuit = get_circuit_breaker(self.name)
        if not circuit.can_execute():
            return ProviderHealth.UNHEALTHY

        return ProviderHealth.HEALTHY

    async def health_check_async(self) -> ProviderHealth:
        """
        Async health check with actual provider ping.

        Makes a minimal request to verify the provider is reachable.
        Returns HEALTHY, UNHEALTHY, or UNKNOWN.
        """
        if not self.is_configured:
            return ProviderHealth.UNHEALTHY

        # Check circuit breaker first (fast path)
        circuit = get_circuit_breaker(self.name)
        if not circuit.can_execute():
            return ProviderHealth.UNHEALTHY

        # Try a minimal API call
        if self._shared_async_client is None:
            return ProviderHealth.UNKNOWN

        try:
            # Make a minimal request with short timeout
            url = self._build_url()
            headers = self._build_headers()

            # Use a very short prompt for health check
            payload = self._build_payload(
                prompt="Hi",
                max_tokens=1,
            )

            response = await self._shared_async_client.post(
                url,
                headers=headers,
                json=payload,
                timeout=5.0,  # Short timeout for health check
            )

            if response.status_code in (200, 201):
                return ProviderHealth.HEALTHY
            elif response.status_code in (401, 403):
                # Auth error - configured but credentials invalid
                return ProviderHealth.UNHEALTHY
            elif response.status_code == 429:
                # Rate limited but reachable
                return ProviderHealth.HEALTHY
            else:
                return ProviderHealth.UNHEALTHY

        except httpx.TimeoutException:
            return ProviderHealth.UNHEALTHY
        except Exception:
            return ProviderHealth.UNKNOWN

    def close(self) -> None:
        """Close the sync HTTP client."""
        if self._sync_client is not None:
            self._sync_client.close()
            self._sync_client = None

    def __enter__(self) -> "BaseProvider":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
