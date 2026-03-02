"""
LLM Provider Protocol - Abstract Base Class for all providers.

This is the contract that all providers must implement.
Maintains exact compatibility with the aperion-legendary-ai LLMClient protocol.

Provides both ABC (for internal implementations) and Protocol (for duck typing).
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class ProviderHealth(Enum):
    """Provider health status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ProviderInfo:
    """Metadata about an available provider."""

    name: str
    description: str
    available: bool
    configured: bool
    health: ProviderHealth = ProviderHealth.UNKNOWN
    error: str | None = None


class LLMClient(ABC):
    """
    Abstract base class for LLM provider implementations.

    All providers must implement this interface to ensure consistent
    behavior across The Switchboard. This maintains exact compatibility
    with the aperion-legendary-ai LLMClient protocol.

    Thread Safety: Implementations should be thread-safe for concurrent
    requests across multiple agents (Sentinel, AR, Aether).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name identifier (e.g., 'openai', 'gemini', 'workers_ai')."""
        ...

    @property
    @abstractmethod
    def is_configured(self) -> bool:
        """Whether this provider has sufficient configuration to make API calls."""
        ...

    @abstractmethod
    def chat(self, prompt: str, **kwargs: Any) -> Mapping[str, Any]:
        """
        Generate structured chat response.

        Args:
            prompt: Input text to process
            **kwargs: Provider-specific parameters (temperature, max_tokens, etc.)

        Returns:
            Structured mapping with at minimum:
                - replies: list[str] - Generated responses
                - provider: str - Provider name
                - processing_time: float - Time in seconds
                - usage: dict - Token usage stats (if available)

        Raises:
            ProviderError: If the API call fails
        """
        ...

    @abstractmethod
    def complete(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate raw text completion.

        Args:
            prompt: Input text to complete
            **kwargs: Provider-specific parameters

        Returns:
            Raw text completion string
        """
        ...

    @abstractmethod
    async def async_generate(self, prompt: str, **kwargs: Any) -> Mapping[str, Any]:
        """
        Asynchronous generation API.

        Same contract as chat() but async-native for high-concurrency scenarios.
        """
        ...

    @abstractmethod
    def stream_generate(
        self, prompt: str, **kwargs: Any
    ) -> AsyncIterator[Mapping[str, Any]]:
        """
        Streaming generation returning an async iterator of chunks.

        Each chunk should have:
            - chunk: str - The text content
            - done: bool - Whether this is the final chunk
        """
        ...

    def health_check(self) -> ProviderHealth:
        """
        Check provider health status.

        Default implementation returns UNKNOWN. Providers may override
        to implement actual health checking (e.g., lightweight API call).
        """
        if not self.is_configured:
            return ProviderHealth.UNHEALTHY
        return ProviderHealth.UNKNOWN

    def get_info(self) -> ProviderInfo:
        """Get provider metadata."""
        return ProviderInfo(
            name=self.name,
            description=self.__class__.__doc__ or f"{self.name} provider",
            available=self.is_configured,
            configured=self.is_configured,
            health=self.health_check(),
        )


class ProviderError(Exception):
    """Base exception for provider errors."""

    def __init__(
        self,
        message: str,
        provider: str,
        recoverable: bool = True,
        status_code: int | None = None,
    ):
        super().__init__(message)
        self.provider = provider
        self.recoverable = recoverable
        self.status_code = status_code


class ProviderNotConfiguredError(ProviderError):
    """Raised when a required provider is not configured."""

    def __init__(self, provider: str, missing: list[str] | None = None):
        missing_str = f" (missing: {', '.join(missing)})" if missing else ""
        super().__init__(
            f"Provider '{provider}' is not configured{missing_str}",
            provider=provider,
            recoverable=False,
        )
        self.missing = missing or []


class ProviderRateLimitError(ProviderError):
    """Raised when provider rate limit is exceeded."""

    def __init__(self, provider: str, retry_after: float | None = None):
        super().__init__(
            f"Rate limit exceeded for provider '{provider}'",
            provider=provider,
            recoverable=True,
            status_code=429,
        )
        self.retry_after = retry_after


# Protocol alias for duck typing compatibility with aperion-legendary-ai
@runtime_checkable
class LLMClientProtocol(Protocol):
    """
    Protocol version of LLMClient for duck typing compatibility.

    Use this when you need to accept any object that implements the
    LLMClient interface without requiring explicit inheritance.

    Example:
        def process(provider: LLMClientProtocol) -> str:
            return provider.chat("hello")["replies"][0]
    """

    @property
    def name(self) -> str:
        """Provider name identifier."""
        ...

    def chat(self, prompt: str, **kwargs: Any) -> Mapping[str, Any]:
        """Generate structured chat response."""
        ...

    def complete(self, prompt: str, **kwargs: Any) -> str:
        """Generate raw text completion."""
        ...

    async def async_generate(self, prompt: str, **kwargs: Any) -> Mapping[str, Any]:
        """Asynchronous generation API."""
        ...

    def stream_generate(
        self, prompt: str, **kwargs: Any
    ) -> AsyncIterator[Mapping[str, Any]]:
        """Streaming generation returning an async iterator of chunks."""
        ...
