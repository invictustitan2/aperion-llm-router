"""Core routing logic, protocols, and fail-closed enforcement."""

from .cache import (
    CacheConfig,
    CacheStats,
    ResponseCache,
    get_response_cache,
    set_response_cache,
)
from .fail_closed import FailClosedError, check_fail_closed, is_echo_allowed
from .protocol import (
    LLMClient,
    LLMClientProtocol,
    ProviderError,
    ProviderHealth,
    ProviderInfo,
    ProviderNotConfiguredError,
    ProviderRateLimitError,
)
from .rate_limit import (
    RateLimitConfig,
    RateLimiter,
    RateLimitResult,
    get_rate_limiter,
    set_rate_limiter,
)
from .resilience import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitOpenError,
    CircuitState,
    RetryConfig,
    get_circuit_breaker,
    reset_circuit_breakers,
)
from .router import (
    LLMRouter,
    RoutingDecision,
    TaskType,
    get_router,
    reset_router,
    route_task,
)

__all__ = [
    # Protocols
    "LLMClient",
    "LLMClientProtocol",  # Duck-typing compatible protocol
    "ProviderInfo",
    "ProviderHealth",
    "ProviderError",
    "ProviderNotConfiguredError",
    "ProviderRateLimitError",
    # Router
    "LLMRouter",
    "RoutingDecision",
    "TaskType",
    "get_router",
    "reset_router",
    "route_task",
    # Fail-closed
    "FailClosedError",
    "is_echo_allowed",
    "check_fail_closed",
    # Resilience
    "RetryConfig",
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitState",
    "CircuitOpenError",
    "get_circuit_breaker",
    "reset_circuit_breakers",
    # Rate limiting
    "RateLimiter",
    "RateLimitConfig",
    "RateLimitResult",
    "get_rate_limiter",
    "set_rate_limiter",
    # Cache
    "ResponseCache",
    "CacheConfig",
    "CacheStats",
    "get_response_cache",
    "set_response_cache",
]
