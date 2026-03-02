"""
Prometheus Metrics for The Switchboard.

Provides observability via:
- Standard HTTP metrics (request count, latency, status codes)
- Custom LLM metrics (tokens, provider usage, routing decisions)
- Circuit breaker state metrics
- Rate limit metrics

Endpoint: GET /metrics
"""

from prometheus_client import Counter, Gauge, Histogram, Info

# Info metric for service metadata
SWITCHBOARD_INFO = Info(
    "switchboard",
    "The Switchboard service metadata"
)

# Request metrics
REQUESTS_TOTAL = Counter(
    "switchboard_requests_total",
    "Total number of LLM requests",
    ["provider", "task_type", "status"]
)

REQUEST_LATENCY = Histogram(
    "switchboard_request_duration_seconds",
    "Request latency in seconds",
    ["provider", "task_type"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]
)

# Token metrics
TOKENS_TOTAL = Counter(
    "switchboard_tokens_total",
    "Total tokens processed",
    ["provider", "token_type"]  # token_type: prompt, completion
)

# Cost metrics (estimated)
COST_USD_TOTAL = Counter(
    "switchboard_cost_usd_total",
    "Estimated cost in USD",
    ["provider"]
)

# Provider health
PROVIDER_HEALTH = Gauge(
    "switchboard_provider_health",
    "Provider health status (1=healthy, 0=unhealthy, -1=unknown)",
    ["provider"]
)

# Circuit breaker state
CIRCUIT_STATE = Gauge(
    "switchboard_circuit_state",
    "Circuit breaker state (0=closed, 1=open, 2=half_open)",
    ["provider"]
)

CIRCUIT_FAILURES = Counter(
    "switchboard_circuit_failures_total",
    "Total circuit breaker failures",
    ["provider"]
)

# Rate limiting
RATE_LIMIT_HITS = Counter(
    "switchboard_rate_limit_hits_total",
    "Total rate limit rejections",
    ["limit_type"]  # global, per_key
)

# Routing decisions
ROUTING_DECISIONS = Counter(
    "switchboard_routing_decisions_total",
    "Routing decisions by task type and target provider",
    ["task_type", "target_provider", "actual_provider"]
)

# Prompt cache metrics (Anthropic-style server-side caching)
PROMPT_CACHE_TOKENS = Counter(
    "switchboard_prompt_cache_tokens_total",
    "Prompt cache tokens (Anthropic cache_control)",
    ["provider", "cache_event"]  # cache_event: creation, read
)

# Response cache metrics
CACHE_HITS = Counter(
    "switchboard_cache_hits_total",
    "Total cache hits"
)

CACHE_MISSES = Counter(
    "switchboard_cache_misses_total",
    "Total cache misses"
)

CACHE_SIZE = Gauge(
    "switchboard_cache_size",
    "Current number of entries in cache"
)


def initialize_metrics(version: str = "0.1.0") -> None:
    """Initialize service info metric."""
    SWITCHBOARD_INFO.info({
        "version": version,
        "service": "switchboard",
    })


def record_request(
    provider: str,
    task_type: str,
    status: str,
    latency_seconds: float,
    tokens_prompt: int = 0,
    tokens_completion: int = 0,
    cost_usd: float = 0.0,
    cache_creation_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> None:
    """Record metrics for a completed request."""
    REQUESTS_TOTAL.labels(
        provider=provider,
        task_type=task_type,
        status=status
    ).inc()

    REQUEST_LATENCY.labels(
        provider=provider,
        task_type=task_type
    ).observe(latency_seconds)

    if tokens_prompt > 0:
        TOKENS_TOTAL.labels(
            provider=provider,
            token_type="prompt"
        ).inc(tokens_prompt)

    if tokens_completion > 0:
        TOKENS_TOTAL.labels(
            provider=provider,
            token_type="completion"
        ).inc(tokens_completion)

    if cost_usd > 0:
        COST_USD_TOTAL.labels(provider=provider).inc(cost_usd)

    # Anthropic prompt cache tokens
    if cache_creation_tokens > 0:
        PROMPT_CACHE_TOKENS.labels(
            provider=provider, cache_event="creation"
        ).inc(cache_creation_tokens)
    if cache_read_tokens > 0:
        PROMPT_CACHE_TOKENS.labels(
            provider=provider, cache_event="read"
        ).inc(cache_read_tokens)


def record_circuit_state(provider: str, state: int) -> None:
    """
    Record circuit breaker state.
    
    Args:
        provider: Provider name
        state: 0=closed, 1=open, 2=half_open
    """
    CIRCUIT_STATE.labels(provider=provider).set(state)


def record_circuit_failure(provider: str) -> None:
    """Record a circuit breaker failure."""
    CIRCUIT_FAILURES.labels(provider=provider).inc()


def record_rate_limit_hit(limit_type: str) -> None:
    """Record a rate limit rejection."""
    RATE_LIMIT_HITS.labels(limit_type=limit_type).inc()


def record_routing_decision(
    task_type: str,
    target_provider: str,
    actual_provider: str
) -> None:
    """Record a routing decision (for tracking fallbacks)."""
    ROUTING_DECISIONS.labels(
        task_type=task_type,
        target_provider=target_provider,
        actual_provider=actual_provider
    ).inc()


def set_provider_health(provider: str, health: int) -> None:
    """
    Set provider health status.
    
    Args:
        provider: Provider name
        health: 1=healthy, 0=unhealthy, -1=unknown
    """
    PROVIDER_HEALTH.labels(provider=provider).set(health)


def record_cache_hit() -> None:
    """Record a cache hit."""
    CACHE_HITS.inc()


def record_cache_miss() -> None:
    """Record a cache miss."""
    CACHE_MISSES.inc()


def set_cache_size(size: int) -> None:
    """Set current cache size."""
    CACHE_SIZE.set(size)
