"""
The Switchboard - FastAPI Application.

Unified LLM API Gateway providing:
- OpenAI-compatible /v1/chat/completions endpoint
- Task-based intelligent routing via X-Aperion-Task-Type header
- Fail-closed semantics (Constitution A6)
- Telemetry injection (Constitution D1)
- Structured cost logging (Constitution D3)
- Async HTTP with retry and circuit breaker patterns
- SSE streaming for real-time responses
- Prometheus metrics for observability
"""

import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
import httpx
from prometheus_fastapi_instrumentator import Instrumentator
import structlog

from ..core import (
    CircuitOpenError,
    FailClosedError,
    LLMRouter,
    TaskType,
    check_fail_closed,
    get_response_cache,
)
from ..core.protocol import LLMClient, ProviderError, ProviderHealth
from ..core.resilience import get_all_circuit_stats
from ..providers import (
    AnthropicProvider,
    EchoProvider,
    GeminiProvider,
    OpenAIProvider,
    WorkersAIProvider,
)
from ..providers.base import BaseProvider
from .middleware import (
    AuthMiddleware,
    CostLoggingMiddleware,
    RateLimitMiddleware,
    RequestSizeLimitMiddleware,
    TelemetryMiddleware,
    configure_structlog,
    get_correlation_id,
)
from .metrics import (
    initialize_metrics,
    record_cache_hit,
    record_cache_miss,
    record_request,
    record_circuit_state,
    set_cache_size,
    set_provider_health,
)
from .schemas import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ChoiceMessage,
    CompletionUsage,
    ErrorResponse,
    HealthResponse,
    StreamChoice,
)

logger = structlog.get_logger(__name__)

# Global state
_providers: dict[str, LLMClient] = {}
_router: LLMRouter | None = None


def _init_providers() -> dict[str, LLMClient]:
    """Initialize all provider instances."""
    return {
        "anthropic": AnthropicProvider(),
        "openai": OpenAIProvider(),
        "gemini": GeminiProvider(),
        "workers_ai": WorkersAIProvider(),
        "echo": EchoProvider(),
    }


def _init_router(providers: dict[str, LLMClient]) -> LLMRouter:
    """Initialize the LLM router with providers."""
    router = LLMRouter(providers)
    return router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - startup and shutdown."""
    global _providers, _router

    # Configure structured logging
    configure_structlog()

    logger.info("Starting The Switchboard...")

    # Create shared async HTTP client with connection pooling
    limits = httpx.Limits(max_connections=100, max_keepalive_connections=20)
    async_client = httpx.AsyncClient(limits=limits, timeout=30.0)
    BaseProvider.set_shared_client(async_client)
    logger.info("Shared async HTTP client initialized")

    # Initialize providers
    _providers = _init_providers()

    # Log provider status
    for name, provider in _providers.items():
        status = "configured" if provider.is_configured else "not configured"
        logger.info(f"Provider {name}: {status}")

    # CRITICAL: Fail-closed check (Constitution A6)
    try:
        check_fail_closed(_providers)
        logger.info("✅ Fail-closed check passed - service ready")
    except FailClosedError as e:
        logger.critical(f"🚨 {e}")
        logger.critical(e.remediation)
        raise RuntimeError(str(e)) from e

    # Initialize router
    _router = _init_router(_providers)

    yield

    # Shutdown
    logger.info("Shutting down The Switchboard...")

    # Close shared async client
    await BaseProvider.close_shared_client()
    logger.info("Shared async HTTP client closed")

    # Close individual provider sync clients
    for provider in _providers.values():
        if hasattr(provider, "close"):
            provider.close()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="The Switchboard",
        description="Unified LLM API Gateway with fail-closed semantics",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Add middleware (order matters - first added is outermost)
    # Order: Size Limit → Rate Limit → Telemetry → Cost → Auth
    api_key = os.environ.get("SWITCHBOARD_API_KEY")
    if api_key:
        app.add_middleware(AuthMiddleware, api_key=api_key)

    app.add_middleware(CostLoggingMiddleware)
    app.add_middleware(TelemetryMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(RequestSizeLimitMiddleware)

    # Register routes
    app.add_api_route("/health", health_check, methods=["GET"])
    app.add_api_route("/healthz", health_check, methods=["GET"])
    app.add_api_route("/circuits", get_circuits, methods=["GET"])
    app.add_api_route("/cache", get_cache_stats, methods=["GET"])
    app.add_api_route("/cache/clear", clear_cache, methods=["POST"])
    app.add_api_route(
        "/v1/chat/completions",
        chat_completions,
        methods=["POST"],
        response_model=ChatCompletionResponse,
        responses={
            503: {"model": ErrorResponse, "description": "No providers available"},
            429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        },
    )

    # Initialize Prometheus metrics
    initialize_metrics(version="0.1.0")
    
    # Add Prometheus instrumentation for automatic HTTP metrics
    # Skip in test mode to avoid registry duplication
    if not os.environ.get("PYTEST_CURRENT_TEST"):
        try:
            instrumentator = Instrumentator(
                should_group_status_codes=True,
                should_ignore_untemplated=True,
                should_instrument_requests_inprogress=True,
                excluded_handlers=["/health", "/healthz", "/metrics"],
            )
            instrumentator.instrument(app)
        except ValueError:
            # Metrics already registered (multiple app instances)
            pass
    
    # Manually add /metrics endpoint
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    from fastapi.responses import Response
    
    @app.get("/metrics", include_in_schema=False)
    async def metrics():
        """Prometheus metrics endpoint."""
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )

    return app


async def health_check() -> HealthResponse:
    """Health check endpoint."""
    global _providers

    provider_status: dict[str, dict[str, Any]] = {}
    any_healthy = False

    for name, provider in _providers.items():
        if name == "echo":
            continue  # Skip echo in health status

        # Use async health check if available, fall back to sync
        if hasattr(provider, 'health_check_async'):
            health = await provider.health_check_async()
        else:
            health = provider.health_check()
        configured = provider.is_configured

        provider_status[name] = {
            "configured": configured,
            "health": health.value,
        }
        
        # Update Prometheus metric
        health_value = 1 if health == ProviderHealth.HEALTHY else (0 if health == ProviderHealth.UNHEALTHY else -1)
        set_provider_health(name, health_value)

        if configured and health in (ProviderHealth.HEALTHY, ProviderHealth.UNKNOWN):
            any_healthy = True

    # Determine overall status
    if any_healthy:
        status = "healthy"
    elif any(p["configured"] for p in provider_status.values()):
        status = "degraded"
    else:
        status = "unhealthy"

    return HealthResponse(
        status=status,  # type: ignore[arg-type]
        version="0.1.0",
        providers=provider_status,
        fail_closed_compliant=any_healthy,
    )


def _parse_task_type(task_type_header: str | None) -> TaskType:
    """Parse X-Aperion-Task-Type header into TaskType enum."""
    if not task_type_header:
        return TaskType.GENERAL

    # Try exact match
    task_type_header = task_type_header.lower().strip()

    for tt in TaskType:
        if tt.value == task_type_header:
            return tt

    # Try name match (e.g., "SECURITY_AUDIT")
    try:
        return TaskType[task_type_header.upper()]
    except KeyError:
        pass

    return TaskType.GENERAL


async def _stream_completion(
    provider: LLMClient,
    prompt: str,
    request_id: str,
    model: str,
    correlation_id: str,
    **kwargs: Any,
) -> AsyncIterator[str]:
    """
    Generate SSE stream from provider.

    Yields Server-Sent Events in OpenAI-compatible format:
    - data: {chunk JSON}
    - data: [DONE]
    """
    created = int(time.time())
    total_content = ""

    try:
        async for chunk in provider.stream_generate(prompt, **kwargs):
            if chunk.get("done"):
                # Final chunk - send finish reason
                final_chunk = ChatCompletionChunk(
                    id=request_id,
                    created=created,
                    model=model,
                    choices=[
                        StreamChoice(
                            index=0,
                            delta={},
                            finish_reason="stop",
                        )
                    ],
                )
                yield f"data: {final_chunk.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"
                return

            content = chunk.get("chunk", "")
            if content:
                total_content += content
                stream_chunk = ChatCompletionChunk(
                    id=request_id,
                    created=created,
                    model=model,
                    choices=[
                        StreamChoice(
                            index=0,
                            delta={"content": content},
                            finish_reason=None,
                        )
                    ],
                )
                yield f"data: {stream_chunk.model_dump_json()}\n\n"

    except Exception as e:
        logger.error(
            "Streaming error",
            correlation_id=correlation_id,
            error=str(e),
        )
        error_data = {
            "error": {
                "message": str(e),
                "type": "stream_error",
            }
        }
        yield f"data: {json.dumps(error_data)}\n\n"


async def chat_completions(
    request: Request,
    body: ChatCompletionRequest,
    x_aperion_task_type: str | None = Header(default=None, alias="X-Aperion-Task-Type"),
    x_switchboard_no_cache: str | None = Header(default=None, alias="X-Switchboard-No-Cache"),
) -> ChatCompletionResponse | StreamingResponse:
    """
    OpenAI-compatible chat completions endpoint.

    Accepts X-Aperion-Task-Type header for intelligent routing:
    - security_audit, production_decision → Premium (OpenAI)
    - doc_update, lint_analysis → Free tier (Gemini)
    - testing, development → Echo (if allowed)

    Features:
    - Response caching (bypass with X-Switchboard-No-Cache: true)
    - Streaming SSE responses when stream=true
    - Async HTTP with retry and circuit breaker
    - Automatic fallback to next provider on failure
    """
    global _router

    if _router is None:
        raise HTTPException(status_code=503, detail="Router not initialized")

    correlation_id = getattr(request.state, "correlation_id", get_correlation_id())
    task_type = _parse_task_type(x_aperion_task_type)
    request_id = f"sw_{uuid.uuid4().hex[:12]}"

    start_time = time.monotonic()

    try:
        # Get routing decision
        provider, decision = _router.get_provider(task_type, fallback=True)

        logger.info(
            "Routing request",
            correlation_id=correlation_id,
            task_type=task_type.value,
            provider=decision.provider_name,
            reason=decision.reason,
            stream=body.stream,
        )

        # Build prompt from messages
        prompt = body.get_prompt()
        kwargs = body.to_provider_kwargs()

        # Add system prompt as persona if present
        if system_prompt := body.get_system_prompt():
            kwargs["persona"] = system_prompt

        # Check cache for non-streaming requests
        use_cache = not body.stream and x_switchboard_no_cache != "true"
        cache = get_response_cache()
        cached_response = None
        
        if use_cache:
            cached_response = cache.get(
                prompt=prompt,
                model=body.model,
                temperature=body.temperature,
                max_tokens=body.max_tokens,
            )
            if cached_response:
                record_cache_hit()
                set_cache_size(cache.get_stats().size)
                logger.info(
                    "Cache hit",
                    correlation_id=correlation_id,
                    prompt_preview=prompt[:50] + "..." if len(prompt) > 50 else prompt,
                )
                # Return cached response with cache header
                return ChatCompletionResponse(
                    id=request_id,
                    created=int(time.time()),
                    model=cached_response.get("model", body.model),
                    choices=[
                        Choice(
                            index=i,
                            message=ChoiceMessage(content=str(reply)),  # Ensure string
                            finish_reason="stop",
                        )
                        for i, reply in enumerate(cached_response.get("replies", []))
                    ],
                    usage=CompletionUsage(
                        prompt_tokens=cached_response.get("usage", {}).get("prompt_tokens", 0),
                        completion_tokens=cached_response.get("usage", {}).get("completion_tokens", 0),
                        total_tokens=cached_response.get("usage", {}).get("total_tokens", 0),
                    ),
                )
            else:
                record_cache_miss()

        # Handle streaming requests
        if body.stream:
            return StreamingResponse(
                _stream_completion(
                    provider=provider,
                    prompt=prompt,
                    request_id=request_id,
                    model=body.model,
                    correlation_id=correlation_id,
                    **kwargs,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Correlation-ID": correlation_id,
                    "X-Switchboard-Provider": provider.name,
                },
            )

        # Non-streaming: add correlation_id to kwargs
        kwargs["correlation_id"] = correlation_id

        # Call provider (async with retry + circuit breaker)
        result = await provider.async_generate(prompt, **kwargs)

        elapsed_ms = (time.monotonic() - start_time) * 1000

        # Extract usage
        usage_data = result.get("usage", {})
        prompt_tokens = usage_data.get("prompt_tokens", 0)
        completion_tokens = usage_data.get("completion_tokens", 0)
        total_tokens = usage_data.get("total_tokens", prompt_tokens + completion_tokens)

        # Anthropic prompt cache tokens
        cache_creation_tokens = usage_data.get("cache_creation_input_tokens", 0)
        cache_read_tokens = usage_data.get("cache_read_input_tokens", 0)

        # Track usage in router
        _router.track_usage(
            provider_name=provider.name,
            tokens=total_tokens,
            latency_ms=elapsed_ms,
            success=True,
        )

        # Store cost info for middleware
        estimated_cost = decision.estimated_cost_per_1m_tokens * total_tokens / 1_000_000
        request.state.cost_info = {
            "provider": provider.name,
            "model": result.get("model", body.model),
            "estimated_cost_usd": estimated_cost,
            "tokens": {
                "prompt": prompt_tokens,
                "completion": completion_tokens,
                "total": total_tokens,
                "cache_creation": cache_creation_tokens,
                "cache_read": cache_read_tokens,
            },
            "task_type": task_type.value,
        }

        # Record Prometheus metrics
        record_request(
            provider=provider.name,
            task_type=task_type.value,
            status="success",
            latency_seconds=elapsed_ms / 1000,
            tokens_prompt=prompt_tokens,
            tokens_completion=completion_tokens,
            cost_usd=estimated_cost,
            cache_creation_tokens=cache_creation_tokens,
            cache_read_tokens=cache_read_tokens,
        )

        # Build response
        replies = result.get("replies", [])
        choices = [
            Choice(
                index=i,
                message=ChoiceMessage(content=str(reply)),  # Ensure string
                finish_reason="stop",
            )
            for i, reply in enumerate(replies)
        ]

        # Store in cache for future requests
        if use_cache:
            cache.set(
                prompt=prompt,
                model=body.model,
                response={
                    "replies": replies,
                    "model": result.get("model", body.model),
                    "usage": usage_data,
                },
                temperature=body.temperature,
                max_tokens=body.max_tokens,
            )
            set_cache_size(cache.get_stats().size)

        # Build usage with optional cache token fields
        usage_response = CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cache_creation_input_tokens=cache_creation_tokens or None,
            cache_read_input_tokens=cache_read_tokens or None,
        )

        return ChatCompletionResponse(
            id=request_id,
            created=int(time.time()),
            model=result.get("model", body.model),
            choices=choices,
            usage=usage_response,
            switchboard_provider=provider.name,
            switchboard_routing_reason=decision.reason,
        )

    except CircuitOpenError as e:
        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.warning(
            "Circuit breaker open - provider unavailable",
            correlation_id=correlation_id,
            provider=e.provider,
            duration_ms=elapsed_ms,
        )
        raise HTTPException(
            status_code=503,
            detail={
                "message": f"Provider {e.provider} temporarily unavailable (circuit open)",
                "type": "circuit_open",
                "provider": e.provider,
                "recoverable": True,
            },
        )

    except ProviderError as e:
        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.error(
            "Provider error",
            correlation_id=correlation_id,
            provider=e.provider,
            error=str(e),
            duration_ms=elapsed_ms,
        )

        if _router:
            _router.track_usage(
                provider_name=e.provider,
                tokens=0,
                latency_ms=elapsed_ms,
                success=False,
            )

        status_code = e.status_code or (429 if e.recoverable else 503)
        raise HTTPException(
            status_code=status_code,
            detail={
                "message": str(e),
                "type": "provider_error",
                "provider": e.provider,
                "recoverable": e.recoverable,
            },
        )


async def get_circuits() -> dict:
    """Get circuit breaker status for all providers."""
    return {
        "circuits": get_all_circuit_stats(),
    }


async def get_cache_stats() -> dict:
    """Get cache statistics."""
    cache = get_response_cache()
    stats = cache.get_stats()
    return {
        "size": stats.size,
        "max_size": stats.max_size,
        "hits": stats.hits,
        "misses": stats.misses,
        "hit_rate": stats.hit_rate,
        "ttl_seconds": cache.config.default_ttl_seconds,
        "min_prompt_length": cache.config.min_prompt_length,
    }


async def clear_cache() -> dict:
    """Clear the response cache."""
    cache = get_response_cache()
    cache.clear()
    set_cache_size(0)
    return {"message": "Cache cleared", "size": 0}


# Singleton app instance
_app: FastAPI | None = None


def get_app() -> FastAPI:
    """Get or create the singleton app instance."""
    global _app
    if _app is None:
        _app = create_app()
    return _app
