"""
Middleware for The Switchboard.

Implements:
- Constitution D1: Telemetry Injection (X-Correlation-ID propagation)
- Constitution D3: Structured Logging (JSON cost/latency metrics)
- Authentication (Bearer token validation)
"""

import logging
import time
import uuid
from contextvars import ContextVar
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
import structlog

# Context variable for correlation ID (thread-safe)
correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="")

logger = structlog.get_logger(__name__)


def get_correlation_id() -> str:
    """Get current correlation ID from context."""
    return correlation_id_var.get()


def set_correlation_id(correlation_id: str) -> None:
    """Set correlation ID in context."""
    correlation_id_var.set(correlation_id)


class TelemetryMiddleware(BaseHTTPMiddleware):
    """
    Telemetry injection middleware (Constitution D1).

    - Extracts X-Correlation-ID from incoming requests
    - Generates one if missing
    - Propagates to all outgoing provider requests
    - Adds to response headers for tracing
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Extract or generate correlation ID
        correlation_id = request.headers.get("X-Correlation-ID")
        if not correlation_id:
            correlation_id = f"sw_{uuid.uuid4().hex[:16]}"

        # Set in context for downstream use
        set_correlation_id(correlation_id)

        # Store in request state for handlers
        request.state.correlation_id = correlation_id

        # Record start time
        start_time = time.monotonic()

        try:
            response = await call_next(request)
        except Exception as e:
            # Log error with correlation ID
            logger.error(
                "Request failed",
                correlation_id=correlation_id,
                path=request.url.path,
                method=request.method,
                error=str(e),
            )
            raise

        # Calculate elapsed time
        elapsed_ms = (time.monotonic() - start_time) * 1000

        # Add headers to response
        response.headers["X-Correlation-ID"] = correlation_id
        response.headers["X-Request-Duration-Ms"] = f"{elapsed_ms:.2f}"

        # Log request completion
        logger.info(
            "Request completed",
            correlation_id=correlation_id,
            path=request.url.path,
            method=request.method,
            status_code=response.status_code,
            duration_ms=round(elapsed_ms, 2),
        )

        return response


class CostLoggingMiddleware(BaseHTTPMiddleware):
    """
    Cost logging middleware (Constitution D3).

    Logs estimated cost per request in structured JSON format:
    {
        "event": "llm_request_cost",
        "correlation_id": "...",
        "provider": "openai",
        "estimated_cost_usd": 0.00015,
        "tokens": {"prompt": 100, "completion": 50},
        "latency_ms": 1234
    }
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.monotonic()

        response = await call_next(request)

        elapsed_ms = (time.monotonic() - start_time) * 1000

        # Only log cost for completion endpoints
        if request.url.path.endswith("/chat/completions"):
            correlation_id = getattr(request.state, "correlation_id", "unknown")

            # Extract cost info from response if available
            # (This would be populated by the endpoint handler)
            cost_info = getattr(request.state, "cost_info", None)

            if cost_info:
                logger.info(
                    "llm_request_cost",
                    correlation_id=correlation_id,
                    provider=cost_info.get("provider", "unknown"),
                    model=cost_info.get("model", "unknown"),
                    estimated_cost_usd=cost_info.get("estimated_cost_usd", 0.0),
                    tokens=cost_info.get("tokens", {}),
                    latency_ms=round(elapsed_ms, 2),
                    task_type=cost_info.get("task_type", "general"),
                )

        return response


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Authentication middleware.

    Validates Bearer tokens for API access.
    Skips auth for health check endpoints.
    """

    SKIP_AUTH_PATHS = {"/health", "/healthz", "/ready", "/metrics"}

    def __init__(self, app, api_key: str | None = None):
        super().__init__(app)
        self.api_key = api_key

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip auth for health endpoints
        if request.url.path in self.SKIP_AUTH_PATHS:
            return await call_next(request)

        # Skip auth if no API key is configured (development mode)
        if not self.api_key:
            return await call_next(request)

        # Validate Authorization header
        auth_header = request.headers.get("Authorization", "")

        if not auth_header.startswith("Bearer "):
            return Response(
                content='{"error": {"message": "Missing or invalid Authorization header", "type": "auth_error"}}',
                status_code=401,
                media_type="application/json",
            )

        token = auth_header[7:]  # Remove "Bearer " prefix

        if token != self.api_key:
            return Response(
                content='{"error": {"message": "Invalid API key", "type": "auth_error"}}',
                status_code=401,
                media_type="application/json",
            )

        return await call_next(request)


def configure_structlog() -> None:
    """Configure structlog for JSON output (Constitution D3)."""
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Set up standard library logging
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO,
    )


# Request body size limits (bytes)
MAX_REQUEST_BODY_SIZE = 10 * 1024 * 1024  # 10 MB


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware to limit request body size.

    Prevents memory exhaustion attacks and expensive provider calls
    with excessively large prompts.
    """

    def __init__(self, app, max_size: int = MAX_REQUEST_BODY_SIZE):
        super().__init__(app)
        self.max_size = max_size

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Check Content-Length header if present
        content_length = request.headers.get("content-length")

        if content_length:
            if int(content_length) > self.max_size:
                return Response(
                    content='{"error": {"message": "Request body too large", "type": "request_error", "param": null, "code": "request_too_large"}}',
                    status_code=413,
                    media_type="application/json",
                )

        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware.

    Uses token bucket algorithm to limit requests per API key and globally.
    Returns 429 Too Many Requests when limit exceeded.
    """

    # Paths that bypass rate limiting
    SKIP_PATHS = {"/health", "/healthz", "/ready", "/metrics", "/circuits"}

    def __init__(self, app, rate_limiter=None):
        super().__init__(app)
        from ..core.rate_limit import get_rate_limiter
        self.rate_limiter = rate_limiter or get_rate_limiter()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip rate limiting for health/metrics endpoints
        if request.url.path in self.SKIP_PATHS:
            return await call_next(request)

        # Extract API key from Authorization header
        api_key = None
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            api_key = auth_header[7:]

        # Check rate limit
        result = self.rate_limiter.check(key=api_key)

        if not result.allowed:
            # Build 429 response with rate limit headers
            headers = {
                "X-RateLimit-Limit": str(result.limit),
                "X-RateLimit-Remaining": str(result.remaining),
                "X-RateLimit-Reset": str(int(result.reset_at)),
                "Retry-After": str(int(result.retry_after or 1)),
            }
            return Response(
                content='{"error": {"message": "Rate limit exceeded", "type": "rate_limit_error", "code": "rate_limit_exceeded"}}',
                status_code=429,
                media_type="application/json",
                headers=headers,
            )

        # Add rate limit headers to successful response
        response = await call_next(request)

        response.headers["X-RateLimit-Limit"] = str(result.limit)
        response.headers["X-RateLimit-Remaining"] = str(result.remaining)
        response.headers["X-RateLimit-Reset"] = str(int(result.reset_at))

        return response
