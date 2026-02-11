# Best Practices for LLM API Gateways

> Compiled from industry sources, 2024-2026

## 1. Rate Limiting Patterns

### Token-Aware Rate Limiting

LLM providers impose limits on both **requests per minute (RPM)** and **tokens per minute (TPM)**.
A production gateway must track both:

```python
class TokenAwareRateLimiter:
    """Track both request count and token usage."""
    
    def __init__(self, rpm_limit: int = 60, tpm_limit: int = 100_000):
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.request_count = 0
        self.token_count = 0
        self.window_start = time.time()
    
    def can_proceed(self, estimated_tokens: int) -> bool:
        self._maybe_reset_window()
        return (
            self.request_count < self.rpm_limit and
            self.token_count + estimated_tokens < self.tpm_limit
        )
    
    def record_usage(self, actual_tokens: int):
        self.request_count += 1
        self.token_count += actual_tokens
```

### Per-User/Tenant Isolation

Prevent "noisy neighbor" problems by tracking limits per API key:

```python
limits = {
    "free_tier": {"rpm": 10, "tpm": 10_000},
    "pro_tier": {"rpm": 100, "tpm": 100_000},
    "enterprise": {"rpm": 1000, "tpm": 1_000_000},
}
```

**Sources:**
- [Collabnix: LLM Gateway Rate Limiting](https://collabnix.com/llm-gateway-patterns-rate-limiting-and-load-balancing-guide/)
- [Apache APISIX AI Gateway](https://apisix.apache.org/ai-gateway/)

---

## 2. Retry & Backoff Strategies

### Exponential Backoff with Jitter

```python
def calculate_backoff(attempt: int, base: float = 1.0, max_wait: float = 60.0) -> float:
    """Calculate wait time with exponential backoff and jitter."""
    wait = min(base * (2 ** attempt), max_wait)
    jitter = random.uniform(0, 0.25 * wait)  # 0-25% jitter
    return wait + jitter
```

### Retryable vs Non-Retryable Errors

| Status Code | Retry? | Notes |
|-------------|--------|-------|
| 400 Bad Request | ❌ | Fix the request |
| 401 Unauthorized | ❌ | Fix credentials |
| 403 Forbidden | ❌ | Check permissions |
| 429 Too Many Requests | ✅ | Respect Retry-After |
| 500 Internal Server Error | ✅ | Transient |
| 502 Bad Gateway | ✅ | Transient |
| 503 Service Unavailable | ✅ | Respect Retry-After |
| 504 Gateway Timeout | ✅ | Transient |

### Retry-After Header

Always check and respect the `Retry-After` header:

```python
retry_after = response.headers.get("Retry-After")
if retry_after:
    wait_seconds = float(retry_after)
    await asyncio.sleep(wait_seconds)
```

**Sources:**
- [OpenAI Error Codes](https://platform.openai.com/docs/guides/error-codes)
- [OpenAI Help: 429 Errors](https://help.openai.com/en/articles/5955604-how-can-i-solve-429-too-many-requests-errors)

---

## 3. Circuit Breaker Pattern

### State Machine

```
       success
    ┌───────────┐
    │           ▼
┌───────┐   ┌───────┐
│ CLOSED│◄──│ HALF- │
└───┬───┘   │ OPEN  │
    │       └───┬───┘
    │ failures  │ probe fails
    ▼           │
┌───────┐       │
│ OPEN  │───────┘
└───────┘  timeout expires
```

### Configuration Recommendations

```python
circuit_config = {
    "failure_threshold": 5,      # Open after 5 consecutive failures
    "success_threshold": 2,      # Close after 2 successful probes
    "timeout_seconds": 60,       # Stay open for 60s before probing
    "half_open_max_calls": 3,    # Max concurrent probes in half-open
}
```

### Provider Fallback on Open Circuit

When a circuit opens, immediately try the next provider in the fallback chain
rather than returning an error:

```python
async def call_with_fallback(prompt: str, chain: list[str]):
    for provider_name in chain:
        circuit = circuits[provider_name]
        if circuit.is_open:
            continue  # Skip, try next
        try:
            return await providers[provider_name].chat(prompt)
        except ProviderError:
            circuit.record_failure()
    raise AllProvidersFailedError()
```

**Sources:**
- [Maxim.ai: Circuit Breakers in LLM Apps](https://www.getmaxim.ai/articles/retries-fallbacks-and-circuit-breakers-in-llm-apps-a-production-guide/)

---

## 4. Async HTTP Client Best Practices

### Connection Pool Reuse

**Critical:** Create one `AsyncClient` per application lifetime, not per request:

```python
# ❌ BAD: Creates new pool per request
async def bad_request():
    async with httpx.AsyncClient() as client:
        return await client.post(url, json=data)

# ✅ GOOD: Reuses connection pool
class Provider:
    def __init__(self):
        self._client = None
    
    async def get_client(self):
        if self._client is None:
            limits = httpx.Limits(max_connections=100, max_keepalive_connections=20)
            self._client = httpx.AsyncClient(limits=limits, timeout=30.0)
        return self._client
    
    async def close(self):
        if self._client:
            await self._client.aclose()
```

### Limit Configuration

| Parameter | Recommended | Description |
|-----------|-------------|-------------|
| `max_connections` | 100 | Total connections across all hosts |
| `max_keepalive_connections` | 20 | Persistent connections per host |
| `keepalive_expiry` | 30.0 | Seconds to keep idle connections |

**Sources:**
- [HTTPX Async Documentation](https://www.python-httpx.org/async/)
- [StackOverflow: HTTPX Connection Pooling](https://stackoverflow.com/questions/69916682/python-httpx-how-does-httpx-clients-connection-pooling-work)

---

## 5. SSE Streaming Implementation

### FastAPI Streaming Response

```python
from fastapi.responses import StreamingResponse
import json

async def stream_llm_response(provider, prompt: str):
    """Generate SSE-formatted streaming response."""
    async for chunk in provider.stream_generate(prompt):
        if chunk.get("done"):
            yield "data: [DONE]\n\n"
            return
        
        sse_data = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": provider._model,
            "choices": [{
                "index": 0,
                "delta": {"content": chunk["chunk"]},
                "finish_reason": None
            }]
        }
        yield f"data: {json.dumps(sse_data)}\n\n"

@app.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest):
    if body.stream:
        return StreamingResponse(
            stream_llm_response(provider, body.get_prompt()),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
```

**Sources:**
- [OpenAI Streaming Responses](https://platform.openai.com/docs/guides/streaming-responses)
- [fangwentong/openai-proxy](https://github.com/fangwentong/openai-proxy)

---

## 6. FastAPI Production Deployment

### ASGI Server Stack

```bash
# Production command
gunicorn switchboard.main:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8080 \
    --preload
```

### Workers Formula

```
workers = 2 * cpu_cores + 1
```

For LLM gateway (I/O bound), consider increasing to `4 * cpu_cores`.

### Essential Middlewares

1. **Security Headers** (HSTS, X-Frame-Options, CSP)
2. **GZip Compression** for response efficiency
3. **Request Timing** for observability
4. **CORS** for browser access
5. **Rate Limiting** for abuse prevention
6. **Authentication** early in chain

**Sources:**
- [Render: FastAPI Production Best Practices](https://render.com/articles/fastapi-production-deployment-best-practices)
- [PyTutorial: FastAPI Performance](https://pytutorial.com/fastapi-performance-optimization-guide/)

---

## 7. LiteLLM Feature Comparison

The Switchboard should aim for feature parity with LiteLLM for critical features:

| Feature | LiteLLM | Switchboard | Priority |
|---------|---------|-------------|----------|
| Multi-provider routing | ✅ | ✅ | - |
| OpenAI-compatible API | ✅ | ✅ | - |
| Fail-closed semantics | ❌ | ✅ | Advantage |
| Streaming support | ✅ | ⚠️ Partial | P1 |
| Rate limiting | ✅ | ❌ | P1 |
| Retry/backoff | ✅ | ❌ | P0 |
| Circuit breaker | ✅ | ❌ | P1 |
| Virtual keys | ✅ | ❌ | P3 |
| Cost tracking | ✅ | ✅ | - |
| Budget enforcement | ✅ | ❌ | P2 |
| Prometheus metrics | ✅ | ❌ | P2 |

**Sources:**
- [LiteLLM Documentation](https://docs.litellm.ai/docs/providers/litellm_proxy)
- [LiteLLM Architecture](https://deepwiki.com/BerriAI/litellm/3.1-architecture-and-request-flow)

---

## 8. Observability Best Practices

### Structured Logging (Constitution D3)

```python
logger.info(
    "llm_request",
    correlation_id=correlation_id,
    provider=provider_name,
    model=model,
    task_type=task_type,
    tokens_in=usage.get("prompt_tokens", 0),
    tokens_out=usage.get("completion_tokens", 0),
    latency_ms=round(elapsed * 1000, 2),
    cost_usd=estimated_cost,
    status="success",
)
```

### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, Gauge

requests_total = Counter(
    "switchboard_requests_total",
    "Total requests",
    ["provider", "task_type", "status"]
)

request_duration = Histogram(
    "switchboard_request_duration_seconds",
    "Request duration",
    ["provider"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
)

provider_health = Gauge(
    "switchboard_provider_health",
    "Provider health (1=healthy, 0=unhealthy)",
    ["provider"]
)
```

### Correlation ID Propagation

Every request should have a correlation ID that flows through:
1. Incoming request → X-Correlation-ID header (or generate)
2. All internal logs
3. All outgoing provider requests
4. Response header

This enables end-to-end tracing across distributed systems.

---

## 9. Security Checklist

- [ ] API key validation before processing requests
- [ ] Rate limiting per key/tenant
- [ ] Max request body size (prevent large prompt attacks)
- [ ] Input sanitization (no prompt injection vectors)
- [ ] TLS in production (behind reverse proxy)
- [ ] No secrets in logs
- [ ] Audit logging for compliance
- [ ] Health endpoints excluded from auth
