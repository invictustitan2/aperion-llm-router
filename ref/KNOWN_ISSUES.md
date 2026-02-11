# Known Issues & Limitations

> Current issues in The Switchboard that need attention  
> Last Updated: 2026-02-08

## ✅ Recently Resolved

### ISSUE-R001: Synchronous HTTP Client in Async Context ✅

**Status:** ✅ Resolved (2026-02-08)
**Resolution:** Added `chat_async()` method with true async `httpx.AsyncClient`.
Shared client with connection pooling set up in app lifespan.
The sync `chat()` retained for backwards compatibility but async path is now default.

### ISSUE-R002: No Retry Logic on Transient Failures ✅

**Status:** ✅ Resolved (2026-02-08)
**Resolution:** Implemented `RetryConfig` in `core/resilience.py` with:
- Exponential backoff (1s, 2s, 4s...) with 25% jitter
- Respects `Retry-After` headers
- Max 3 attempts by default
- Retryable codes: 429, 500, 502, 503, 504

### ISSUE-R003: No Circuit Breaker ✅

**Status:** ✅ Resolved (2026-02-08)
**Resolution:** Implemented `CircuitBreaker` in `core/resilience.py` with:
- Per-provider circuit breaker state machine
- Opens after 5 consecutive failures
- 60-second timeout before recovery probe
- New `/circuits` endpoint for monitoring

### ISSUE-R004: Echo Provider Includes Blocking Sleep ✅

**Status:** ✅ Resolved (2026-02-08)
**Resolution:** Replaced `time.sleep()` with `asyncio.sleep()` in async methods.
Sync `chat()` no longer sleeps to avoid accidental blocking.

---

### ISSUE-R005: Streaming Not Implemented in Service Layer ✅

**Status:** ✅ Resolved (2026-02-08)
**Resolution:** Implemented SSE streaming in `/v1/chat/completions`:
- `_stream_completion()` async generator yields SSE `data:` lines
- Uses `ChatCompletionChunk` and `StreamChoice` schemas for OpenAI compatibility
- Final chunk has `finish_reason="stop"`, then `data: [DONE]`
- Proper headers: text/event-stream, no-cache, keep-alive, X-Correlation-ID
- 9 streaming tests passing

---

## High Priority Issues (Remaining)

### ISSUE-R007: No Rate Limiting ✅

**Status:** ✅ Resolved (2026-02-08)
**Resolution:** Implemented token bucket rate limiting:
- `RateLimiter` class in `core/rate_limit.py`
- Per-API-key quotas with independent buckets
- Global rate limit fallback
- `RateLimitMiddleware` returns 429 with headers
- Configurable burst size and requests per minute
- 16 rate limit tests passing

---

## Medium Priority Issues

### ISSUE-005: No Maximum Request Size

**Status:** 🟡 Open
### ISSUE-R006: No Maximum Request Size ✅

**Status:** ✅ Resolved (2026-02-08)
**Resolution:** Implemented multi-layer validation:
- `RequestSizeLimitMiddleware` rejects requests with Content-Length > 10MB
- `ChatMessage.content` max length: 128,000 characters
- `ChatCompletionRequest.messages` max count: 100
- `ChatCompletionRequest.user` max length: 256
- 20 validation tests passing

---

## Medium Priority Issues

### ISSUE-008: Health Check Returns UNKNOWN for Real Providers

**Status:** 🟡 Open
**Severity:** Medium
**Component:** `src/switchboard/providers/base.py`

**Description:**
`health_check()` in BaseProvider returns `UNKNOWN` for configured providers.
No actual health probe is performed.

**Current Code:**
```python
def health_check(self) -> ProviderHealth:
    if not self.is_configured:
        return ProviderHealth.UNHEALTHY
    return ProviderHealth.UNKNOWN  # <-- Never actually checks
```

**Required Fix:**
Implement lightweight health probe (e.g., small completion request).

---

## Low Priority Issues

### ISSUE-009: No Prometheus Metrics Endpoint

**Status:** 🟢 Open
**Severity:** Low
**Component:** `src/switchboard/service/`

**Description:**
Structured logging exists but no `/metrics` endpoint for Prometheus scraping.

---

### ISSUE-008: Cost Tracking Not Persisted

**Status:** 🟢 Backlog
**Severity:** Low
**Component:** `src/switchboard/core/router.py`

**Description:**
`LLMRouter` tracks usage in memory only. Stats lost on restart.

**Required Fix:**
Optionally persist to database or emit to metrics system.

---

## Resolved Issues

### ISSUE-R005: Fail-Closed Logic Allowed Unconfigured Providers in Chain

**Status:** ✅ Resolved (2026-02-08)
**Resolution:** Fixed `get_safe_fallback_chain()` to only include configured providers.

---

## Issue Template

```markdown
### ISSUE-XXX: Title

**Status:** 🔴 Open | 🟠 In Progress | 🟢 Backlog | ✅ Resolved
**Severity:** Critical | High | Medium | Low
**Component:** file/path

**Description:**
What is the issue?

**Impact:**
What problems does this cause?

**Current Code:**
```python
# Show problematic code
```

**Required Fix:**
What needs to change?

**Workaround:**
Any temporary mitigation?
```

---

## Last Updated

2026-02-08
