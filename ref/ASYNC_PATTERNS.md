# Async/Sync Architecture Patterns

> The "Grand Unification" of The Switchboard's runtime model  
> Reference for ensuring non-blocking behavior throughout the system

## Critical Principle: Never Block the Event Loop

FastAPI/Uvicorn run on `asyncio`. Any blocking call (sync HTTP, `time.sleep`,
CPU-bound work) freezes the entire worker, preventing other requests from
being processed.

## ✅ Correct Patterns (Implemented)

### 1. Shared Async HTTP Client

**Location:** `src/switchboard/providers/base.py`, `src/switchboard/service/app.py`

```python
# App lifespan creates shared client with connection pooling
limits = httpx.Limits(max_connections=100, max_keepalive_connections=20)
async_client = httpx.AsyncClient(limits=limits, timeout=30.0)
BaseProvider.set_shared_client(async_client)

# Providers use shared client for non-blocking requests
async def chat_async(self, prompt: str, **kwargs) -> Mapping[str, Any]:
    response = await self._shared_async_client.post(url, headers=headers, json=payload)
    return response.json()
```

### 2. Async Sleep for Delays

**Location:** `src/switchboard/providers/echo.py`

```python
# WRONG: Blocks event loop
def chat(self, prompt):
    time.sleep(0.05)  # 🚫 Blocks EVERYTHING

# CORRECT: Yields to other coroutines
async def async_generate(self, prompt):
    await asyncio.sleep(0.05)  # ✅ Non-blocking
```

### 3. Sync Wrapper for Legacy Contexts

When code must support both sync and async callers:

```python
def chat(self, prompt: str) -> Mapping[str, Any]:
    """Sync method - for testing/legacy only. Does NOT block."""
    # No sleep, no blocking calls
    return self._build_response(prompt)

async def async_generate(self, prompt: str) -> Mapping[str, Any]:
    """Async method - preferred in production."""
    await asyncio.sleep(0.05)  # Simulated latency, non-blocking
    return self._build_response(prompt)
```

## ⚠️ Anti-Patterns to Avoid

### 1. Creating New Clients Per Request

```python
# WRONG: Connection overhead, no pooling
async def chat(self, prompt):
    async with httpx.AsyncClient() as client:  # 🚫 Creates new client every time
        response = await client.post(url, json=payload)

# CORRECT: Reuse shared client
async def chat(self, prompt):
    response = await self._shared_async_client.post(url, json=payload)  # ✅
```

### 2. Using `requests` Library in Async Code

```python
# WRONG: requests is synchronous
import requests
async def fetch_data():
    response = requests.get(url)  # 🚫 Blocks event loop

# CORRECT: Use httpx
async def fetch_data():
    response = await async_client.get(url)  # ✅
```

### 3. Blocking in Async Context

```python
# WRONG: time.sleep blocks
async def process():
    time.sleep(1)  # 🚫 Freezes entire server

# CORRECT: asyncio.sleep yields
async def process():
    await asyncio.sleep(1)  # ✅ Other requests can be handled
```

### 4. CPU-Bound Work in Event Loop

```python
# WRONG: Heavy computation blocks
async def hash_large_file(data):
    return hashlib.sha256(data).hexdigest()  # 🚫 Blocks if data is large

# CORRECT: Offload to thread pool
async def hash_large_file(data):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, hashlib.sha256(data).hexdigest)  # ✅
```

## Current Architecture Status

| Component | Async-Safe | Notes |
|-----------|------------|-------|
| BaseProvider.chat_async() | ✅ | Uses shared AsyncClient |
| BaseProvider.chat() | ⚠️ | Sync client for legacy use only |
| EchoProvider.async_generate() | ✅ | Uses asyncio.sleep() |
| EchoProvider.chat() | ✅ | No blocking calls |
| app.chat_completions() | ✅ | Calls async_generate() |
| RetryConfig | ✅ | Uses asyncio.sleep() in base.py |
| CircuitBreaker | ✅ | No async methods needed |

## Integration with aperion-legendary-ai

When integrating The Switchboard with the main project, ensure:

1. **Switchboard Client Adapter** uses `httpx.AsyncClient`
2. **Event Bus** handles both sync and async contexts (loop detection)
3. **Agents** call Switchboard via async HTTP, not sync

### Recommended Adapter Pattern

```python
# In aperion-legendary-ai: stack/aperion/foundation/llm/switchboard_adapter.py

class SwitchboardAdapter:
    """Async-first adapter for The Switchboard gateway."""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self._client: httpx.AsyncClient | None = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client
    
    async def chat_async(self, prompt: str, task_type: str = "general", **kwargs):
        client = await self._get_client()
        response = await client.post(
            f"{self.base_url}/v1/chat/completions",
            headers={"X-Aperion-Task-Type": task_type},
            json={
                "model": kwargs.get("model", "gpt-4.1-mini"),
                "messages": [{"role": "user", "content": prompt}],
            }
        )
        data = response.json()
        return {
            "replies": [c["message"]["content"] for c in data["choices"]],
            "provider": data.get("switchboard_provider"),
            "usage": data.get("usage", {}),
        }
    
    def chat(self, prompt: str, **kwargs):
        """Sync wrapper for legacy code - creates temporary event loop."""
        try:
            loop = asyncio.get_running_loop()
            future = asyncio.run_coroutine_threadsafe(
                self.chat_async(prompt, **kwargs), loop
            )
            return future.result(timeout=30)
        except RuntimeError:
            return asyncio.run(self.chat_async(prompt, **kwargs))
```

## Verification Commands

Check for remaining blocking patterns:

```bash
# Find time.sleep in source code
grep -rn "time\.sleep" src/

# Find sync httpx.Client (should only be in base.py for legacy)
grep -rn "httpx\.Client\(" src/

# Find requests library (should not be used)
grep -rn "import requests" src/

# Find blocking patterns in async functions
grep -rn "async def" src/ | xargs grep "time\.sleep\|requests\."
```

## References

- [HTTPX Async Guide](https://www.python-httpx.org/async/)
- [FastAPI Async Performance](https://fastapi.tiangolo.com/async/)
- [Python asyncio Best Practices](https://docs.python.org/3/library/asyncio-dev.html)
