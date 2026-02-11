# The Switchboard 🔌

> Unified LLM API Gateway with Fail-Closed Semantics

The Switchboard is a high-performance proxy service that provides intelligent routing
to multiple LLM providers. It serves as the central nervous system for all Aperion
agents (Sentinel, AR, Aether), ensuring they can access LLMs reliably and cost-effectively.

## 🎯 Core Features

- **OpenAI-Compatible API**: Drop-in replacement - just change `base_url`
- **Intelligent Task Routing**: Security tasks → Premium, Docs → Free tier
- **Fail-Closed Semantics**: Never silently falls back to Echo in production
- **Cost Optimization**: Target 75% savings by routing volume to free tiers
- **Telemetry Injection**: X-Correlation-ID propagation for tracing
- **Structured Logging**: JSON cost/latency metrics (Constitution D3)

## 🚀 Quick Start

### Installation

```bash
pip install aperion-switchboard

# From source
pip install -e .

# With dev dependencies
pip install -e ".[dev]"
```

### Configuration

Set environment variables for your providers:

```bash
# OpenAI (Premium tier)
export OPENAI_API_KEY=sk-...

# Google Gemini (Free tier)
export GEMINI_API_KEY=AIza...

# Cloudflare Workers AI (Low-cost tier)
export WORKERS_AI_API_KEY=your-cf-token
export WORKERS_AI_BASE_URL=https://api.cloudflare.com/client/v4/accounts/ACCT/ai/run
```

### Running

```bash
# Development
python -m aperion_switchboard.main

# Production
uvicorn aperion_switchboard.main:app --host 0.0.0.0 --port 8080

# Docker
docker build -t switchboard .
docker run -p 8080:8080 \
  -e OPENAI_API_KEY=sk-... \
  -e GEMINI_API_KEY=AIza... \
  switchboard
```

## 📡 API Usage

### OpenAI-Compatible Endpoint

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Aperion-Task-Type: security_audit" \
  -d '{
    "model": "gpt-4.1-mini",
    "messages": [{"role": "user", "content": "Analyze this code for vulnerabilities"}]
  }'
```

### Task Types

Use the `X-Aperion-Task-Type` header to trigger intelligent routing:

| Task Type | Routes To | Use Case |
|-----------|-----------|----------|
| `security_audit` | OpenAI | Critical security analysis |
| `production_decision` | OpenAI | High-stakes decisions |
| `strategic_analysis` | OpenAI | Complex reasoning |
| `code_review` | OpenAI | Quality reviews |
| `doc_update` | Gemini | Documentation updates |
| `doc_generation` | Gemini | Batch doc creation |
| `lint_analysis` | Gemini | Fast batch processing |
| `test_generation` | Gemini | High-volume generation |
| `general` | Gemini | Default (cost-optimized) |

## 🔒 Constitution Compliance

### A6: Fail-Closed Semantics (Iron Rule)

The Switchboard **MUST NEVER** silently fall back to the Echo provider in production.

- If no real providers are configured AND `APERION_ALLOW_ECHO` is not "true":
  - Service crashes on startup
  - Returns 503 for all requests
  - Logs CRITICAL error with remediation steps

```bash
# Production mode (default) - will crash if no providers configured
export APERION_ALLOW_ECHO=false

# Development mode - allows echo fallback
export APERION_ALLOW_ECHO=true
```

### B1: Secrets Management

All credentials are loaded from environment variables:

- `OPENAI_API_KEY`
- `GEMINI_API_KEY`
- `WORKERS_AI_API_KEY`
- `SWITCHBOARD_API_KEY` (optional - for Switchboard auth)

### D1: Telemetry Injection

- Extracts `X-Correlation-ID` from incoming requests
- Generates one if missing (format: `sw_{uuid}`)
- Propagates to all upstream provider requests
- Adds to response headers

### D3: Structured Logging

All cost/latency metrics are logged as JSON:

```json
{
  "event": "llm_request_cost",
  "correlation_id": "sw_abc123",
  "provider": "openai",
  "model": "gpt-4.1-mini",
  "estimated_cost_usd": 0.00015,
  "tokens": {"prompt": 100, "completion": 50, "total": 150},
  "latency_ms": 1234,
  "task_type": "security_audit"
}
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run safety tests (fail-closed verification)
pytest -m safety

# Run unit tests only
pytest -m unit

# Run integration tests (requires API keys)
pytest -m integration

# With coverage
pytest --cov=aperion_switchboard --cov-report=html
```

## 📊 Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | OpenAI-compatible chat |
| `/health` | GET | Health check |
| `/healthz` | GET | Kubernetes health probe |
| `/docs` | GET | OpenAPI documentation |

## 🏗️ Architecture

```
src/aperion_switchboard/
├── core/
│   ├── router.py      # Task routing & fallback logic
│   ├── protocol.py    # LLMClient abstract base class
│   └── fail_closed.py # Constitution A6 enforcement
├── providers/
│   ├── openai.py      # OpenAI/compatible providers
│   ├── gemini.py      # Google Gemini
│   ├── workers.py     # Cloudflare Workers AI
│   └── echo.py        # Test-only echo provider
├── service/
│   ├── app.py         # FastAPI application
│   ├── middleware.py  # Auth, telemetry, cost logging
│   └── schemas.py     # OpenAI-compatible Pydantic models
└── main.py            # Entry point
```

## 📈 Cost Optimization

The Switchboard achieves ~75% cost savings by:

1. Routing 80% of requests (docs, linting, tests) to free tiers
2. Reserving premium providers for critical tasks only
3. Tracking and reporting cost per request

View cost summary:

```python
from aperion_switchboard.core.router import get_router

router = get_router()
summary = router.get_cost_summary()
print(f"Savings: {summary['savings_percent']:.1f}%")
```

## 🔧 Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run linter
ruff check src tests

# Run type checker
mypy src

# Run tests with coverage
pytest --cov=aperion_switchboard --cov-report=term-missing
```

## License

MIT
