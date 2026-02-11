# Authoritative Sources & References

> Curated documentation for LLM gateway development

## Official Provider Documentation

### OpenAI

| Topic | URL |
|-------|-----|
| API Reference | https://platform.openai.com/docs/api-reference |
| Error Codes | https://platform.openai.com/docs/guides/error-codes |
| Rate Limits | https://platform.openai.com/docs/guides/rate-limits |
| Streaming | https://platform.openai.com/docs/guides/streaming-responses |
| Best Practices | https://platform.openai.com/docs/guides/production-best-practices |

### Google Gemini

| Topic | URL |
|-------|-----|
| API Overview | https://ai.google.dev/gemini-api/docs |
| Models | https://ai.google.dev/gemini-api/docs/models |
| Rate Limits | https://ai.google.dev/gemini-api/docs/quota |
| Error Handling | https://ai.google.dev/gemini-api/docs/troubleshooting |

### Cloudflare Workers AI

| Topic | URL |
|-------|-----|
| API Reference | https://developers.cloudflare.com/workers-ai/ |
| Models | https://developers.cloudflare.com/workers-ai/models/ |
| REST API | https://developers.cloudflare.com/workers-ai/configuration/open-ai-compatibility/ |

---

## Framework Documentation

### FastAPI

| Topic | URL |
|-------|-----|
| Official Docs | https://fastapi.tiangolo.com/ |
| Deployment | https://fastapi.tiangolo.com/deployment/ |
| Middleware | https://fastapi.tiangolo.com/tutorial/middleware/ |
| Dependencies | https://fastapi.tiangolo.com/tutorial/dependencies/ |
| Testing | https://fastapi.tiangolo.com/tutorial/testing/ |

### HTTPX

| Topic | URL |
|-------|-----|
| Async Support | https://www.python-httpx.org/async/ |
| Connection Pools | https://www.python-httpx.org/advanced/#pool-limit-configuration |
| Timeouts | https://www.python-httpx.org/advanced/#timeout-configuration |
| Streaming | https://www.python-httpx.org/advanced/#streaming-responses |

### Pydantic

| Topic | URL |
|-------|-----|
| V2 Documentation | https://docs.pydantic.dev/latest/ |
| Settings Management | https://docs.pydantic.dev/latest/concepts/pydantic_settings/ |
| Validation | https://docs.pydantic.dev/latest/concepts/validators/ |

---

## Architectural References

### LLM Gateway Patterns

| Source | URL | Key Topics |
|--------|-----|------------|
| Collabnix | https://collabnix.com/llm-gateway-patterns-rate-limiting-and-load-balancing-guide/ | Rate limiting, load balancing |
| API7 | https://api7.ai/learning-center/api-gateway-guide/api-gateway-proxy-llm-requests | Proxy architecture |
| Apache APISIX | https://apisix.apache.org/ai-gateway/ | Plugin patterns, token accounting |
| AWS | https://aws.amazon.com/blogs/machine-learning/streamline-ai-operations-with-the-multi-provider-generative-ai-gateway-reference-architecture/ | Multi-provider architecture |

### Resilience Patterns

| Source | URL | Key Topics |
|--------|-----|------------|
| Maxim.ai | https://www.getmaxim.ai/articles/retries-fallbacks-and-circuit-breakers-in-llm-apps-a-production-guide/ | Circuit breakers, fallbacks |
| Orq.ai | https://docs.orq.ai/docs/proxy/retries | Retry configuration |
| DeepWiki | https://deepwiki.com/openai/openai-python/3.4-error-handling-and-retry-logic | Error handling |

### Production Deployment

| Source | URL | Key Topics |
|--------|-----|------------|
| Render | https://render.com/articles/fastapi-production-deployment-best-practices | Gunicorn, workers |
| PyTutorial | https://pytutorial.com/fastapi-performance-optimization-guide/ | Caching, async |
| Dev.to | https://dev.to/nithinbharathwaj/advanced-fastapi-patterns-building-production-ready-apis-with-python-2024-guide-2mf9 | Advanced patterns |

---

## Related Open Source Projects

### LLM Gateways

| Project | URL | Notes |
|---------|-----|-------|
| LiteLLM | https://github.com/BerriAI/litellm | Multi-provider, feature-rich |
| OpenAI Proxy (fangwentong) | https://github.com/fangwentong/openai-proxy | Streaming, logging |
| OpenAI Proxy (wujianguo) | https://github.com/wujianguo/openai-proxy | Flask, SSE |

### Resilience Libraries

| Library | URL | Use Case |
|---------|-----|----------|
| tenacity | https://github.com/jd/tenacity | Retry with backoff |
| circuitbreaker | https://github.com/fabfuel/circuitbreaker | Circuit breaker decorator |
| stamina | https://github.com/hynek/stamina | Modern retry library |

### Observability

| Library | URL | Use Case |
|---------|-----|----------|
| structlog | https://www.structlog.org/ | Structured logging |
| prometheus-client | https://github.com/prometheus/client_python | Metrics |
| opentelemetry-python | https://github.com/open-telemetry/opentelemetry-python | Tracing |

---

## Aperion-Specific References

### Internal Documentation

| Document | Path | Description |
|----------|------|-------------|
| Architecture | `docs/ARCHITECTURE.md` | System overview |
| LLM Provider Refresh | `docs/llm_provider_refresh.md` | Provider lifecycle |
| Testing Guidelines | `testing/README.md` | Test patterns |
| Copilot Guide | `docs/ci/COPILOT_GUIDE.md` | AI coding guidelines |

### Constitution References

| Constitution | Topic | Relevance |
|--------------|-------|-----------|
| A6 | Fail-Closed Semantics | Echo blocking in production |
| B1 | Secrets Management | API key handling |
| D1 | Telemetry Injection | Correlation ID propagation |
| D3 | Structured Logging | JSON log format |

---

## Books & Papers

| Title | Author | Topics |
|-------|--------|--------|
| "Building Microservices" | Sam Newman | Circuit breakers, resilience |
| "Release It!" | Michael Nygard | Stability patterns |
| "Designing Data-Intensive Applications" | Martin Kleppmann | Distributed systems |

---

## Last Updated

2026-02-08

## Maintenance Notes

- Review provider documentation quarterly for API changes
- Check for new resilience patterns annually
- Update version compatibility as libraries evolve
