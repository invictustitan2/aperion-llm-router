"""
The Switchboard - Unified LLM API Gateway.

Constitution Compliance:
- A6: Fail-Closed Semantics - Never silently fall back to Echo in production
- B1: Secrets Management - All credentials via environment variables
- D1: Telemetry Injection - Correlation IDs propagated end-to-end
- D3: Structured Logging - JSON format for all cost/latency metrics
"""

__version__ = "0.1.0"
