"""
Echo Provider - Deterministic testing/development provider.

⚠️  WARNING: This provider MUST be strictly gated in production.

The Echo provider is ONLY for:
- Unit tests requiring deterministic responses
- Local development without API access
- Debugging/diagnostics

Constitution A6 (Fail-Closed) prohibits silent fallback to Echo
in production environments. Use of Echo triggers a warning and
requires explicit APERION_ALLOW_ECHO=true.

Environment Variables:
    APERION_ALLOW_ECHO - Must be "true" to enable (default: varies by context)
"""

import asyncio
from collections.abc import AsyncIterator, Mapping
from typing import Any

from ..core.protocol import LLMClient, ProviderHealth


class EchoProvider(LLMClient):
    """
    Deterministic echo provider for testing and diagnostics.

    ⚠️  MUST be gated via APERION_ALLOW_ECHO in production.

    Echoes back user input with minimal processing, useful for:
    - Testing: Predictable responses for assertions
    - Debugging: Verifying request/response flow
    - Offline: Development without API keys

    Note: All delays use asyncio.sleep() to avoid blocking the event loop.
    """

    @property
    def name(self) -> str:
        return "echo"

    @property
    def is_configured(self) -> bool:
        # Echo is always "configured" - gating is via fail_closed.py
        return True

    def _build_response(
        self, prompt: str, processing_time: float, **kwargs: Any
    ) -> Mapping[str, Any]:
        """Build the echo response (shared logic)."""
        context = kwargs.get("context", {})
        temperature = kwargs.get("temperature", 0.0)
        top_k = kwargs.get("top_k", 1)

        # Generate deterministic echo response
        reply = f"Echo: {prompt}"

        # Add parameter information if non-default
        param_info: list[str] = []
        if temperature > 0:
            param_info.append(f"temp={temperature}")
        if top_k != 1:
            param_info.append(f"top_k={top_k}")

        if param_info:
            reply += f" [params: {', '.join(param_info)}]"

        if context:
            reply += f" (Context: {len(context)} keys)"

        return {
            "replies": [reply],
            "provider": self.name,
            "model": "echo-v1",
            "prompt_length": len(prompt),
            "processing_time": processing_time,
            "context_keys": list(context.keys()) if context else [],
            "parameters": {"temperature": temperature, "top_k": top_k},
            "deterministic": True,
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(reply.split()),
                "total_tokens": len(prompt.split()) + len(reply.split()),
            },
        }

    def chat(self, prompt: str, **kwargs: Any) -> Mapping[str, Any]:
        """
        Synchronous chat - for legacy/testing contexts only.

        Note: Does NOT simulate delay to avoid blocking event loops.
        For async contexts, use async_generate() instead.
        """
        return self._build_response(prompt, processing_time=0.0, **kwargs)

    def complete(self, prompt: str, **kwargs: Any) -> str:
        """Generate raw text completion by echoing input."""
        return f"Completion: {prompt}"

    async def async_generate(self, prompt: str, **kwargs: Any) -> Mapping[str, Any]:
        """
        Async generation - yields to event loop during simulated delay.

        This is the preferred method in async contexts (FastAPI handlers).
        """
        # Non-blocking delay simulation
        await asyncio.sleep(0.05)
        return self._build_response(prompt, processing_time=0.05, **kwargs)

    async def _stream_gen(self, prompt: str) -> AsyncIterator[Mapping[str, Any]]:
        """Internal streaming generator with non-blocking delays."""
        text = f"Streaming response to: {prompt}"
        for word in text.split():
            await asyncio.sleep(0.01)  # Non-blocking inter-chunk delay
            yield {"chunk": word + " ", "done": False}
        yield {"chunk": "", "done": True}

    def stream_generate(
        self, prompt: str, **kwargs: Any
    ) -> AsyncIterator[Mapping[str, Any]]:
        return self._stream_gen(prompt)

    def health_check(self) -> ProviderHealth:
        """Echo is always healthy."""
        return ProviderHealth.HEALTHY
