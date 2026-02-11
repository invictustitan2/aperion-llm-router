"""
OpenAI Provider - OpenAI-compatible chat completions API.

Supports:
- Direct OpenAI API
- Azure OpenAI
- Any OpenAI-compatible endpoint (vLLM, LiteLLM, etc.)

Environment Variables:
    OPENAI_API_KEY     - Bearer token for Authorization header
    OPENAI_BASE_URL    - API root (default: https://api.openai.com/v1)
    OPENAI_MODEL       - Model identifier (default: gpt-4.1-mini)
    OPENAI_TIMEOUT     - Request timeout in seconds (default: 30)
"""

import json
from collections.abc import AsyncIterator, Mapping
from typing import Any

from .base import BaseProvider


class OpenAIProvider(BaseProvider):
    """
    OpenAI-compatible provider for chat completions API.

    Works with OpenAI, Azure OpenAI, vLLM, LiteLLM, and any endpoint
    implementing the OpenAI chat completions format.
    """

    ENV_PREFIX = "OPENAI"
    DEFAULT_BASE_URL = "https://api.openai.com/v1"
    DEFAULT_MODEL = "gpt-4.1-mini"
    DEFAULT_TIMEOUT = 30

    @property
    def name(self) -> str:
        return "openai"

    @property
    def is_configured(self) -> bool:
        return bool(self._api_key)

    def _build_headers(self, correlation_id: str | None = None) -> dict[str, str]:
        headers: dict[str, str] = {
            "Content-Type": "application/json",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        if correlation_id:
            headers["X-Correlation-ID"] = correlation_id
        return headers

    def _build_url(self) -> str:
        return f"{self._base_url.rstrip('/')}/chat/completions"

    def _build_payload(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        messages: list[dict[str, str]] = []

        # System message from persona kwarg (for council burst pattern)
        if persona := kwargs.get("persona"):
            messages.append({"role": "system", "content": str(persona)})

        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
        }

        # Forward supported OpenAI parameters
        for param in ("temperature", "max_tokens", "top_p", "stop", "n"):
            if param in kwargs:
                payload[param] = kwargs[param]

        return payload

    def _parse_response(self, data: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
        replies = [
            choice["message"]["content"]
            for choice in data.get("choices", [])
            if choice.get("message", {}).get("content")
        ]
        usage = data.get("usage", {})
        return replies, usage

    async def _stream_impl(
        self, prompt: str, **kwargs: Any
    ) -> AsyncIterator[Mapping[str, Any]]:
        """Internal async generator for SSE streaming."""
        url = self._build_url()
        headers = self._build_headers(kwargs.get("correlation_id"))
        payload = self._build_payload(prompt, **kwargs)
        payload["stream"] = True

        client = self._get_client()
        with client.stream("POST", url, headers=headers, json=payload) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line or not line.startswith("data: "):
                    continue

                data_str = line[len("data: "):]
                if data_str.strip() == "[DONE]":
                    yield {"chunk": "", "done": True}
                    return

                try:
                    chunk_data = json.loads(data_str)
                    delta = chunk_data["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        yield {"chunk": content, "done": False}
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

        yield {"chunk": "", "done": True}

    def stream_generate(
        self, prompt: str, **kwargs: Any
    ) -> AsyncIterator[Mapping[str, Any]]:
        return self._stream_impl(prompt, **kwargs)
