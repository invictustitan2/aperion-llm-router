"""
Anthropic Provider - Claude Messages API with prompt caching and extended thinking.

Supports:
- Claude Messages API (v1/messages)
- Prompt caching via cache_control (ephemeral)
- Extended thinking (budget_tokens)
- Streaming via SSE

Environment Variables:
    ANTHROPIC_API_KEY     - API key for x-api-key header
    ANTHROPIC_BASE_URL    - API root (default: https://api.anthropic.com)
    ANTHROPIC_MODEL       - Model identifier (default: claude-sonnet-4-20250514)
    ANTHROPIC_TIMEOUT     - Request timeout in seconds (default: 60)
"""

import json
from collections.abc import AsyncIterator, Mapping
from typing import Any

from .base import BaseProvider


class AnthropicProvider(BaseProvider):
    """
    Anthropic Claude provider for the Messages API.

    Implements prompt caching (cache_control), extended thinking,
    and streaming via Anthropic's SSE format.
    """

    ENV_PREFIX = "ANTHROPIC"
    DEFAULT_BASE_URL = "https://api.anthropic.com"
    DEFAULT_MODEL = "claude-sonnet-4-20250514"
    DEFAULT_TIMEOUT = 60

    # Anthropic API version
    API_VERSION = "2023-06-01"

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def is_configured(self) -> bool:
        return bool(self._api_key)

    def _build_headers(self, correlation_id: str | None = None) -> dict[str, str]:
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "anthropic-version": self.API_VERSION,
        }
        if self._api_key:
            headers["x-api-key"] = self._api_key
        if correlation_id:
            headers["X-Correlation-ID"] = correlation_id
        return headers

    def _build_url(self) -> str:
        return f"{self._base_url.rstrip('/')}/v1/messages"

    def _build_payload(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []

        # System message — Claude uses top-level 'system' field, not a message
        system: list[dict[str, Any]] | str | None = None
        if persona := kwargs.get("persona"):
            system_block: dict[str, Any] = {
                "type": "text",
                "text": str(persona),
            }
            # Prompt caching: mark system message as cacheable
            if kwargs.get("cache_control"):
                system_block["cache_control"] = {"type": "ephemeral"}
            system = [system_block]

        # User message
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 4096),
        }

        if system is not None:
            payload["system"] = system

        # Extended thinking
        if thinking_budget := kwargs.get("thinking_budget"):
            payload["thinking"] = {
                "type": "enabled",
                "budget_tokens": int(thinking_budget),
            }

        # Forward supported parameters
        for param in ("temperature", "top_p", "stop_sequences", "top_k"):
            if param in kwargs:
                payload[param] = kwargs[param]

        # Note: temperature is not compatible with extended thinking
        # Let the API return the error if misused

        return payload

    def _parse_response(
        self, data: dict[str, Any]
    ) -> tuple[list[str], dict[str, Any]]:
        replies: list[str] = []

        for block in data.get("content", []):
            if block.get("type") == "text":
                replies.append(block["text"])
            elif block.get("type") == "thinking":
                # Extended thinking block — include as metadata, not reply
                pass

        usage = data.get("usage", {})
        return replies, usage

    async def _stream_impl(
        self, prompt: str, **kwargs: Any
    ) -> AsyncIterator[Mapping[str, Any]]:
        """Internal async generator for Anthropic SSE streaming."""
        url = self._build_url()
        headers = self._build_headers(kwargs.get("correlation_id"))
        payload = self._build_payload(prompt, **kwargs)
        payload["stream"] = True

        client = self._get_async_client()
        async with client.stream(
            "POST", url, headers=headers, json=payload
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue

                data_str = line[len("data: ") :]

                try:
                    event_data = json.loads(data_str)
                    event_type = event_data.get("type", "")

                    if event_type == "content_block_delta":
                        delta = event_data.get("delta", {})
                        if delta.get("type") == "text_delta":
                            yield {"chunk": delta.get("text", ""), "done": False}

                    elif event_type == "message_stop":
                        yield {"chunk": "", "done": True}
                        return

                    elif event_type == "message_delta":
                        # Final usage stats in message_delta
                        pass

                except (json.JSONDecodeError, KeyError):
                    continue

        yield {"chunk": "", "done": True}

    def stream_generate(
        self, prompt: str, **kwargs: Any
    ) -> AsyncIterator[Mapping[str, Any]]:
        return self._stream_impl(prompt, **kwargs)
