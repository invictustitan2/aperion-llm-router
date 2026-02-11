"""
Cloudflare Workers AI Provider.

Supports both:
- Direct API: https://api.cloudflare.com/client/v4/accounts/{ACCT}/ai/run
- AI Gateway: https://gateway.ai.cloudflare.com/v1/{ACCT}/{GW}/workers-ai

Environment Variables:
    WORKERS_AI_API_KEY     - Cloudflare API token
    WORKERS_AI_CF_AIG_TOKEN - AI Gateway token (optional)
    WORKERS_AI_BASE_URL    - API base URL (required)
    WORKERS_AI_MODEL       - Model identifier (default: @cf/meta/llama-3.1-8b-instruct)
    WORKERS_AI_TIMEOUT     - Request timeout in seconds (default: 30)
"""

from typing import Any

from .base import BaseProvider


class WorkersAIProvider(BaseProvider):
    """
    Cloudflare Workers AI provider.

    Low-cost option using Cloudflare's edge AI infrastructure.
    Response format: {result: {response: "...", usage: {...}}}
    """

    ENV_PREFIX = "WORKERS_AI"
    DEFAULT_BASE_URL = ""  # Must be set via environment
    DEFAULT_MODEL = "@cf/meta/llama-3.1-8b-instruct"
    DEFAULT_TIMEOUT = 30

    def __init__(self) -> None:
        super().__init__()
        import os
        self._cf_aig_token = os.environ.get(f"{self.ENV_PREFIX}_CF_AIG_TOKEN", "")

    @property
    def name(self) -> str:
        return "workers_ai"

    @property
    def is_configured(self) -> bool:
        has_auth = bool(self._api_key) or bool(self._cf_aig_token)
        has_url = bool(self._base_url)
        return has_auth and has_url

    def _build_headers(self, correlation_id: str | None = None) -> dict[str, str]:
        headers: dict[str, str] = {
            "Content-Type": "application/json",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        if self._cf_aig_token:
            headers["cf-aig-authorization"] = f"Bearer {self._cf_aig_token}"
        if correlation_id:
            headers["X-Correlation-ID"] = correlation_id
        return headers

    def _build_url(self) -> str:
        # Workers AI URL format: {base_url}/{model}
        return f"{self._base_url.rstrip('/')}/{self._model}"

    def _build_payload(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        # Guard: Ensure prompt is a string
        if isinstance(prompt, (list, dict)):
            raise TypeError(
                "WorkersAIProvider.chat() expects a string prompt, not a list/dict."
            )

        messages: list[dict[str, str]] = []

        # System message from persona
        if persona := kwargs.get("persona"):
            messages.append({"role": "system", "content": f"You are {persona}."})

        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {"messages": messages}

        # Add max_tokens (default 1024 for Workers AI)
        payload["max_tokens"] = kwargs.get("max_tokens", 1024)

        return payload

    def _parse_response(self, data: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
        # Workers AI format: {result: {response: "...", usage: {...}}}
        result_data = data.get("result", {})
        text = result_data.get("response", "")
        usage = result_data.get("usage", {})

        replies = [text] if text else []
        return replies, usage
