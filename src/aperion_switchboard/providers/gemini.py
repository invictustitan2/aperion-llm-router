"""
Google Gemini Provider - Google's Generative AI API.

Environment Variables:
    GEMINI_API_KEY     - Google API key
    GEMINI_BASE_URL    - API root (default: https://generativelanguage.googleapis.com/v1beta)
    GEMINI_MODEL       - Model identifier (default: gemini-1.5-flash)
    GEMINI_TIMEOUT     - Request timeout in seconds (default: 30)
"""

from typing import Any

from .base import BaseProvider


class GeminiProvider(BaseProvider):
    """
    Google Gemini provider for Generative AI API.

    Free tier with 60 requests per minute - ideal for high-volume,
    non-critical tasks like documentation and linting.
    """

    ENV_PREFIX = "GEMINI"
    DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
    DEFAULT_MODEL = "gemini-1.5-flash"
    DEFAULT_TIMEOUT = 30

    @property
    def name(self) -> str:
        return "gemini"

    @property
    def is_configured(self) -> bool:
        return bool(self._api_key)

    def _build_headers(self, correlation_id: str | None = None) -> dict[str, str]:
        headers: dict[str, str] = {
            "Content-Type": "application/json",
        }
        if correlation_id:
            headers["X-Correlation-ID"] = correlation_id
        return headers

    def _build_url(self) -> str:
        # Gemini API format: /models/{model}:generateContent?key={api_key}
        return (
            f"{self._base_url.rstrip('/')}/models/{self._model}:generateContent"
            f"?key={self._api_key}"
        )

    def _build_payload(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        contents: list[dict[str, Any]] = []

        # System instruction (persona)
        system_instruction = None
        if persona := kwargs.get("persona"):
            system_instruction = {"parts": [{"text": str(persona)}]}

        # User content
        contents.append({"role": "user", "parts": [{"text": prompt}]})

        payload: dict[str, Any] = {"contents": contents}

        if system_instruction:
            payload["systemInstruction"] = system_instruction

        # Generation config
        generation_config: dict[str, Any] = {}
        if "temperature" in kwargs:
            generation_config["temperature"] = kwargs["temperature"]
        if "max_tokens" in kwargs:
            generation_config["maxOutputTokens"] = kwargs["max_tokens"]
        if "top_p" in kwargs:
            generation_config["topP"] = kwargs["top_p"]
        if "stop" in kwargs:
            generation_config["stopSequences"] = kwargs["stop"]

        if generation_config:
            payload["generationConfig"] = generation_config

        return payload

    def _parse_response(self, data: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
        replies: list[str] = []

        candidates = data.get("candidates", [])
        for candidate in candidates:
            content = candidate.get("content", {})
            parts = content.get("parts", [])
            for part in parts:
                if text := part.get("text"):
                    replies.append(text)

        # Gemini usage metadata
        usage_metadata = data.get("usageMetadata", {})
        usage = {
            "prompt_tokens": usage_metadata.get("promptTokenCount", 0),
            "completion_tokens": usage_metadata.get("candidatesTokenCount", 0),
            "total_tokens": usage_metadata.get("totalTokenCount", 0),
        }

        return replies, usage
