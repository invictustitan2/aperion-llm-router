"""
Unit tests for the Anthropic Claude provider.

Tests cover:
- Configuration (API key detection)
- Header construction (x-api-key, anthropic-version)
- Payload format (Messages API, system as top-level)
- Prompt caching (cache_control injection)
- Extended thinking (thinking block in payload)
- Response parsing (text blocks, usage with cache tokens)
"""

import os
from unittest.mock import patch

import pytest

from aperion_switchboard.providers.anthropic import AnthropicProvider


@pytest.fixture
def provider():
    """Anthropic provider with test API key."""
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test-key"}):
        return AnthropicProvider()


@pytest.fixture
def unconfigured_provider():
    """Anthropic provider without API key."""
    with patch.dict(os.environ, {}, clear=True):
        env = {k: v for k, v in os.environ.items() if not k.startswith("ANTHROPIC")}
        with patch.dict(os.environ, env, clear=True):
            return AnthropicProvider()


class TestAnthropicConfiguration:
    def test_name(self, provider):
        assert provider.name == "anthropic"

    def test_is_configured_with_key(self, provider):
        assert provider.is_configured is True

    def test_is_not_configured_without_key(self, unconfigured_provider):
        assert unconfigured_provider.is_configured is False

    def test_default_model(self, provider):
        assert provider._model == "claude-sonnet-4-20250514"

    def test_default_base_url(self, provider):
        assert provider._base_url == "https://api.anthropic.com"

    def test_default_timeout(self, provider):
        assert provider._timeout == 60

    def test_custom_model(self):
        with patch.dict(
            os.environ,
            {
                "ANTHROPIC_API_KEY": "test",
                "ANTHROPIC_MODEL": "claude-opus-4-20250514",
            },
        ):
            p = AnthropicProvider()
            assert p._model == "claude-opus-4-20250514"


class TestAnthropicHeaders:
    def test_headers_include_api_key(self, provider):
        headers = provider._build_headers()
        assert headers["x-api-key"] == "sk-ant-test-key"

    def test_headers_include_version(self, provider):
        headers = provider._build_headers()
        assert headers["anthropic-version"] == "2023-06-01"

    def test_headers_include_content_type(self, provider):
        headers = provider._build_headers()
        assert headers["Content-Type"] == "application/json"

    def test_headers_include_correlation_id(self, provider):
        headers = provider._build_headers(correlation_id="corr-123")
        assert headers["X-Correlation-ID"] == "corr-123"

    def test_headers_without_correlation_id(self, provider):
        headers = provider._build_headers()
        assert "X-Correlation-ID" not in headers


class TestAnthropicPayload:
    def test_basic_payload(self, provider):
        payload = provider._build_payload("Hello, Claude!")
        assert payload["model"] == "claude-sonnet-4-20250514"
        assert payload["messages"] == [{"role": "user", "content": "Hello, Claude!"}]
        assert payload["max_tokens"] == 4096
        assert "system" not in payload

    def test_payload_with_system_message(self, provider):
        payload = provider._build_payload("Hello", persona="You are helpful.")
        assert payload["system"] == [{"type": "text", "text": "You are helpful."}]

    def test_payload_with_cache_control(self, provider):
        payload = provider._build_payload(
            "Hello", persona="You are helpful.", cache_control=True
        )
        system = payload["system"]
        assert len(system) == 1
        assert system[0]["cache_control"] == {"type": "ephemeral"}
        assert system[0]["text"] == "You are helpful."

    def test_cache_control_without_persona_has_no_system(self, provider):
        payload = provider._build_payload("Hello", cache_control=True)
        assert "system" not in payload

    def test_payload_with_extended_thinking(self, provider):
        payload = provider._build_payload("Analyze this", thinking_budget=10000)
        assert payload["thinking"] == {
            "type": "enabled",
            "budget_tokens": 10000,
        }

    def test_payload_with_custom_max_tokens(self, provider):
        payload = provider._build_payload("Hello", max_tokens=8192)
        assert payload["max_tokens"] == 8192

    def test_payload_forwards_temperature(self, provider):
        payload = provider._build_payload("Hello", temperature=0.7)
        assert payload["temperature"] == 0.7

    def test_payload_forwards_top_p(self, provider):
        payload = provider._build_payload("Hello", top_p=0.9)
        assert payload["top_p"] == 0.9

    def test_payload_forwards_stop_sequences(self, provider):
        payload = provider._build_payload("Hello", stop_sequences=["\n\n"])
        assert payload["stop_sequences"] == ["\n\n"]

    def test_payload_does_not_forward_unknown_params(self, provider):
        payload = provider._build_payload("Hello", unknown_param="x")
        assert "unknown_param" not in payload


class TestAnthropicResponseParsing:
    def test_parse_text_response(self, provider):
        data = {
            "content": [{"type": "text", "text": "Hello! How can I help?"}],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 8,
            },
        }
        replies, usage = provider._parse_response(data)
        assert replies == ["Hello! How can I help?"]
        assert usage["input_tokens"] == 10
        assert usage["output_tokens"] == 8

    def test_parse_multiple_text_blocks(self, provider):
        data = {
            "content": [
                {"type": "text", "text": "First part."},
                {"type": "text", "text": "Second part."},
            ],
            "usage": {"input_tokens": 5, "output_tokens": 10},
        }
        replies, usage = provider._parse_response(data)
        assert replies == ["First part.", "Second part."]

    def test_parse_thinking_block_excluded_from_replies(self, provider):
        data = {
            "content": [
                {"type": "thinking", "thinking": "Let me reason..."},
                {"type": "text", "text": "The answer is 42."},
            ],
            "usage": {"input_tokens": 20, "output_tokens": 15},
        }
        replies, usage = provider._parse_response(data)
        assert replies == ["The answer is 42."]

    def test_parse_cache_token_usage(self, provider):
        data = {
            "content": [{"type": "text", "text": "Cached response"}],
            "usage": {
                "input_tokens": 100,
                "output_tokens": 20,
                "cache_creation_input_tokens": 50,
                "cache_read_input_tokens": 30,
            },
        }
        replies, usage = provider._parse_response(data)
        assert usage["cache_creation_input_tokens"] == 50
        assert usage["cache_read_input_tokens"] == 30

    def test_parse_empty_content(self, provider):
        data = {"content": [], "usage": {}}
        replies, usage = provider._parse_response(data)
        assert replies == []

    def test_parse_missing_content(self, provider):
        data = {"usage": {}}
        replies, usage = provider._parse_response(data)
        assert replies == []


class TestAnthropicUrl:
    def test_default_url(self, provider):
        url = provider._build_url()
        assert url == "https://api.anthropic.com/v1/messages"

    def test_custom_base_url(self):
        with patch.dict(
            os.environ,
            {
                "ANTHROPIC_API_KEY": "test",
                "ANTHROPIC_BASE_URL": "https://custom.proxy.com",
            },
        ):
            p = AnthropicProvider()
            assert p._build_url() == "https://custom.proxy.com/v1/messages"

    def test_trailing_slash_stripped(self):
        with patch.dict(
            os.environ,
            {
                "ANTHROPIC_API_KEY": "test",
                "ANTHROPIC_BASE_URL": "https://api.anthropic.com/",
            },
        ):
            p = AnthropicProvider()
            assert p._build_url() == "https://api.anthropic.com/v1/messages"


class TestProviderRegistry:
    def test_anthropic_in_registry(self):
        from aperion_switchboard.providers import PROVIDER_REGISTRY

        assert "anthropic" in PROVIDER_REGISTRY

    def test_claude_alias_in_registry(self):
        from aperion_switchboard.providers import PROVIDER_REGISTRY

        assert "claude" in PROVIDER_REGISTRY
        assert PROVIDER_REGISTRY["claude"] is PROVIDER_REGISTRY["anthropic"]

    def test_load_provider_by_name(self):
        from aperion_switchboard.providers import load_provider

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test"}):
            p = load_provider("anthropic")
            assert p.name == "anthropic"

    def test_load_provider_by_alias(self):
        from aperion_switchboard.providers import load_provider

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test"}):
            p = load_provider("claude")
            assert p.name == "anthropic"
