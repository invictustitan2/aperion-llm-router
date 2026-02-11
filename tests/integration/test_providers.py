"""
Integration Tests for LLM Providers.

These tests verify provider functionality with actual API calls.
Tests are skipped if the required API keys are not set.
"""

import os

import pytest

from aperion_switchboard.providers import (
    EchoProvider,
    GeminiProvider,
    OpenAIProvider,
    WorkersAIProvider,
)


@pytest.mark.integration
class TestEchoProvider:
    """Test Echo provider functionality (always available)."""

    def test_echo_chat_basic(self):
        """Test basic echo chat functionality."""
        provider = EchoProvider()

        result = provider.chat("Hello world")

        assert "replies" in result
        assert len(result["replies"]) > 0
        assert "Hello world" in result["replies"][0]
        assert result["provider"] == "echo"
        assert result["deterministic"] is True

    def test_echo_chat_with_parameters(self):
        """Test echo chat with parameters."""
        provider = EchoProvider()

        result = provider.chat(
            "Test message",
            temperature=0.5,
            top_k=20,
        )

        assert "parameters" in result
        assert result["parameters"]["temperature"] == 0.5
        assert result["parameters"]["top_k"] == 20

    def test_echo_complete(self):
        """Test echo completion functionality."""
        provider = EchoProvider()

        response = provider.complete("Complete this:")

        assert isinstance(response, str)
        assert len(response) > 0
        assert "Complete this:" in response

    def test_echo_is_always_configured(self):
        """Echo provider should always be configured."""
        provider = EchoProvider()
        assert provider.is_configured is True

    def test_echo_has_usage_stats(self):
        """Echo should return usage statistics."""
        provider = EchoProvider()

        result = provider.chat("Test prompt")

        assert "usage" in result
        assert "prompt_tokens" in result["usage"]
        assert "completion_tokens" in result["usage"]


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)
class TestOpenAIProvider:
    """Test OpenAI provider (requires API key)."""

    def test_openai_is_configured(self):
        """OpenAI should be configured when key is set."""
        provider = OpenAIProvider()
        assert provider.is_configured is True

    def test_openai_chat(self):
        """Test OpenAI chat functionality."""
        provider = OpenAIProvider()

        result = provider.chat("What is 2+2? Reply in one word.", max_tokens=10)

        assert "replies" in result
        assert len(result["replies"]) > 0
        assert result["provider"] == "openai"
        assert "usage" in result

    def test_openai_complete(self):
        """Test OpenAI completion."""
        provider = OpenAIProvider()

        response = provider.complete(
            "The capital of France is",
            max_tokens=10,
        )

        assert isinstance(response, str)
        assert len(response) > 0


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set",
)
class TestGeminiProvider:
    """Test Gemini provider (requires API key)."""

    def test_gemini_is_configured(self):
        """Gemini should be configured when key is set."""
        provider = GeminiProvider()
        assert provider.is_configured is True

    def test_gemini_chat(self):
        """Test Gemini chat functionality."""
        provider = GeminiProvider()

        result = provider.chat("What is 2+2? Reply in one word.", max_tokens=10)

        assert "replies" in result
        assert len(result["replies"]) > 0
        assert result["provider"] == "gemini"

    def test_gemini_complete(self):
        """Test Gemini completion."""
        provider = GeminiProvider()

        response = provider.complete("Hello", max_tokens=10)

        assert isinstance(response, str)


@pytest.mark.integration
@pytest.mark.skipif(
    not (os.getenv("WORKERS_AI_API_KEY") and os.getenv("WORKERS_AI_BASE_URL")),
    reason="WORKERS_AI_API_KEY or WORKERS_AI_BASE_URL not set",
)
class TestWorkersAIProvider:
    """Test Workers AI provider (requires API key and base URL)."""

    def test_workers_ai_is_configured(self):
        """Workers AI should be configured when credentials are set."""
        provider = WorkersAIProvider()
        assert provider.is_configured is True

    def test_workers_ai_chat(self):
        """Test Workers AI chat functionality."""
        provider = WorkersAIProvider()

        result = provider.chat("Say hello in one word.")

        assert "replies" in result
        assert len(result["replies"]) > 0
        assert result["provider"] == "workers_ai"


@pytest.mark.integration
class TestProviderProtocolCompliance:
    """Test all providers implement required protocol."""

    def test_echo_implements_protocol(self):
        """Echo provider should implement LLMClient protocol."""
        provider = EchoProvider()

        assert hasattr(provider, "name")
        assert hasattr(provider, "is_configured")
        assert hasattr(provider, "chat")
        assert hasattr(provider, "complete")
        assert callable(provider.chat)
        assert callable(provider.complete)

    def test_all_providers_return_structured_chat(self):
        """All configured providers should return structured chat responses."""
        provider = EchoProvider()

        result = provider.chat("Test prompt")

        assert isinstance(result, dict)
        assert "replies" in result
        assert isinstance(result["replies"], list)
        assert "provider" in result

    def test_all_providers_return_string_complete(self):
        """All configured providers should return string completions."""
        provider = EchoProvider()

        result = provider.complete("Test prompt")

        assert isinstance(result, str)
        assert len(result) > 0
