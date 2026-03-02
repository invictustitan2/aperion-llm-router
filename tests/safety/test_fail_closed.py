"""
CRITICAL: Fail-Closed Semantics Tests (Constitution A6).

These tests verify that The Switchboard NEVER silently falls back to
the Echo provider in production mode. This is the Iron Rule.

A silent fallback to Echo is a catastrophic failure because:
1. Security audits would receive deterministic echoes, not real analysis
2. Strategic decisions would be made on fake LLM output
3. The system would appear to work while providing no value

Test Categories:
1. Startup behavior - Service must crash if no providers and echo blocked
2. Fallback chain - Echo must be excluded when APERION_ALLOW_ECHO != "true"
3. Request handling - 503 must be returned, never echo fallback
"""

import os
from unittest.mock import patch

import pytest

from aperion_switchboard.core.fail_closed import (
    FailClosedError,
    check_fail_closed,
    get_safe_fallback_chain,
    is_echo_allowed,
    is_production_mode,
)
from aperion_switchboard.core.protocol import LLMClient


class MockProvider(LLMClient):
    """Mock provider for testing."""

    def __init__(self, name: str, configured: bool = True):
        self._name = name
        self._configured = configured

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_configured(self) -> bool:
        return self._configured

    def chat(self, prompt, **kwargs):
        return {"replies": [f"Mock: {prompt}"], "provider": self._name}

    def complete(self, prompt, **kwargs):
        return f"Mock complete: {prompt}"

    async def async_generate(self, prompt, **kwargs):
        return self.chat(prompt, **kwargs)

    def stream_generate(self, prompt, **kwargs):
        raise NotImplementedError()


@pytest.mark.safety
class TestIsEchoAllowed:
    """Test the is_echo_allowed() gating function."""

    def test_echo_allowed_when_explicitly_true(self):
        """Echo should be allowed when APERION_ALLOW_ECHO=true."""
        with patch.dict(os.environ, {"APERION_ALLOW_ECHO": "true"}, clear=False):
            # Remove pytest marker temporarily
            with patch.dict(os.environ, {"PYTEST_CURRENT_TEST": ""}, clear=False):
                os.environ.pop("PYTEST_CURRENT_TEST", None)
                assert is_echo_allowed() is True

    def test_echo_allowed_when_true_uppercase(self):
        """Echo should be allowed when APERION_ALLOW_ECHO=TRUE (case insensitive)."""
        with patch.dict(os.environ, {"APERION_ALLOW_ECHO": "TRUE"}, clear=False):
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("PYTEST_CURRENT_TEST", None)
                assert is_echo_allowed() is True

    def test_echo_blocked_when_explicitly_false(self):
        """Echo must be blocked when APERION_ALLOW_ECHO=false."""
        env = {"APERION_ALLOW_ECHO": "false"}
        with patch.dict(os.environ, env, clear=True):
            assert is_echo_allowed() is False

    def test_echo_blocked_when_not_set_in_production(self):
        """Echo must be blocked when env var not set (production default)."""
        # Clear all relevant env vars
        env = {}
        with patch.dict(os.environ, env, clear=True):
            assert is_echo_allowed() is False

    def test_echo_allowed_in_pytest(self):
        """Echo is always allowed when running under pytest."""
        # PYTEST_CURRENT_TEST is set by pytest
        assert os.environ.get("PYTEST_CURRENT_TEST") is not None
        assert is_echo_allowed() is True


@pytest.mark.safety
class TestIsProductionMode:
    """Test production mode detection."""

    def test_production_when_aperion_production_true(self):
        """Should detect production when APERION_PRODUCTION=true."""
        with patch.dict(os.environ, {"APERION_PRODUCTION": "true"}, clear=True):
            assert is_production_mode() is True

    def test_production_when_environment_is_prod(self):
        """Should detect production from ENVIRONMENT=production."""
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}, clear=True):
            assert is_production_mode() is True

    def test_production_when_env_is_live(self):
        """Should detect production from ENV=live."""
        with patch.dict(os.environ, {"ENV": "live"}, clear=True):
            assert is_production_mode() is True

    def test_production_when_echo_explicitly_disabled(self):
        """Should detect production when APERION_ALLOW_ECHO=false."""
        with patch.dict(os.environ, {"APERION_ALLOW_ECHO": "false"}, clear=True):
            assert is_production_mode() is True

    def test_not_production_when_no_indicators(self):
        """Should not detect production with no env vars."""
        with patch.dict(os.environ, {}, clear=True):
            assert is_production_mode() is False


@pytest.mark.safety
class TestCheckFailClosed:
    """Test the fail-closed startup check."""

    def test_passes_with_configured_real_provider(self):
        """Should pass when at least one real provider is configured."""
        providers = {
            "openai": MockProvider("openai", configured=True),
            "echo": MockProvider("echo", configured=True),
        }
        # Should not raise
        check_fail_closed(providers)

    def test_passes_with_echo_allowed_and_no_real_providers(self):
        """Should pass when echo is explicitly allowed and no real providers."""
        providers = {
            "openai": MockProvider("openai", configured=False),
            "echo": MockProvider("echo", configured=True),
        }
        # In pytest, echo is always allowed
        check_fail_closed(providers)

    def test_fails_with_no_providers_and_echo_blocked(self):
        """CRITICAL: Must raise when no real providers and echo blocked."""
        providers = {
            "openai": MockProvider("openai", configured=False),
            "gemini": MockProvider("gemini", configured=False),
            "echo": MockProvider("echo", configured=True),
        }

        # Simulate production environment
        env = {"APERION_ALLOW_ECHO": "false"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(FailClosedError) as exc_info:
                check_fail_closed(providers)

            # Verify error has remediation info
            assert exc_info.value.remediation is not None
            assert "OPENAI_API_KEY" in exc_info.value.remediation

    def test_fail_closed_error_contains_remediation(self):
        """FailClosedError must include actionable remediation steps."""
        providers = {"echo": MockProvider("echo", configured=True)}

        env = {"APERION_ALLOW_ECHO": "false"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(FailClosedError) as exc_info:
                check_fail_closed(providers)

            remediation = exc_info.value.remediation
            # Must include at least one way to fix
            assert "OPENAI_API_KEY" in remediation or "GEMINI_API_KEY" in remediation
            assert "APERION_ALLOW_ECHO" in remediation


@pytest.mark.safety
class TestGetSafeFallbackChain:
    """Test fallback chain building with fail-closed semantics."""

    def test_fallback_chain_excludes_echo_when_blocked(self):
        """Fallback chain must NOT include echo when blocked."""
        providers = {
            "openai": MockProvider("openai", configured=True),
            "gemini": MockProvider("gemini", configured=True),
            "echo": MockProvider("echo", configured=True),
        }

        env = {"APERION_ALLOW_ECHO": "false"}
        with patch.dict(os.environ, env, clear=True):
            chain = get_safe_fallback_chain(
                "openai", providers, include_echo=True
            )
            assert "echo" not in chain
            assert "openai" in chain
            assert "gemini" in chain

    def test_fallback_chain_includes_echo_when_allowed(self):
        """Fallback chain includes echo when explicitly allowed."""
        providers = {
            "openai": MockProvider("openai", configured=True),
            "echo": MockProvider("echo", configured=True),
        }

        env = {"APERION_ALLOW_ECHO": "true"}
        with patch.dict(os.environ, env, clear=True):
            os.environ.pop("PYTEST_CURRENT_TEST", None)
            chain = get_safe_fallback_chain(
                "openai", providers, include_echo=True
            )
            assert "echo" in chain

    def test_fallback_chain_raises_when_empty_and_echo_blocked(self):
        """Must raise FailClosedError when chain would be empty."""
        providers = {
            "openai": MockProvider("openai", configured=False),
            "echo": MockProvider("echo", configured=True),
        }

        env = {"APERION_ALLOW_ECHO": "false"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(FailClosedError):
                get_safe_fallback_chain("openai", providers, include_echo=True)

    def test_unconfigured_providers_excluded(self):
        """Unconfigured real providers should be excluded from chain."""
        providers = {
            "openai": MockProvider("openai", configured=False),
            "gemini": MockProvider("gemini", configured=True),
            "echo": MockProvider("echo", configured=True),
        }

        chain = get_safe_fallback_chain("openai", providers, include_echo=True)
        # openai is unconfigured, should not be in chain
        # (unless it was specifically requested and is first)
        assert "gemini" in chain


@pytest.mark.safety
class TestFailClosedIntegration:
    """Integration tests for fail-closed behavior."""

    def test_full_production_scenario(self):
        """Test complete production fail-closed scenario."""
        # Simulate: No API keys configured, production mode
        providers = {
            "openai": MockProvider("openai", configured=False),
            "gemini": MockProvider("gemini", configured=False),
            "workers_ai": MockProvider("workers_ai", configured=False),
            "echo": MockProvider("echo", configured=True),
        }

        env = {
            "APERION_PRODUCTION": "true",
            "APERION_ALLOW_ECHO": "false",
        }

        with patch.dict(os.environ, env, clear=True):
            # 1. Startup check must fail
            with pytest.raises(FailClosedError) as exc_info:
                check_fail_closed(providers)

            assert "Constitution A6" in str(exc_info.value)

            # 2. Fallback chain must fail
            with pytest.raises(FailClosedError):
                get_safe_fallback_chain("openai", providers, include_echo=True)

    def test_development_scenario_with_echo(self):
        """Test development scenario where echo is explicitly allowed."""
        providers = {
            "openai": MockProvider("openai", configured=False),
            "echo": MockProvider("echo", configured=True),
        }

        env = {"APERION_ALLOW_ECHO": "true"}

        with patch.dict(os.environ, env, clear=True):
            os.environ.pop("PYTEST_CURRENT_TEST", None)

            # Startup check should pass with warning
            check_fail_closed(providers)

            # Fallback chain should include echo
            chain = get_safe_fallback_chain("openai", providers, include_echo=True)
            assert "echo" in chain

    def test_pytest_always_allows_echo(self):
        """Pytest environment must always allow echo for testing."""
        # This test runs under pytest, so PYTEST_CURRENT_TEST is set
        assert is_echo_allowed() is True

        providers = {
            "echo": MockProvider("echo", configured=True),
        }

        # Should not raise even with no real providers
        check_fail_closed(providers)
