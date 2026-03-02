"""
Constitution A6: Fail-Closed Semantics Enforcement.

CRITICAL: This module enforces the Iron Rule that production systems
must NEVER silently fall back to the Echo provider.

If APERION_ALLOW_ECHO is not explicitly "true" AND no real provider
is configured, the service MUST:
1. Raise FailClosedError on startup
2. Return 503 Service Unavailable for all LLM requests
3. Log a CRITICAL error with remediation steps

This prevents silent failures where the system appears to work but
is actually returning deterministic echo responses instead of real
LLM output - a catastrophic failure mode for security/strategy tasks.
"""

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .protocol import LLMClient

logger = logging.getLogger(__name__)


class FailClosedError(RuntimeError):
    """
    Raised when fail-closed semantics are violated.

    This error indicates that the system would silently fall back to
    Echo provider in a production environment, which is prohibited.
    """

    def __init__(self, message: str, remediation: str | None = None):
        super().__init__(message)
        self.remediation = remediation or (
            "Configure at least one real LLM provider:\n"
            "  - Set OPENAI_API_KEY for OpenAI\n"
            "  - Set WORKERS_AI_API_KEY + WORKERS_AI_BASE_URL for Workers AI\n"
            "  - Set GEMINI_API_KEY for Google Gemini\n"
            "Or explicitly enable Echo fallback (TEST ONLY):\n"
            "  - Set APERION_ALLOW_ECHO=true"
        )


def is_echo_allowed() -> bool:
    """
    Check whether Echo fallback is permitted.

    Returns True (echo allowed) ONLY if:
    1. APERION_ALLOW_ECHO is explicitly set to "true", OR
    2. Running under pytest (PYTEST_CURRENT_TEST is set)

    In production (neither condition met), returns False.
    """
    # Always allow in test environments
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return True

    # Check explicit allow flag - must be exactly "true" (case-insensitive)
    allow_echo = os.environ.get("APERION_ALLOW_ECHO", "").lower()
    return allow_echo == "true"


def is_production_mode() -> bool:
    """Check if running in production mode."""
    # Explicit production flag
    if os.environ.get("APERION_PRODUCTION", "").lower() == "true":
        return True

    # Common production environment indicators
    env = os.environ.get("ENVIRONMENT", os.environ.get("ENV", "")).lower()
    if env in ("production", "prod", "live"):
        return True

    # If APERION_ALLOW_ECHO is explicitly false, treat as production
    if os.environ.get("APERION_ALLOW_ECHO", "").lower() == "false":
        return True

    return False


def check_fail_closed(providers: dict[str, "LLMClient"]) -> None:
    """
    Verify that fail-closed semantics are satisfied.

    This MUST be called at service startup. It will raise FailClosedError
    if the system would silently fall back to Echo in production.

    Args:
        providers: Dictionary of provider name -> provider instance

    Raises:
        FailClosedError: If no real providers configured and echo not allowed
    """
    # Identify which providers are real (not echo) and configured
    real_providers: list[str] = []
    unconfigured_providers: list[str] = []

    for name, provider in providers.items():
        if name == "echo":
            continue  # Skip echo provider

        if provider.is_configured:
            real_providers.append(name)
        else:
            unconfigured_providers.append(name)

    # If we have at least one real, configured provider, we're good
    if real_providers:
        logger.info(
            "Fail-closed check passed",
            extra={
                "configured_providers": real_providers,
                "unconfigured_providers": unconfigured_providers,
            },
        )
        return

    # No real providers configured - check if echo is allowed
    if is_echo_allowed():
        logger.warning(
            "⚠️  No real LLM providers configured - using Echo fallback. "
            "This is only acceptable for testing/development.",
            extra={
                "echo_allowed": True,
                "unconfigured_providers": unconfigured_providers,
            },
        )
        return

    # CRITICAL: No real providers AND echo not allowed = FAIL CLOSED
    logger.critical(
        "🚨 FAIL-CLOSED VIOLATION: No LLM providers configured and "
        "echo fallback is disabled. Service cannot start.",
        extra={
            "unconfigured_providers": unconfigured_providers,
            "echo_allowed": False,
            "production_mode": is_production_mode(),
        },
    )

    raise FailClosedError(
        "No LLM providers are configured and echo fallback is disabled. "
        "The Switchboard cannot start in this state (Constitution A6).",
    )


def get_safe_fallback_chain(
    requested: str,
    available_providers: dict[str, "LLMClient"],
    include_echo: bool = True,
) -> list[str]:
    """
    Build a safe fallback chain respecting fail-closed semantics.

    Args:
        requested: Originally requested provider name
        available_providers: All available provider instances
        include_echo: Whether to include echo (will be blocked if not allowed)

    Returns:
        Ordered list of provider names to try (only CONFIGURED providers)

    Raises:
        FailClosedError: If no configured providers and echo not allowed
    """
    chain: list[str] = []

    # Start with requested if available AND configured
    if requested in available_providers:
        provider = available_providers[requested]
        if provider.is_configured and requested != "echo":
            chain.append(requested)

    # Add other real, configured providers
    for name in available_providers:
        if name not in chain and name != "echo":
            provider = available_providers[name]
            if provider.is_configured:
                chain.append(name)

    # Handle echo at the end
    if include_echo and "echo" in available_providers:
        if is_echo_allowed():
            chain.append("echo")

    # Check fail-closed: no configured providers and echo blocked
    if not chain:
        if not is_echo_allowed():
            raise FailClosedError(
                f"Cannot build fallback chain for '{requested}': no real providers "
                "are configured and echo fallback is disabled.",
            )

    return chain
