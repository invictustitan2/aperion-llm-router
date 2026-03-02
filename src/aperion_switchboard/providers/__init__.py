"""LLM Provider implementations."""

import os
from typing import Any

import structlog

from .anthropic import AnthropicProvider
from .base import BaseProvider
from .echo import EchoProvider
from .gemini import GeminiProvider
from .openai import OpenAIProvider
from .workers import WorkersAIProvider

logger = structlog.get_logger(__name__)

__all__ = [
    "AnthropicProvider",
    "BaseProvider",
    "EchoProvider",
    "GeminiProvider",
    "OpenAIProvider",
    "WorkersAIProvider",
    "load_provider",
    "safe_provider_load",
    "provider_catalogue",
    "get_provider_class",
    "list_providers",
    "PROVIDER_REGISTRY",
]

# Provider registry for dynamic loading
PROVIDER_REGISTRY: dict[str, type[BaseProvider]] = {
    "anthropic": AnthropicProvider,
    "claude": AnthropicProvider,  # Alias
    "openai": OpenAIProvider,
    "gemini": GeminiProvider,
    "workers_ai": WorkersAIProvider,
    "workers": WorkersAIProvider,  # Alias
    "echo": EchoProvider,
}


def get_provider_class(name: str) -> type[BaseProvider] | None:
    """Get provider class by name."""
    return PROVIDER_REGISTRY.get(name.lower())


def list_providers() -> list[str]:
    """List all registered provider names."""
    return list(PROVIDER_REGISTRY.keys())


def load_provider(name: str, **kwargs: Any) -> BaseProvider:
    """
    Load and instantiate a provider by name.
    
    This is the primary factory function for creating providers,
    matching the aperion-legendary-ai API.
    
    Args:
        name: Provider name (e.g., 'openai', 'gemini', 'workers_ai', 'echo')
        **kwargs: Additional arguments passed to the provider constructor
        
    Returns:
        Instantiated provider
        
    Raises:
        ValueError: If provider name is unknown
        
    Example:
        provider = load_provider("workers_ai")
        result = provider.chat("Hello, world!")
    """
    provider_class = get_provider_class(name)
    if provider_class is None:
        raise ValueError(
            f"Unknown provider '{name}'. Available: {', '.join(list_providers())}"
        )
    
    return provider_class(**kwargs)


def safe_provider_load(name: str, **kwargs: Any) -> BaseProvider | None:
    """
    Safely load a provider, returning None if not available.
    
    Unlike load_provider(), this function:
    - Returns None if provider is not configured
    - Catches and logs initialization errors
    - Useful for graceful fallback chains
    
    Args:
        name: Provider name
        **kwargs: Additional arguments
        
    Returns:
        Provider instance or None if unavailable
    """
    try:
        provider = load_provider(name, **kwargs)
        if not provider.is_configured:
            logger.debug(
                "provider_not_configured",
                provider=name,
            )
            return None
        return provider
    except Exception as e:
        logger.warning(
            "provider_load_failed",
            provider=name,
            error=str(e),
        )
        return None


def provider_catalogue() -> list[dict[str, Any]]:
    """
    Get information about all available providers.
    
    Returns:
        List of provider info dicts with name, configured status, and health
    """
    catalogue = []
    for name in list_providers():
        try:
            provider = load_provider(name)
            info = provider.get_info()
            catalogue.append({
                "name": info.name,
                "description": info.description,
                "available": info.available,
                "configured": info.configured,
                "health": info.health.value,
            })
        except Exception as e:
            catalogue.append({
                "name": name,
                "description": "Failed to load",
                "available": False,
                "configured": False,
                "health": "unknown",
                "error": str(e),
            })
    return catalogue
