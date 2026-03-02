"""
Intelligent LLM Provider Router - Task-Based Routing and Cost Optimization.

Routes tasks to optimal providers based on:
- Task criticality (security, production → paid premium providers)
- Volume requirements (batch, docs → free tier providers)
- Speed needs (real-time → fastest available)
- Cost constraints (optimize spending across provider tiers)

Cost optimization target: 75% savings by routing 80% volume to free tiers.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .fail_closed import is_echo_allowed
from .protocol import LLMClient, ProviderError


class TaskType(Enum):
    """Task types for intelligent routing."""

    # Critical tasks → Premium provider (OpenAI)
    SECURITY_AUDIT = "security_audit"
    PRODUCTION_DECISION = "production_decision"
    STRATEGIC_ANALYSIS = "strategic_analysis"
    CODE_REVIEW = "code_review"

    # High-volume tasks → Free fast provider (Gemini/Groq)
    LINT_ANALYSIS = "lint_analysis"
    TEST_GENERATION = "test_generation"
    CODE_COMPLETION = "code_completion"
    ERROR_DIAGNOSIS = "error_diagnosis"

    # Documentation tasks → Free volume provider (Gemini)
    DOC_UPDATE = "doc_update"
    DOC_GENERATION = "doc_generation"
    SUMMARY_GENERATION = "summary_generation"
    API_DOCUMENTATION = "api_documentation"

    # Premium reasoning tasks → Anthropic (Claude)
    EXTENDED_THINKING = "extended_thinking"
    COMPLEX_ANALYSIS = "complex_analysis"

    # Development tasks → Local provider (Echo) ONLY when allowed
    TESTING = "testing"
    DEVELOPMENT = "development"
    OFFLINE_WORK = "offline_work"

    # General/unspecified → Default routing
    GENERAL = "general"


@dataclass(frozen=True)
class RoutingDecision:
    """Result of routing decision with full audit trail."""

    provider_name: str
    reason: str
    fallback_chain: list[str]
    estimated_cost_per_1m_tokens: float
    expected_latency: str  # "fast", "medium", "slow"
    task_type: TaskType = TaskType.GENERAL

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON logging."""
        return {
            "provider": self.provider_name,
            "reason": self.reason,
            "fallback_chain": self.fallback_chain,
            "estimated_cost_per_1m": self.estimated_cost_per_1m_tokens,
            "expected_latency": self.expected_latency,
            "task_type": self.task_type.value,
        }


@dataclass
class UsageStats:
    """Usage statistics for a single provider."""

    requests: int = 0
    tokens: int = 0
    cost_usd: float = 0.0
    errors: int = 0
    total_latency_ms: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        """Average latency per request."""
        return self.total_latency_ms / self.requests if self.requests > 0 else 0.0


class LLMRouter:
    """
    Intelligent task-based LLM provider router.

    Routes requests to optimal provider based on task characteristics.
    Tracks usage and costs per provider for observability.
    """

    # Provider tier definitions
    TIER_ANTHROPIC = ["anthropic"]  # Paid, advanced reasoning + caching
    TIER_PREMIUM = ["openai"]  # Paid, best quality
    TIER_CLOUDFLARE = ["workers_ai"]  # Low-cost Cloudflare Workers AI
    TIER_FREE_VOLUME = ["gemini"]  # Free, high volume (60 RPM)
    TIER_LOCAL = ["echo"]  # Always available, testing ONLY

    # Cost estimates per 1M tokens (input + output averaged)
    COST_PER_1M_TOKENS: dict[str, float] = {
        "anthropic": 3.00,  # Claude Sonnet input avg
        "openai": 0.30,  # GPT-4.1-mini average
        "workers_ai": 0.011,  # Cloudflare Workers AI
        "gemini": 0.00,  # Free tier
        "echo": 0.00,  # Local
    }

    # Task type → (preferred_provider, reason) mapping
    TASK_ROUTING: dict[TaskType, tuple[str, str]] = {
        TaskType.SECURITY_AUDIT: ("openai", "Critical security analysis requires premium"),
        TaskType.PRODUCTION_DECISION: ("openai", "High-stakes decision requires premium"),
        TaskType.STRATEGIC_ANALYSIS: ("openai", "Complex reasoning requires premium"),
        TaskType.CODE_REVIEW: ("openai", "Quality code review requires premium"),
        TaskType.LINT_ANALYSIS: ("gemini", "Fast batch processing - free tier"),
        TaskType.TEST_GENERATION: ("gemini", "High-volume generation - free tier"),
        TaskType.CODE_COMPLETION: ("gemini", "Speed-critical - free tier"),
        TaskType.ERROR_DIAGNOSIS: ("gemini", "Quick turnaround - free tier"),
        TaskType.DOC_UPDATE: ("gemini", "High-volume documentation - free tier"),
        TaskType.DOC_GENERATION: ("gemini", "Batch doc creation - free tier"),
        TaskType.SUMMARY_GENERATION: ("gemini", "Volume processing - free tier"),
        TaskType.API_DOCUMENTATION: ("gemini", "Documentation task - free tier"),
        TaskType.EXTENDED_THINKING: ("anthropic", "Deep reasoning requires extended thinking"),
        TaskType.COMPLEX_ANALYSIS: ("anthropic", "Multi-step analysis benefits from Claude"),
        TaskType.TESTING: ("echo", "Development/testing - local only"),
        TaskType.DEVELOPMENT: ("echo", "Local development - no API needed"),
        TaskType.OFFLINE_WORK: ("echo", "No API needed - local only"),
        TaskType.GENERAL: ("gemini", "Default routing - cost-optimized"),
    }

    def __init__(self, providers: dict[str, LLMClient] | None = None):
        """
        Initialize router with available providers.

        Args:
            providers: Optional dict of provider name -> instance.
                      If not provided, uses lazy loading.
        """
        self._providers = providers or {}
        self._usage_stats: dict[str, UsageStats] = {}
        self._init_stats()

    def _init_stats(self) -> None:
        """Initialize usage statistics for all known providers."""
        for provider_name in list(self.COST_PER_1M_TOKENS.keys()):
            self._usage_stats[provider_name] = UsageStats()

    def register_provider(self, name: str, provider: LLMClient) -> None:
        """Register a provider instance."""
        self._providers[name] = provider
        if name not in self._usage_stats:
            self._usage_stats[name] = UsageStats()

    def route(self, task_type: TaskType, fallback: bool = True) -> RoutingDecision:
        """
        Route task to optimal provider.

        Args:
            task_type: Type of task to perform
            fallback: Whether to build fallback chain

        Returns:
            RoutingDecision with provider name and routing metadata
        """
        # Get preferred provider for task type
        provider_name, reason = self.TASK_ROUTING.get(
            task_type, ("gemini", "Default routing")
        )

        # Special handling for local tasks - verify echo is allowed
        if provider_name == "echo" and not is_echo_allowed():
            # Redirect to free tier instead
            provider_name = "gemini"
            reason = f"[Redirected from echo] {reason}"

        # Build fallback chain based on task criticality
        if task_type in (
            TaskType.EXTENDED_THINKING,
            TaskType.COMPLEX_ANALYSIS,
        ):
            # Anthropic tasks: anthropic → openai → cloudflare → free
            fallback_chain = (
                self.TIER_ANTHROPIC
                + self.TIER_PREMIUM
                + self.TIER_CLOUDFLARE
                + self.TIER_FREE_VOLUME
            )
        elif task_type in (
            TaskType.SECURITY_AUDIT,
            TaskType.PRODUCTION_DECISION,
            TaskType.STRATEGIC_ANALYSIS,
        ):
            # Critical tasks: premium → anthropic → cloudflare → free
            fallback_chain = (
                self.TIER_PREMIUM
                + self.TIER_ANTHROPIC
                + self.TIER_CLOUDFLARE
                + self.TIER_FREE_VOLUME
            )
        else:
            # Volume tasks: free → cloudflare → premium → (local if allowed)
            fallback_chain = (
                self.TIER_FREE_VOLUME
                + self.TIER_CLOUDFLARE
                + self.TIER_PREMIUM
            )

        # Only add echo if explicitly allowed
        if is_echo_allowed():
            fallback_chain = fallback_chain + self.TIER_LOCAL

        # Estimate cost and latency
        estimated_cost = self.COST_PER_1M_TOKENS.get(provider_name, 0.0)

        if provider_name in self.TIER_FREE_VOLUME:
            expected_latency = "fast"
        elif provider_name in self.TIER_PREMIUM:
            expected_latency = "medium"
        elif provider_name in self.TIER_ANTHROPIC:
            expected_latency = "slow"  # Extended thinking can take longer
        else:
            expected_latency = "instant"

        return RoutingDecision(
            provider_name=provider_name,
            reason=reason,
            fallback_chain=fallback_chain if fallback else [provider_name],
            estimated_cost_per_1m_tokens=estimated_cost,
            expected_latency=expected_latency,
            task_type=task_type,
        )

    def get_provider(
        self, task_type: TaskType, fallback: bool = True
    ) -> tuple[LLMClient, RoutingDecision]:
        """
        Get provider instance for task type.

        Args:
            task_type: Type of task to perform
            fallback: Whether to use fallback chain on failure

        Returns:
            Tuple of (provider instance, routing decision)

        Raises:
            ProviderError: If no provider available in fallback chain
        """
        decision = self.route(task_type, fallback=fallback)

        # Try providers in fallback chain order
        for provider_name in decision.fallback_chain:
            if provider_name in self._providers:
                provider = self._providers[provider_name]
                if provider.is_configured:
                    return provider, decision

        # No configured provider found
        raise ProviderError(
            f"No configured provider available for task '{task_type.value}'",
            provider=decision.provider_name,
            recoverable=False,
        )

    def track_usage(
        self,
        provider_name: str,
        tokens: int,
        latency_ms: float,
        success: bool = True,
    ) -> None:
        """
        Track provider usage for cost monitoring.

        Args:
            provider_name: Name of provider used
            tokens: Number of tokens consumed (input + output)
            latency_ms: Request latency in milliseconds
            success: Whether request succeeded
        """
        if provider_name not in self._usage_stats:
            self._usage_stats[provider_name] = UsageStats()

        stats = self._usage_stats[provider_name]
        stats.requests += 1
        stats.tokens += tokens
        stats.total_latency_ms += latency_ms

        if not success:
            stats.errors += 1
        else:
            # Calculate cost (tokens are raw count, cost is per million)
            cost_per_token = self.COST_PER_1M_TOKENS.get(provider_name, 0.0) / 1_000_000
            stats.cost_usd += tokens * cost_per_token

    def get_stats(self) -> dict[str, dict[str, Any]]:
        """Get usage statistics for all providers."""
        return {
            name: {
                "requests": stats.requests,
                "tokens": stats.tokens,
                "cost_usd": stats.cost_usd,
                "errors": stats.errors,
                "avg_latency_ms": stats.avg_latency_ms,
            }
            for name, stats in self._usage_stats.items()
        }

    def get_cost_summary(self) -> dict[str, Any]:
        """
        Get cost summary across all providers.

        Returns:
            Dictionary with total cost, breakdown, and savings vs OpenAI-only
        """
        total_cost = sum(s.cost_usd for s in self._usage_stats.values())
        total_requests = sum(s.requests for s in self._usage_stats.values())
        total_tokens = sum(s.tokens for s in self._usage_stats.values())

        # Calculate what it would cost with OpenAI only
        openai_only_cost = total_tokens * self.COST_PER_1M_TOKENS["openai"] / 1_000_000
        savings = openai_only_cost - total_cost
        savings_percent = (savings / openai_only_cost * 100) if openai_only_cost > 0 else 0

        return {
            "total_cost_usd": round(total_cost, 6),
            "total_requests": total_requests,
            "total_tokens": total_tokens,
            "openai_only_cost_usd": round(openai_only_cost, 6),
            "savings_usd": round(savings, 6),
            "savings_percent": round(savings_percent, 2),
            "breakdown": {
                name: round(stats.cost_usd, 6)
                for name, stats in self._usage_stats.items()
            },
        }


# Global router instance (lazy initialized)
_router: LLMRouter | None = None


def get_router() -> LLMRouter:
    """Get global router instance (singleton)."""
    global _router
    if _router is None:
        _router = LLMRouter()
    return _router


def reset_router() -> None:
    """Reset global router (for testing)."""
    global _router
    _router = None


def route_task(
    task_type: TaskType | str,
    providers: dict[str, LLMClient] | None = None,
) -> tuple[LLMClient, RoutingDecision]:
    """
    Convenience function to route a task and get provider.

    This matches the aperion-legendary-ai API pattern.

    Args:
        task_type: TaskType enum or string (e.g., "security_audit")
        providers: Optional providers dict (uses global router if not provided)

    Returns:
        Tuple of (provider, routing decision)

    Example:
        provider, decision = route_task(TaskType.CODE_REVIEW)
        result = provider.chat("Review this code...")
    """
    # Convert string to enum if needed
    if isinstance(task_type, str):
        try:
            task_type = TaskType(task_type)
        except ValueError:
            task_type = TaskType.GENERAL

    if providers:
        router = LLMRouter(providers)
    else:
        router = get_router()

    return router.get_provider(task_type)
