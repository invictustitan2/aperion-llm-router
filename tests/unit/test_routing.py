"""
Unit Tests for LLM Router - Task-based routing and cost optimization.
"""

import pytest

from aperion_switchboard.core.router import (
    LLMRouter,
    RoutingDecision,
    TaskType,
    get_router,
    reset_router,
)
from aperion_switchboard.core.fail_closed import is_echo_allowed


@pytest.mark.unit
class TestTaskType:
    """Test TaskType enum."""

    def test_all_task_types_have_values(self):
        """All task types should have string values."""
        for task_type in TaskType:
            assert isinstance(task_type.value, str)
            assert len(task_type.value) > 0

    def test_critical_task_types_exist(self):
        """Critical task types must exist."""
        assert TaskType.SECURITY_AUDIT
        assert TaskType.PRODUCTION_DECISION
        assert TaskType.STRATEGIC_ANALYSIS

    def test_volume_task_types_exist(self):
        """Volume/documentation task types must exist."""
        assert TaskType.DOC_UPDATE
        assert TaskType.DOC_GENERATION
        assert TaskType.LINT_ANALYSIS

    def test_general_task_type_exists(self):
        """Default general task type must exist."""
        assert TaskType.GENERAL


@pytest.mark.unit
class TestRoutingDecision:
    """Test RoutingDecision dataclass."""

    def test_routing_decision_fields(self):
        """RoutingDecision should have all required fields."""
        decision = RoutingDecision(
            provider_name="openai",
            reason="Test reason",
            fallback_chain=["openai", "gemini"],
            estimated_cost_per_1m_tokens=0.30,
            expected_latency="medium",
            task_type=TaskType.SECURITY_AUDIT,
        )

        assert decision.provider_name == "openai"
        assert decision.reason == "Test reason"
        assert decision.fallback_chain == ["openai", "gemini"]
        assert decision.estimated_cost_per_1m_tokens == 0.30
        assert decision.expected_latency == "medium"
        assert decision.task_type == TaskType.SECURITY_AUDIT

    def test_routing_decision_to_dict(self):
        """RoutingDecision.to_dict() should return serializable dict."""
        decision = RoutingDecision(
            provider_name="gemini",
            reason="Free tier",
            fallback_chain=["gemini", "openai"],
            estimated_cost_per_1m_tokens=0.0,
            expected_latency="fast",
        )

        data = decision.to_dict()

        assert data["provider"] == "gemini"
        assert data["reason"] == "Free tier"
        assert data["task_type"] == "general"
        assert isinstance(data["fallback_chain"], list)


@pytest.mark.unit
class TestLLMRouter:
    """Test LLMRouter routing logic."""

    def test_router_initialization(self):
        """Router should initialize with empty stats."""
        router = LLMRouter()

        stats = router.get_stats()
        assert "openai" in stats
        assert "gemini" in stats
        assert "echo" in stats

        for provider_stats in stats.values():
            assert provider_stats["requests"] == 0
            assert provider_stats["tokens"] == 0

    def test_route_security_audit_to_premium(self):
        """Security audits should route to premium provider (OpenAI)."""
        router = LLMRouter()
        decision = router.route(TaskType.SECURITY_AUDIT)

        assert decision.provider_name == "openai"
        assert "security" in decision.reason.lower() or "premium" in decision.reason.lower()
        assert decision.estimated_cost_per_1m_tokens > 0

    def test_route_doc_update_to_free_tier(self):
        """Documentation updates should route to free tier (Gemini)."""
        router = LLMRouter()
        decision = router.route(TaskType.DOC_UPDATE)

        assert decision.provider_name == "gemini"
        assert decision.estimated_cost_per_1m_tokens == 0.0
        assert decision.expected_latency == "fast"

    def test_route_lint_analysis_to_free_tier(self):
        """Lint analysis should route to free tier."""
        router = LLMRouter()
        decision = router.route(TaskType.LINT_ANALYSIS)

        assert decision.provider_name == "gemini"
        assert decision.estimated_cost_per_1m_tokens == 0.0

    def test_route_testing_respects_echo_gating(self):
        """Testing tasks should respect echo gating."""
        router = LLMRouter()
        decision = router.route(TaskType.TESTING)

        # In test environment, echo is allowed
        if is_echo_allowed():
            # Echo should be in fallback chain
            assert "echo" in decision.fallback_chain
        else:
            # Echo should NOT be in fallback chain
            assert "echo" not in decision.fallback_chain

    def test_fallback_chain_for_critical_tasks(self):
        """Critical tasks should have premium-first fallback chain."""
        router = LLMRouter()
        decision = router.route(TaskType.SECURITY_AUDIT)

        # Premium should be first
        assert decision.fallback_chain[0] == "openai"
        # Other tiers should follow
        assert "gemini" in decision.fallback_chain or "workers_ai" in decision.fallback_chain

    def test_fallback_chain_for_volume_tasks(self):
        """Volume tasks should prioritize free tiers."""
        router = LLMRouter()
        decision = router.route(TaskType.DOC_GENERATION)

        # Free tier should come before premium
        if "gemini" in decision.fallback_chain and "openai" in decision.fallback_chain:
            gemini_idx = decision.fallback_chain.index("gemini")
            openai_idx = decision.fallback_chain.index("openai")
            assert gemini_idx < openai_idx

    def test_track_usage_updates_stats(self):
        """track_usage should update provider statistics."""
        router = LLMRouter()

        router.track_usage("openai", tokens=1000, latency_ms=2500, success=True)

        stats = router.get_stats()
        assert stats["openai"]["requests"] == 1
        assert stats["openai"]["tokens"] == 1000
        assert stats["openai"]["cost_usd"] > 0

    def test_track_usage_handles_errors(self):
        """track_usage should count errors separately."""
        router = LLMRouter()

        router.track_usage("gemini", tokens=500, latency_ms=1000, success=False)

        stats = router.get_stats()
        assert stats["gemini"]["errors"] == 1
        assert stats["gemini"]["requests"] == 1

    def test_cost_summary_calculates_savings(self):
        """get_cost_summary should calculate multi-provider savings."""
        router = LLMRouter()

        # Simulate realistic workload - mostly free, some paid
        router.track_usage("gemini", tokens=600000, latency_ms=15000, success=True)
        router.track_usage("openai", tokens=50000, latency_ms=8000, success=True)

        summary = router.get_cost_summary()

        assert summary["total_cost_usd"] >= 0
        assert summary["openai_only_cost_usd"] > summary["total_cost_usd"]
        assert summary["savings_usd"] > 0
        assert 0 <= summary["savings_percent"] <= 100

    def test_cost_summary_with_free_only(self):
        """Cost summary with only free providers should show ~100% savings."""
        router = LLMRouter()

        router.track_usage("gemini", tokens=300000, latency_ms=5000, success=True)

        summary = router.get_cost_summary()

        assert summary["total_cost_usd"] == 0.0
        assert summary["openai_only_cost_usd"] > 0
        assert summary["savings_percent"] == 100.0

    def test_all_task_types_have_routing(self):
        """All TaskType enums should have routing defined."""
        router = LLMRouter()

        for task_type in TaskType:
            decision = router.route(task_type)
            assert decision.provider_name is not None
            assert len(decision.fallback_chain) > 0

    @pytest.mark.parametrize(
        "task_type,expected_tier",
        [
            (TaskType.SECURITY_AUDIT, "premium"),
            (TaskType.PRODUCTION_DECISION, "premium"),
            (TaskType.LINT_ANALYSIS, "free_volume"),
            (TaskType.DOC_UPDATE, "free_volume"),
        ],
    )
    def test_task_routing_to_correct_tier(self, task_type, expected_tier):
        """Tasks should route to correct provider tier."""
        router = LLMRouter()
        decision = router.route(task_type)

        if expected_tier == "premium":
            assert decision.provider_name in router.TIER_PREMIUM
        elif expected_tier == "free_volume":
            assert decision.provider_name in router.TIER_FREE_VOLUME


@pytest.mark.unit
class TestGlobalRouter:
    """Test global router singleton."""

    def test_get_router_returns_instance(self):
        """get_router should return a router instance."""
        reset_router()
        router = get_router()

        assert router is not None
        assert isinstance(router, LLMRouter)

    def test_get_router_singleton(self):
        """get_router should return same instance."""
        reset_router()
        router1 = get_router()
        router2 = get_router()

        assert router1 is router2

    def test_reset_router_clears_singleton(self):
        """reset_router should clear the singleton."""
        router1 = get_router()
        reset_router()
        router2 = get_router()

        assert router1 is not router2
