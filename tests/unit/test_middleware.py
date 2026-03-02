"""
Tests for request size limit middleware.

Validates that oversized requests are rejected before processing.
"""

import pytest
from fastapi.testclient import TestClient

from aperion_switchboard.service.app import create_app
from aperion_switchboard.service.middleware import MAX_REQUEST_BODY_SIZE


@pytest.fixture
def client():
    """Create test client with fresh rate limiter."""
    # Reset rate limiter to avoid cross-test interference
    from aperion_switchboard.core.rate_limit import RateLimitConfig, RateLimiter, set_rate_limiter
    set_rate_limiter(RateLimiter(RateLimitConfig(burst_size=100)))

    app = create_app()
    return TestClient(app)


class TestRequestSizeLimitMiddleware:
    """Tests for RequestSizeLimitMiddleware."""

    def test_normal_request_allowed(self, client):
        """Normal-sized request should be allowed."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "model": "gpt-4.1-mini"
            }
        )
        # Should get 200 (or other non-413 status)
        assert response.status_code != 413

    def test_large_content_length_rejected(self, client):
        """Request with Content-Length exceeding limit should be rejected."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello"}]
            },
            headers={"Content-Length": str(MAX_REQUEST_BODY_SIZE + 1)}
        )
        assert response.status_code == 413
        data = response.json()
        assert "error" in data
        assert "too large" in data["error"]["message"].lower()

    def test_health_endpoint_no_limit(self, client):
        """Health endpoints should not be affected."""
        response = client.get("/health")
        assert response.status_code == 200


class TestValidationErrorResponses:
    """Tests for validation error responses in the API."""

    def test_invalid_role_rejected(self, client):
        """Invalid message role should return 422."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "invalid_role", "content": "Hello"}]
            }
        )
        assert response.status_code == 422

    def test_missing_messages_rejected(self, client):
        """Request without messages should return 422."""
        response = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4"}
        )
        assert response.status_code == 422

    def test_temperature_out_of_range_rejected(self, client):
        """Temperature > 2.0 should return 422."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "temperature": 3.0
            }
        )
        assert response.status_code == 422
