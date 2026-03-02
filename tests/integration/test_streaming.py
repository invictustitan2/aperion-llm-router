"""
Tests for SSE streaming functionality.

Verifies OpenAI-compatible streaming responses.
"""

import json
import os

import pytest
from fastapi.testclient import TestClient

# Set environment for testing
os.environ["APERION_ALLOW_ECHO"] = "true"
os.environ.setdefault("PYTEST_CURRENT_TEST", "test_streaming")


class TestStreamingEndpoint:
    """Tests for the streaming chat completions endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from aperion_switchboard.service.app import create_app

        app = create_app()
        with TestClient(app) as client:
            yield client

    def test_streaming_returns_sse_content_type(self, client):
        """Streaming request returns text/event-stream content type."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4.1-mini",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            },
        )

        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

    def test_streaming_includes_correlation_id(self, client):
        """Streaming response includes X-Correlation-ID header."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4.1-mini",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            },
        )

        assert "X-Correlation-ID" in response.headers

    def test_streaming_includes_provider_header(self, client):
        """Streaming response includes X-Switchboard-Provider header."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4.1-mini",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            },
        )

        assert "X-Switchboard-Provider" in response.headers
        assert response.headers["X-Switchboard-Provider"] == "echo"

    def test_streaming_yields_valid_sse_format(self, client):
        """Streaming response yields valid SSE data lines."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4.1-mini",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            },
        )

        content = response.text
        lines = [ln for ln in content.split("\n") if ln.startswith("data:")]

        assert len(lines) > 0, "Should have at least one data line"
        assert any("[DONE]" in ln for ln in lines), "Should end with [DONE]"

    def test_streaming_chunks_are_valid_json(self, client):
        """Each streaming chunk (except [DONE]) is valid JSON."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4.1-mini",
                "messages": [{"role": "user", "content": "Hello world"}],
                "stream": True,
            },
        )

        content = response.text
        lines = [ln for ln in content.split("\n") if ln.startswith("data:")]

        for line in lines:
            data = line.removeprefix("data:").strip()
            if data == "[DONE]":
                continue

            # Should parse as valid JSON
            parsed = json.loads(data)
            assert "id" in parsed
            assert "object" in parsed
            assert parsed["object"] == "chat.completion.chunk"
            assert "choices" in parsed

    def test_streaming_contains_content_delta(self, client):
        """Streaming chunks contain content in delta field."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4.1-mini",
                "messages": [{"role": "user", "content": "Test message"}],
                "stream": True,
            },
        )

        content = response.text
        lines = [ln for ln in content.split("\n") if ln.startswith("data:")]

        # Find chunks with content
        content_chunks = []
        for line in lines:
            data = line.removeprefix("data:").strip()
            if data == "[DONE]":
                continue

            parsed = json.loads(data)
            if parsed["choices"] and parsed["choices"][0]["delta"].get("content"):
                content_chunks.append(parsed["choices"][0]["delta"]["content"])

        assert len(content_chunks) > 0, "Should have at least one content chunk"

        # Reassemble content
        full_content = "".join(content_chunks)
        assert "Test message" in full_content or "Streaming" in full_content

    def test_streaming_final_chunk_has_stop_reason(self, client):
        """Last chunk before [DONE] has finish_reason=stop."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4.1-mini",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            },
        )

        content = response.text
        lines = [ln for ln in content.split("\n") if ln.startswith("data:")]

        # Find the chunk before [DONE]
        final_chunk = None
        for line in reversed(lines):
            data = line.removeprefix("data:").strip()
            if data == "[DONE]":
                continue
            final_chunk = json.loads(data)
            break

        assert final_chunk is not None
        assert final_chunk["choices"][0]["finish_reason"] == "stop"

    def test_non_streaming_still_works(self, client):
        """Non-streaming request returns normal JSON response."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4.1-mini",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": False,
            },
        )

        assert response.status_code == 200
        assert "application/json" in response.headers["content-type"]

        data = response.json()
        assert data["object"] == "chat.completion"
        assert "choices" in data
        assert data["choices"][0]["message"]["content"] is not None


class TestStreamingWithTaskType:
    """Test streaming with different task types."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from aperion_switchboard.service.app import create_app

        app = create_app()
        with TestClient(app) as client:
            yield client

    def test_streaming_respects_task_type_header(self, client):
        """Streaming uses task type from header for routing."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4.1-mini",
                "messages": [{"role": "user", "content": "Analyze this"}],
                "stream": True,
            },
            headers={"X-Aperion-Task-Type": "testing"},
        )

        assert response.status_code == 200
        # Echo provider is used for testing task type
        assert response.headers["X-Switchboard-Provider"] == "echo"
