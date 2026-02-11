"""
Tests for request validation.

Validates:
- Message content length limits
- Message count limits
- Request body size limits
"""

import pytest
from pydantic import ValidationError

from aperion_switchboard.service.schemas import (
    ChatCompletionRequest,
    ChatMessage,
    MAX_MESSAGE_CONTENT_LENGTH,
    MAX_MESSAGES_COUNT,
)


class TestMessageContentValidation:
    """Tests for message content length validation."""

    def test_message_content_within_limit(self):
        """Content within limits should be accepted."""
        msg = ChatMessage(role="user", content="Hello, world!")
        assert msg.content == "Hello, world!"

    def test_message_content_at_limit(self):
        """Content exactly at limit should be accepted."""
        content = "x" * MAX_MESSAGE_CONTENT_LENGTH
        msg = ChatMessage(role="user", content=content)
        assert len(msg.content) == MAX_MESSAGE_CONTENT_LENGTH

    def test_message_content_exceeds_limit(self):
        """Content exceeding limit should be rejected."""
        content = "x" * (MAX_MESSAGE_CONTENT_LENGTH + 1)
        with pytest.raises(ValidationError) as exc_info:
            ChatMessage(role="user", content=content)
        
        assert "exceeds maximum length" in str(exc_info.value)

    def test_message_content_none_allowed(self):
        """None content should be allowed (function messages)."""
        msg = ChatMessage(role="function", content=None)
        assert msg.content is None

    def test_message_content_empty_allowed(self):
        """Empty string content should be allowed."""
        msg = ChatMessage(role="user", content="")
        assert msg.content == ""


class TestMessageCountValidation:
    """Tests for message count validation."""

    def test_single_message_allowed(self):
        """Single message should be allowed."""
        request = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="Hello")]
        )
        assert len(request.messages) == 1

    def test_max_messages_allowed(self):
        """Exactly max messages should be allowed."""
        messages = [
            ChatMessage(role="user", content=f"Message {i}")
            for i in range(MAX_MESSAGES_COUNT)
        ]
        request = ChatCompletionRequest(messages=messages)
        assert len(request.messages) == MAX_MESSAGES_COUNT

    def test_too_many_messages_rejected(self):
        """Exceeding max messages should be rejected."""
        messages = [
            ChatMessage(role="user", content=f"Message {i}")
            for i in range(MAX_MESSAGES_COUNT + 1)
        ]
        with pytest.raises(ValidationError) as exc_info:
            ChatCompletionRequest(messages=messages)
        
        assert "Too many messages" in str(exc_info.value)

    def test_empty_messages_rejected(self):
        """Empty messages list should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ChatCompletionRequest(messages=[])
        
        assert "At least one message" in str(exc_info.value)


class TestUserIdValidation:
    """Tests for user ID validation."""

    def test_user_id_within_limit(self):
        """User ID within limits should be accepted."""
        request = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="Hello")],
            user="user_123"
        )
        assert request.user == "user_123"

    def test_user_id_too_long_rejected(self):
        """User ID exceeding limit should be rejected."""
        from aperion_switchboard.service.schemas import MAX_USER_ID_LENGTH
        
        long_user_id = "x" * (MAX_USER_ID_LENGTH + 1)
        with pytest.raises(ValidationError) as exc_info:
            ChatCompletionRequest(
                messages=[ChatMessage(role="user", content="Hello")],
                user=long_user_id
            )
        
        # Pydantic max_length constraint
        assert "String should have at most" in str(exc_info.value)


class TestRequestDefaults:
    """Tests for request default values."""

    def test_default_model(self):
        """Default model should be gpt-4.1-mini."""
        request = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="Hello")]
        )
        assert request.model == "gpt-4.1-mini"

    def test_default_stream_false(self):
        """Default stream should be False."""
        request = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="Hello")]
        )
        assert request.stream is False

    def test_default_n_is_one(self):
        """Default n should be 1."""
        request = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="Hello")]
        )
        assert request.n == 1
