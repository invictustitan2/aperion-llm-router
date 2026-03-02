"""
OpenAI-Compatible Pydantic Schemas.

These schemas provide drop-in compatibility with the OpenAI Chat Completions API,
allowing existing tools to use The Switchboard by simply changing their base_url.

Reference: https://platform.openai.com/docs/api-reference/chat/create
"""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# Size limits - tuneable via constants
MAX_MESSAGE_CONTENT_LENGTH = 128_000  # ~32K tokens at 4 chars/token
MAX_MESSAGES_COUNT = 100  # Maximum messages in a conversation
MAX_USER_ID_LENGTH = 256


class ChatMessage(BaseModel):
    """A single message in the chat conversation."""

    role: Literal["system", "user", "assistant", "function"] = Field(
        description="The role of the message author"
    )
    content: str | None = Field(
        default=None, description="The message content"
    )
    name: str | None = Field(
        default=None, description="Name of the author (optional)"
    )

    @field_validator("content")
    @classmethod
    def validate_content_length(cls, v: str | None) -> str | None:
        """Ensure message content is within size limits."""
        if v is not None and len(v) > MAX_MESSAGE_CONTENT_LENGTH:
            raise ValueError(
                f"Message content exceeds maximum length of {MAX_MESSAGE_CONTENT_LENGTH:,} characters"
            )
        return v


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model: str = Field(
        default="gpt-4.1-mini",
        description="Model identifier (may be overridden by routing)",
    )
    messages: list[ChatMessage] = Field(
        description="List of messages in the conversation"
    )
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )
    top_p: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter",
    )
    n: int | None = Field(
        default=1,
        ge=1,
        le=10,
        description="Number of completions to generate",
    )
    stream: bool = Field(
        default=False,
        description="Whether to stream partial responses",
    )
    stop: str | list[str] | None = Field(
        default=None,
        description="Stop sequences",
    )
    max_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Maximum tokens to generate",
    )
    presence_penalty: float | None = Field(
        default=None,
        ge=-2.0,
        le=2.0,
        description="Presence penalty",
    )
    frequency_penalty: float | None = Field(
        default=None,
        ge=-2.0,
        le=2.0,
        description="Frequency penalty",
    )
    user: str | None = Field(
        default=None,
        max_length=MAX_USER_ID_LENGTH,
        description="User identifier for abuse tracking",
    )

    @field_validator("messages")
    @classmethod
    def validate_messages_count(cls, v: list[ChatMessage]) -> list[ChatMessage]:
        """Ensure message count is within limits."""
        if len(v) > MAX_MESSAGES_COUNT:
            raise ValueError(
                f"Too many messages: {len(v)} exceeds maximum of {MAX_MESSAGES_COUNT}"
            )
        if len(v) == 0:
            raise ValueError("At least one message is required")
        return v

    def to_provider_kwargs(self) -> dict[str, Any]:
        """Convert to provider-agnostic kwargs."""
        kwargs: dict[str, Any] = {}

        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if self.stop is not None:
            kwargs["stop"] = self.stop
        if self.n is not None:
            kwargs["n"] = self.n

        return kwargs

    def get_prompt(self) -> str:
        """Extract user prompt from messages."""
        # Get the last user message
        for msg in reversed(self.messages):
            if msg.role == "user" and msg.content:
                return msg.content
        return ""

    def get_system_prompt(self) -> str | None:
        """Extract system prompt from messages."""
        for msg in self.messages:
            if msg.role == "system" and msg.content:
                return msg.content
        return None


class CompletionUsage(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int = Field(default=0, description="Tokens in the prompt")
    completion_tokens: int = Field(default=0, description="Tokens in the completion")
    total_tokens: int = Field(default=0, description="Total tokens used")

    # Anthropic prompt cache tokens (Switchboard extension)
    cache_creation_input_tokens: int | None = Field(
        default=None,
        description="Tokens written to Anthropic prompt cache (1.25x input price)",
    )
    cache_read_input_tokens: int | None = Field(
        default=None,
        description="Tokens read from Anthropic prompt cache (0.1x input price)",
    )


class ChoiceMessage(BaseModel):
    """Message in a completion choice."""

    role: Literal["assistant"] = "assistant"
    content: str | None = None


class Choice(BaseModel):
    """A single completion choice."""

    index: int = 0
    message: ChoiceMessage
    finish_reason: Literal["stop", "length", "content_filter"] | None = "stop"


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""

    id: str = Field(description="Unique completion ID")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(description="Unix timestamp of creation")
    model: str = Field(description="Model used for completion")
    choices: list[Choice] = Field(description="Completion choices")
    usage: CompletionUsage = Field(description="Token usage statistics")

    # Switchboard extensions (not in OpenAI spec)
    switchboard_provider: str | None = Field(
        default=None,
        description="Actual provider used (Switchboard extension)",
    )
    switchboard_routing_reason: str | None = Field(
        default=None,
        description="Reason for provider selection (Switchboard extension)",
    )


class StreamChoice(BaseModel):
    """A single streaming choice."""

    index: int = 0
    delta: dict[str, Any] = Field(default_factory=dict)
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    """OpenAI-compatible streaming chunk."""

    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[StreamChoice]


class ErrorResponse(BaseModel):
    """Error response format."""

    error: dict[str, Any] = Field(
        description="Error details",
        examples=[{
            "message": "Provider not available",
            "type": "service_unavailable",
            "code": "no_provider",
        }],
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: Literal["healthy", "degraded", "unhealthy"]
    version: str
    providers: dict[str, dict[str, Any]]
    fail_closed_compliant: bool
