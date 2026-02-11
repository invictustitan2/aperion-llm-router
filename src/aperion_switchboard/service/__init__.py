"""FastAPI Service Layer for The Switchboard."""

from .app import create_app, get_app
from .middleware import CostLoggingMiddleware, TelemetryMiddleware
from .schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    CompletionUsage,
)

__all__ = [
    "create_app",
    "get_app",
    "CostLoggingMiddleware",
    "TelemetryMiddleware",
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "ChatMessage",
    "Choice",
    "CompletionUsage",
]
