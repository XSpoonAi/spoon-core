"""
Standardized error hierarchy for LLM operations.
"""

from __future__ import annotations

from typing import Any


class LLMError(Exception):
    """Base exception for all LLM-related errors."""

    def __init__(self, message: str, context: dict[str, Any] | None = None):
        self.message = message
        self.context = context or {}
        super().__init__(message)


class ProviderError(LLMError):
    """Provider-specific error with detailed context."""

    def __init__(
        self,
        provider: str,
        message: str,
        original_error: Exception | None = None,
        context: dict[str, Any] | None = None,
    ):
        self.provider = provider
        self.original_error = original_error
        formatted_message = f"[{provider}] {message}"
        super().__init__(formatted_message, context)


class ConfigurationError(LLMError):
    """Configuration validation or loading error."""

    def __init__(
        self,
        message: str,
        config_key: str | None = None,
        context: dict[str, Any] | None = None,
    ):
        self.config_key = config_key
        if config_key:
            formatted_message = f"Configuration error for '{config_key}': {message}"
        else:
            formatted_message = f"Configuration error: {message}"
        super().__init__(formatted_message, context)


class RateLimitError(ProviderError):
    """Rate limit exceeded error."""

    def __init__(
        self,
        provider: str,
        retry_after: int | None = None,
        context: dict[str, Any] | None = None,
    ):
        self.retry_after = retry_after
        message = "Rate limit exceeded"
        if retry_after:
            message += f", retry after {retry_after} seconds"
        super().__init__(provider, message, context=context)


class AuthenticationError(ProviderError):
    """Authentication failed error."""

    def __init__(self, provider: str, context: dict[str, Any] | None = None):
        super().__init__(provider, "Authentication failed - check API key", context=context)


class ModelNotFoundError(ProviderError):
    """Model not found or not available error."""

    def __init__(self, provider: str, model: str, context: dict[str, Any] | None = None):
        self.model = model
        super().__init__(provider, f"Model '{model}' not found or not available", context=context)


class TokenLimitError(ProviderError):
    """Token limit exceeded error."""

    def __init__(
        self,
        provider: str,
        requested_tokens: int,
        max_tokens: int,
        context: dict[str, Any] | None = None,
    ):
        self.requested_tokens = requested_tokens
        self.max_tokens = max_tokens
        message = f"Token limit exceeded: requested {requested_tokens}, max {max_tokens}"
        super().__init__(provider, message, context=context)


class NetworkError(ProviderError):
    """Network connectivity or timeout error."""

    def __init__(
        self,
        provider: str,
        message: str = "Network error",
        original_error: Exception | None = None,
        context: dict[str, Any] | None = None,
    ):
        super().__init__(provider, message, original_error, context)


class ProviderUnavailableError(ProviderError):
    """Provider service unavailable error."""

    def __init__(self, provider: str, context: dict[str, Any] | None = None):
        super().__init__(provider, "Provider service unavailable", context=context)


class ValidationError(LLMError):
    """Input validation error."""

    def __init__(
        self,
        message: str,
        field: str | None = None,
        context: dict[str, Any] | None = None,
    ):
        self.field = field
        if field:
            formatted_message = f"Validation error for '{field}': {message}"
        else:
            formatted_message = f"Validation error: {message}"
        super().__init__(formatted_message, context)