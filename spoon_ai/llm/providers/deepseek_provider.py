"""
DeepSeek Provider implementation for the unified LLM interface.
DeepSeek provides access to their models through an OpenAI-compatible API.
"""

from typing import Dict, Any
from logging import getLogger

from ..interface import ProviderMetadata, ProviderCapability
from ..registry import register_provider
from .openai_compatible_provider import OpenAICompatibleProvider

logger = getLogger(__name__)


@register_provider("deepseek", [
    ProviderCapability.CHAT,
    ProviderCapability.COMPLETION,
    ProviderCapability.TOOLS,
    ProviderCapability.STREAMING
])
class DeepSeekProvider(OpenAICompatibleProvider):
    """DeepSeek provider implementation using OpenAI-compatible API."""
    
    def __init__(self):
        super().__init__()
        self.provider_name = "deepseek"
        self.default_base_url = "https://api.deepseek.com/v1"
        self.default_model = "deepseek-reasoner"
    
    def get_metadata(self) -> ProviderMetadata:
        """Get DeepSeek provider metadata."""
        return ProviderMetadata(
            name="deepseek",
            version="1.0.0",
            capabilities=[
                ProviderCapability.CHAT,
                ProviderCapability.COMPLETION,
                ProviderCapability.TOOLS,
                ProviderCapability.STREAMING
            ],
            max_tokens=64000,  # DeepSeek context limit
            supports_system_messages=True,
            rate_limits={
                "requests_per_minute": 60,   # DeepSeek rate limits
                "tokens_per_minute": 1000000  # DeepSeek token limits
            }
        )