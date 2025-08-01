"""
OpenRouter Provider implementation for the unified LLM interface.
OpenRouter provides access to multiple LLM models through a unified API compatible with OpenAI.
"""

from typing import Dict, Any
from logging import getLogger

from ..interface import ProviderMetadata, ProviderCapability
from ..registry import register_provider
from .openai_compatible_provider import OpenAICompatibleProvider

logger = getLogger(__name__)


@register_provider("openrouter", [
    ProviderCapability.CHAT,
    ProviderCapability.COMPLETION,
    ProviderCapability.TOOLS,
    ProviderCapability.STREAMING
])
class OpenRouterProvider(OpenAICompatibleProvider):
    """OpenRouter provider implementation using OpenAI-compatible API."""
    
    def __init__(self):
        super().__init__()
        self.provider_name = "openrouter"
        self.default_base_url = "https://openrouter.ai/api/v1"
        self.default_model = "openai/gpt-3.5-turbo"
    
    def get_additional_headers(self, config: Dict[str, Any]) -> Dict[str, str]:
        """Get OpenRouter-specific headers."""
        headers = {}
        
        # Check for custom_headers first (new format)
        if 'custom_headers' in config and config['custom_headers']:
            headers.update(config['custom_headers'])
        
        # Add default headers if not already specified
        if "HTTP-Referer" not in headers:
            headers["HTTP-Referer"] = config.get('http_referer', 'https://spoon-ai.com')
        
        if "X-Title" not in headers:
            headers["X-Title"] = config.get('x_title', 'SpoonAI Framework')
        
        return headers

    
    def get_metadata(self) -> ProviderMetadata:
        """Get OpenRouter provider metadata."""
        return ProviderMetadata(
            name="openrouter",
            version="1.0.0",
            capabilities=[
                ProviderCapability.CHAT,
                ProviderCapability.COMPLETION,
                ProviderCapability.TOOLS,
                ProviderCapability.STREAMING
            ],
            max_tokens=200000,  # OpenRouter supports various models with different limits
            supports_system_messages=True,
            rate_limits={
                "requests_per_minute": 200,  # Varies by model and plan
                "tokens_per_minute": 40000   # Varies by model and plan
            }
        )