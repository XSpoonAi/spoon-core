"""
Unified LLM infrastructure package.

This package provides a unified interface for working with different LLM providers,
including comprehensive configuration management, monitoring, and error handling.
"""

from .interface import (
    LLMProviderInterface,
    ProviderCapability,
    ProviderMetadata,
    LLMResponse
)

from .registry import (
    LLMProviderRegistry,
    register_provider,
    get_global_registry
)

from .config import (
    ConfigurationManager,
    ProviderConfig
)

from .monitoring import (
    DebugLogger,
    MetricsCollector,
    RequestMetrics,
    ProviderStats,
    get_debug_logger,
    get_metrics_collector
)

from .errors import (
    LLMError,
    ProviderError,
    ConfigurationError,
    RateLimitError,
    InsufficientCreditsError,
    AuthenticationError,
    ModelNotFoundError,
    TokenLimitError,
    NetworkError,
    ProviderUnavailableError,
    ValidationError
)

from .manager import (
    LLMManager,
    FallbackStrategy,
    LoadBalancer,
    get_llm_manager,
    set_llm_manager
)

from .response_normalizer import (
    ResponseNormalizer,
    get_response_normalizer
)

from .providers import (
    OpenAIProvider,
    AnthropicProvider,
    GeminiProvider
)

# Legacy imports for backward compatibility
from .base import LLMBase, LLMConfig
from .factory import LLMFactory

__all__ = [
    # New unified interface
    'LLMProviderInterface',
    'ProviderCapability', 
    'ProviderMetadata',
    'LLMResponse',
    
    # Registry
    'LLMProviderRegistry',
    'register_provider',
    'get_global_registry',
    
    # Configuration
    'ConfigurationManager',
    'ProviderConfig',
    
    # Monitoring
    'DebugLogger',
    'MetricsCollector', 
    'RequestMetrics',
    'ProviderStats',
    'get_debug_logger',
    'get_metrics_collector',
    
    # Errors
    'LLMError',
    'ProviderError',
    'ConfigurationError',
    'RateLimitError',
    'InsufficientCreditsError',
    'AuthenticationError',
    'ModelNotFoundError',
    'TokenLimitError',
    'NetworkError',
    'ProviderUnavailableError',
    'ValidationError',
    
    # Manager and orchestration
    'LLMManager',
    'FallbackStrategy',
    'LoadBalancer',
    'get_llm_manager',
    'set_llm_manager',
    
    # Response normalization
    'ResponseNormalizer',
    'get_response_normalizer',
    
    # Provider implementations
    'OpenAIProvider',
    'AnthropicProvider',
    'GeminiProvider',
    
    # Legacy (for backward compatibility)
    'LLMBase',
    'LLMConfig',
    'LLMFactory'
]