# Unified LLM Infrastructure

This package provides a unified, extensible interface for working with different LLM providers. It includes comprehensive configuration management, monitoring, error handling, and debugging capabilities.

## Key Features

- **Unified Interface**: All providers implement the same interface for consistent usage
- **Dynamic Registration**: Providers can be registered at runtime or via decorators
- **Comprehensive Configuration**: Flexible configuration loading from files and environment variables
- **Monitoring & Debugging**: Built-in request logging, metrics collection, and performance tracking
- **Error Handling**: Standardized error hierarchy with detailed context
- **Fallback Support**: Automatic provider fallback and load balancing (via LLMManager)

## Supported Providers

The infrastructure supports the following LLM providers:

### OpenAI-Compatible Providers
- **OpenAI**: Direct access to OpenAI's GPT models
- **OpenRouter**: Access to multiple models through OpenRouter's unified API
- **DeepSeek**: Access to DeepSeek's models through their OpenAI-compatible API

### Other Providers
- **Anthropic**: Claude models with advanced reasoning capabilities
- **Google Gemini**: Google's multimodal AI models

All OpenAI-compatible providers share the same base implementation and support:
- Chat completions
- Streaming responses
- Tool/function calling
- System messages
- Configurable parameters (temperature, max_tokens, etc.)

## Quick Start

### 1. Basic Provider Usage

```python
from spoon_ai.llm import get_global_registry
from spoon_ai.schema import Message

# Get a provider (assumes it's already registered and configured)
registry = get_global_registry()
provider = registry.get_provider("openai", {
    "api_key": "your-api-key",
    "model": "gpt-4"
})

# Initialize and use
await provider.initialize(config)
response = await provider.chat([
    Message(role="user", content="Hello, world!")
])
print(response.content)
```

### 2. Using OpenRouter

```python
# OpenRouter provides access to multiple models through one API
provider = registry.get_provider("openrouter", {
    "api_key": "your-openrouter-api-key",
    "model": "openai/gpt-4",  # or "anthropic/claude-3", etc.
    "http_referer": "https://your-app.com",
    "x_title": "Your App Name"
})
```

### 3. Using DeepSeek

```python
# DeepSeek's own models
provider = registry.get_provider("deepseek", {
    "api_key": "your-deepseek-api-key",
    "model": "deepseek-chat"
})
```

### 4. Creating a Custom OpenAI-Compatible Provider

```python
from spoon_ai.llm.providers.openai_compatible_provider import OpenAICompatibleProvider
from spoon_ai.llm import register_provider, ProviderCapability

@register_provider("custom_openai", [ProviderCapability.CHAT, ProviderCapability.COMPLETION])
class CustomOpenAIProvider(OpenAICompatibleProvider):
    def __init__(self):
        super().__init__()
        self.provider_name = "custom_openai"
        self.default_base_url = "https://api.custom-provider.com/v1"
        self.default_model = "custom-model"
    
    def get_additional_headers(self, config):
        # Add any custom headers your provider needs
        return {
            "Custom-Header": config.get("custom_value", "default")
        }
    
    def get_metadata(self):
        return ProviderMetadata(
            name="custom_openai",
            version="1.0.0",
            capabilities=[ProviderCapability.CHAT, ProviderCapability.COMPLETION],
            max_tokens=8192,
            supports_system_messages=True
        )
```

### 5. Creating a Fully Custom Provider

```python
from spoon_ai.llm import LLMProviderInterface, register_provider, ProviderCapability

@register_provider("custom", [ProviderCapability.CHAT, ProviderCapability.COMPLETION])
class CustomProvider(LLMProviderInterface):
    async def initialize(self, config):
        self.config = config
        # Initialize your provider here
    
    async def chat(self, messages, **kwargs):
        # Implement chat functionality
        return LLMResponse(
            content="Custom response",
            provider="custom",
            model=self.config.get("model", "custom-model"),
            finish_reason="stop",
            native_finish_reason="stop"
        )
    
    # Implement other required methods...
```

### 6. Configuration Management

```python
from spoon_ai.llm import ConfigurationManager

config_manager = ConfigurationManager()

# Load provider configuration
provider_config = config_manager.load_provider_config("openai")
print(f"Model: {provider_config.model}")
print(f"Max tokens: {provider_config.max_tokens}")

# Get default provider
default = config_manager.get_default_provider()
```

### 7. Monitoring and Debugging

```python
from spoon_ai.llm import get_debug_logger, get_metrics_collector

debug_logger = get_debug_logger()
metrics_collector = get_metrics_collector()

# Log a request
request_id = debug_logger.log_request("openai", "chat", {"model": "gpt-4.1"})

# After the request completes
debug_logger.log_response(request_id, response, duration)

# Record metrics
metrics_collector.record_request("openai", "chat", duration, True, tokens=150)

# Get statistics
stats = metrics_collector.get_provider_stats("openai")
print(f"Success rate: {stats.successful_requests / stats.total_requests * 100:.1f}%")
```

## Architecture

### Core Components

1. **LLMProviderInterface**: Abstract base class that all providers must implement
2. **LLMProviderRegistry**: Manages provider registration and instance creation
3. **ConfigurationManager**: Handles configuration loading and validation
4. **DebugLogger**: Provides comprehensive request/response logging
5. **MetricsCollector**: Collects and aggregates performance metrics

### Provider Interface

All providers must implement these methods:

- `initialize(config)`: Initialize with configuration
- `chat(messages, **kwargs)`: Send chat request
- `chat_stream(messages, **kwargs)`: Send streaming chat request
- `completion(prompt, **kwargs)`: Send completion request
- `chat_with_tools(messages, tools, **kwargs)`: Send tool-enabled chat request
- `get_metadata()`: Return provider metadata and capabilities
- `health_check()`: Check if provider is healthy
- `cleanup()`: Cleanup resources

### Configuration

Configuration can be loaded from:

1. **Configuration files** (JSON or TOML):
   ```json
   {
     "providers": {
       "openai": {
         "api_key": "your-key",
         "model": "gpt-4",
         "max_tokens": 4096
       },
       "openrouter": {
         "api_key": "your-openrouter-key",
         "model": "openai/gpt-3.5-turbo",
         "http_referer": "https://your-app.com",
         "x_title": "Your App Name"
       },
       "deepseek": {
         "api_key": "your-deepseek-key",
         "model": "deepseek-chat",
         "max_tokens": 4096
       }
     }
   }
   ```

2. **Environment variables**:
   ```bash
   # OpenAI
   OPENAI_API_KEY=your-key
   OPENAI_MODEL=gpt-4
   OPENAI_MAX_TOKENS=4096
   
   # OpenRouter
   OPENROUTER_API_KEY=your-openrouter-key
   OPENROUTER_MODEL=openai/gpt-3.5-turbo
   OPENROUTER_HTTP_REFERER=https://your-app.com
   OPENROUTER_X_TITLE=Your App Name
   
   # DeepSeek
   DEEPSEEK_API_KEY=your-deepseek-key
   DEEPSEEK_MODEL=deepseek-chat
   DEEPSEEK_MAX_TOKENS=4096
   ```

3. **Direct configuration**:
   ```python
   # OpenAI
   openai_config = {
       "api_key": "your-key",
       "model": "gpt-4",
       "max_tokens": 4096
   }
   
   # OpenRouter
   openrouter_config = {
       "api_key": "your-openrouter-key",
       "model": "openai/gpt-3.5-turbo",
       "http_referer": "https://your-app.com",
       "x_title": "Your App Name"
   }
   
   # DeepSeek
   deepseek_config = {
       "api_key": "your-deepseek-key",
       "model": "deepseek-chat",
       "max_tokens": 4096
   }
   ```

### Error Handling

The infrastructure provides a comprehensive error hierarchy:

- `LLMError`: Base error class
- `ProviderError`: Provider-specific errors
- `ConfigurationError`: Configuration validation errors
- `RateLimitError`: Rate limit exceeded
- `AuthenticationError`: Authentication failed
- `ModelNotFoundError`: Model not available
- `TokenLimitError`: Token limit exceeded
- `NetworkError`: Network connectivity issues

### Monitoring

The monitoring system provides:

- **Request Logging**: Detailed logs of all requests and responses
- **Metrics Collection**: Performance statistics and usage tracking
- **Error Tracking**: Comprehensive error logging with context
- **Health Monitoring**: Provider availability checking

## Migration from Legacy Code

The new infrastructure is designed to be backward compatible. Existing code using `LLMBase` and `LLMFactory` will continue to work, but new code should use the unified interface.

### Legacy Usage
```python
from spoon_ai.llm import LLMFactory
llm = LLMFactory.create("openai")
```

### New Usage
```python
from spoon_ai.llm import get_global_registry
registry = get_global_registry()
provider = registry.get_provider("openai", config)
```

## Migration Guide

### Step 1: Update Imports

**Before:**
```python
from spoon_ai.llm.factory import LLMFactory
from spoon_ai.llm.base import LLMBase
```

**After:**
```python
from spoon_ai.llm import LLMManager, get_global_registry
from spoon_ai.llm.interface import LLMProviderInterface
```

### Step 2: Replace Factory Usage

**Before:**
```python
llm = LLMFactory.create("openai", api_key="your-key")
response = llm.chat("Hello")
```

**After:**
```python
config_manager = ConfigurationManager()
llm_manager = LLMManager(config_manager)
response = await llm_manager.chat([
    {"role": "user", "content": "Hello"}
], provider="openai")
```

### Step 3: Update Custom Providers

**Before:**
```python
class CustomLLM(LLMBase):
    def __init__(self, api_key):
        self.api_key = api_key
    
    def chat(self, message):
        # Implementation
        return response
```

**After:**
```python
@register_provider("custom")
class CustomProvider(LLMProviderInterface):
    async def initialize(self, config):
        self.api_key = config["api_key"]
    
    async def chat(self, messages, **kwargs):
        # Implementation
        return LLMResponse(...)
```

### Step 4: Update Configuration

**Before:**
```python
# Configuration scattered across code
openai_llm = LLMFactory.create("openai", api_key="key1")
anthropic_llm = LLMFactory.create("anthropic", api_key="key2")
```

**After:**
```json
// config.json
{
  "llm_providers": {
    "openai": {"api_key": "key1"},
    "anthropic": {"api_key": "key2"}
  }
}
```

### Step 5: Enable Monitoring (Optional)

```python
from spoon_ai.llm import get_debug_logger

# Add monitoring to existing code
debug_logger = get_debug_logger()
debug_logger.enable_request_logging()
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Provider Not Found Error

**Error:** `ProviderError: Provider 'openai' not found`

**Causes:**
- Provider not registered
- Typo in provider name
- Provider module not imported

**Solutions:**
```python
# Check available providers
from spoon_ai.llm import get_global_registry
registry = get_global_registry()
print(registry.list_providers())

# Ensure provider is imported
from spoon_ai.llm.providers import openai_provider  # This registers the provider
```

#### 2. Configuration Validation Errors

**Error:** `ConfigurationError: Missing required field 'api_key'`

**Causes:**
- Missing API key in configuration
- Incorrect configuration format
- Environment variable not set

**Solutions:**
```python
# Check configuration
from spoon_ai.llm import ConfigurationManager
config_manager = ConfigurationManager()
try:
    config = config_manager.load_provider_config("openai")
    print("Configuration valid")
except ConfigurationError as e:
    print(f"Configuration error: {e}")

# Set environment variable
import os
os.environ["OPENAI_API_KEY"] = "your-api-key"
```

#### 3. Authentication Errors

**Error:** `AuthenticationError: Invalid API key for provider 'openai'`

**Causes:**
- Incorrect API key
- Expired API key
- API key format issues

**Solutions:**
```python
# Test API key manually
import openai
openai.api_key = "your-key"
try:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "test"}],
        max_tokens=5
    )
    print("API key valid")
except Exception as e:
    print(f"API key invalid: {e}")
```

#### 4. Rate Limit Errors

**Error:** `RateLimitError: Rate limit exceeded for provider 'openai'`

**Causes:**
- Too many requests per minute
- Quota exceeded
- Concurrent request limits

**Solutions:**
```python
# Enable automatic retry with backoff
config = {
    "api_key": "your-key",
    "retry_attempts": 3,
    "retry_delay": 1.0,
    "max_concurrent_requests": 5
}

# Use fallback providers
llm_manager.set_fallback_chain(["openai", "anthropic", "gemini"])
```

#### 5. Network Connectivity Issues

**Error:** `NetworkError: Connection timeout to provider 'openai'`

**Causes:**
- Network connectivity problems
- Firewall blocking requests
- Provider service outage

**Solutions:**
```python
# Test connectivity
import requests
try:
    response = requests.get("https://api.openai.com/v1/models", timeout=10)
    print(f"Connectivity OK: {response.status_code}")
except requests.RequestException as e:
    print(f"Connectivity issue: {e}")

# Configure timeout settings
config = {
    "api_key": "your-key",
    "timeout": 30,
    "max_retries": 3
}
```

#### 6. Model Not Available Errors

**Error:** `ModelNotFoundError: Model 'gpt-5' not available for provider 'openai'`

**Causes:**
- Model name typo
- Model not available in region
- Model deprecated or removed

**Solutions:**
```python
# Check available models
provider = registry.get_provider("openai", config)
metadata = provider.get_metadata()
print("Available models:", metadata.available_models)

# Use model validation
from spoon_ai.llm.config import validate_model
if validate_model("openai", "gpt-4.1"):
    print("Model available")
else:
    print("Model not available")
```

#### 7. Memory and Performance Issues

**Symptoms:**
- Slow response times
- High memory usage
- Request timeouts

**Solutions:**
```python
# Enable caching
config = {
    "api_key": "your-key",
    "enable_caching": True,
    "cache_ttl": 3600  # 1 hour
}

# Monitor performance
from spoon_ai.llm import get_metrics_collector
metrics = get_metrics_collector()
stats = metrics.get_performance_stats()
print(f"Average response time: {stats['avg_response_time']:.2f}s")

# Clean up resources
await provider.cleanup()
```

### Debug Mode

Enable comprehensive debugging:

```python
import logging
from spoon_ai.llm import get_debug_logger

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
debug_logger = get_debug_logger()
debug_logger.set_log_level("DEBUG")

# Enable request/response logging
debug_logger.enable_request_logging()
debug_logger.enable_response_logging()
```

### Health Monitoring

Monitor provider health:

```python
# Check all providers
health_status = await llm_manager.health_check_all()
for provider, is_healthy in health_status.items():
    status = "✅ Healthy" if is_healthy else "❌ Unhealthy"
    print(f"{provider}: {status}")

# Set up health monitoring
from spoon_ai.llm.monitoring import HealthMonitor
monitor = HealthMonitor()
monitor.start_monitoring(interval=60)  # Check every minute
```

### Performance Optimization

Optimize performance:

```python
# Use connection pooling
config = {
    "api_key": "your-key",
    "connection_pool_size": 10,
    "keep_alive": True
}

# Enable request batching
batch_responses = await llm_manager.batch_chat([
    [{"role": "user", "content": "Hello 1"}],
    [{"role": "user", "content": "Hello 2"}],
    [{"role": "user", "content": "Hello 3"}]
])

# Use streaming for long responses
async for chunk in llm_manager.chat_stream([
    {"role": "user", "content": "Write a long story"}
]):
    print(chunk, end="")
```

## Best Practices

1. **Always initialize providers** before use
2. **Use monitoring** to track performance and errors
3. **Handle errors gracefully** with appropriate fallbacks
4. **Configure providers** through configuration files when possible
5. **Clean up resources** when providers are no longer needed
6. **Use health checks** to verify provider availability

## Examples

See `examples/llm_infrastructure_example.py` for a comprehensive demonstration of all features.

## Testing

Run the infrastructure tests:

```bash
python test_llm_infrastructure.py
```

This will verify that all components work correctly together.