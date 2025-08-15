"""
LLM Manager - Central orchestrator for managing providers, fallback, and load balancing.
"""

import asyncio
import random
from typing import List, Dict, Any, Optional, AsyncGenerator, Set
from logging import getLogger

from contextlib import asynccontextmanager
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from spoon_ai.schema import Message
from .interface import LLMProviderInterface, LLMResponse, ProviderCapability
from .registry import LLMProviderRegistry, get_global_registry
from .config import ConfigurationManager
from .monitoring import DebugLogger, MetricsCollector, get_debug_logger, get_metrics_collector
from .response_normalizer import ResponseNormalizer, get_response_normalizer
from .errors import ProviderError, ConfigurationError, ProviderUnavailableError

logger = getLogger(__name__)


@dataclass
class ProviderState:
    """Track provider initialization and health state."""
    is_initializing: bool = False
    is_initialized: bool = False
    initialization_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_error: Optional[Exception] = None
    last_error_time: Optional[datetime] = None
    initialization_attempts: int = 0
    max_attempts: int = 3
    backoff_until: Optional[datetime] = None

    def can_retry_initialization(self) -> bool:
        """Check if provider initialization can be retried."""
        if self.initialization_attempts >= self.max_attempts:
            return False
        
        if self.backoff_until and datetime.now() < self.backoff_until:
            return False
            
        return True

    def record_initialization_failure(self, error: Exception) -> None:
        """Record initialization failure with exponential backoff."""
        self.initialization_attempts += 1
        self.last_error = error
        self.last_error_time = datetime.now()
        self.is_initialized = False
        self.is_initializing = False
        
        # Exponential backoff: 2^attempts seconds
        backoff_seconds = min(2 ** self.initialization_attempts, 300)  # Max 5 minutes
        self.backoff_until = datetime.now() + timedelta(seconds=backoff_seconds)

    def record_initialization_success(self) -> None:
        """Record successful initialization."""
        self.is_initialized = True
        self.is_initializing = False
        self.initialization_attempts = 0
        self.last_error = None
        self.last_error_time = None
        self.backoff_until = None

class FallbackStrategy:
    """Handles fallback logic between providers."""

    def __init__(self, debug_logger: DebugLogger):
        self.debug_logger = debug_logger

    async def execute_with_fallback(self, providers: List[str], operation, *args, **kwargs) -> LLMResponse:
        """Execute operation with fallback chain.

        Args:
            providers: List of provider names in fallback order
            operation: Async operation to execute
            *args, **kwargs: Arguments for the operation

        Returns:
            LLMResponse: Response from successful provider

        Raises:
            ProviderError: If all providers fail
        """
        last_error = None

        for i, provider_name in enumerate(providers):
            try:
                logger.info(f"Attempting operation with provider: {provider_name}")
                result = await operation(provider_name, *args, **kwargs)

                if i > 0:  # Log successful fallback
                    logger.info(f"Successfully fell back to {provider_name} after {i} failures")

                return result

            except Exception as e:
                last_error = e
                logger.warning(f"Provider {provider_name} failed: {str(e)}")

                # Log fallback event if not the last provider
                if i < len(providers) - 1:
                    next_provider = providers[i + 1]
                    self.debug_logger.log_fallback(provider_name, next_provider, str(e))

                continue

        # All providers failed
        raise ProviderError(
            "fallback",
            f"All providers failed. Last error: {str(last_error)}",
            original_error=last_error,
            context={"attempted_providers": providers}
        )


class LoadBalancer:
    """Handles load balancing between multiple provider instances."""

    def __init__(self):
        self.provider_weights: Dict[str, float] = {}
        self.provider_health: Dict[str, bool] = {}

    def select_provider(self, providers: List[str], strategy: str = "round_robin") -> str:
        """Select a provider based on load balancing strategy.

        Args:
            providers: List of available providers
            strategy: Load balancing strategy ('round_robin', 'weighted', 'random')

        Returns:
            str: Selected provider name
        """
        # Filter out unhealthy providers
        healthy_providers = [p for p in providers if self.provider_health.get(p, True)]

        if not healthy_providers:
            # If no healthy providers, use all providers as fallback
            healthy_providers = providers

        if strategy == "random":
            return random.choice(healthy_providers)
        elif strategy == "weighted":
            return self._weighted_selection(healthy_providers)
        else:  # round_robin (default)
            return self._round_robin_selection(healthy_providers)

    def _round_robin_selection(self, providers: List[str]) -> str:
        """Simple round-robin selection."""
        if not hasattr(self, '_round_robin_index'):
            self._round_robin_index = 0

        provider = providers[self._round_robin_index % len(providers)]
        self._round_robin_index += 1
        return provider

    def _weighted_selection(self, providers: List[str]) -> str:
        """Weighted selection based on provider weights."""
        if not self.provider_weights:
            return random.choice(providers)

        # Calculate total weight
        total_weight = sum(self.provider_weights.get(p, 1.0) for p in providers)

        # Random selection based on weights
        r = random.uniform(0, total_weight)
        current_weight = 0

        for provider in providers:
            current_weight += self.provider_weights.get(provider, 1.0)
            if r <= current_weight:
                return provider

        return providers[-1]  # Fallback

    def update_provider_health(self, provider: str, is_healthy: bool) -> None:
        """Update provider health status."""
        self.provider_health[provider] = is_healthy
        logger.debug(f"Updated {provider} health status: {is_healthy}")

    def set_provider_weight(self, provider: str, weight: float) -> None:
        """Set provider weight for weighted load balancing."""
        self.provider_weights[provider] = weight
        logger.debug(f"Set {provider} weight: {weight}")


class LLMManager:
    """Central orchestrator for LLM providers with fallback and load balancing."""

    def __init__(self,
                 config_manager: Optional[ConfigurationManager] = None,
                 debug_logger: Optional[DebugLogger] = None,
                 metrics_collector: Optional[MetricsCollector] = None,
                 response_normalizer: Optional[ResponseNormalizer] = None,
                 registry: Optional[LLMProviderRegistry] = None):
        """Initialize LLM Manager with enhanced provider state tracking."""
        self.config_manager = config_manager or ConfigurationManager()
        self.debug_logger = debug_logger or get_debug_logger()
        self.metrics_collector = metrics_collector or get_metrics_collector()
        self.response_normalizer = response_normalizer or get_response_normalizer()
        self.registry = registry or get_global_registry()

        self.fallback_strategy = FallbackStrategy(self.debug_logger)
        self.load_balancer = LoadBalancer()

        # Enhanced provider state management
        self.provider_states: Dict[str, ProviderState] = {}
        self.provider_cleanup_tasks: Set[asyncio.Task] = set()
        self._manager_lock = asyncio.Lock()
        self._shutdown_event = asyncio.Event()
        
        # Existing configuration
        self.fallback_chain: List[str] = []
        self.default_provider: Optional[str] = None
        self.load_balancing_enabled: bool = False
        self.load_balancing_strategy: str = "round_robin"

        # Initialize providers from configuration
        self._initialize_providers()

        # Register cleanup on shutdown
        self._register_cleanup()

    def _register_cleanup(self) -> None:
        """Register cleanup callback for graceful shutdown."""
        import atexit
        import signal
        
        def cleanup_sync():
            """Synchronous cleanup wrapper."""
            try:
                # Try to get the current event loop
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is running, schedule cleanup
                    asyncio.create_task(self.cleanup())
                else:
                    # If no loop, run cleanup synchronously
                    loop.run_until_complete(self.cleanup())
            except RuntimeError:
                # No event loop available, skip async cleanup
                logger.warning("No event loop available for async cleanup")
        
        atexit.register(cleanup_sync)

    def _get_provider_state(self, provider_name: str) -> ProviderState:
        """Get or create provider state."""
        if provider_name not in self.provider_states:
            self.provider_states[provider_name] = ProviderState()
        return self.provider_states[provider_name]

    async def _ensure_provider_initialized(self, provider_name: str) -> bool:
        """Ensure provider is properly initialized with thread safety.
        
        Returns:
            bool: True if provider is initialized, False if initialization failed
        """
        state = self._get_provider_state(provider_name)
        
        # Fast path: already initialized
        if state.is_initialized:
            return True
        
        # Check if initialization is possible
        if not state.can_retry_initialization():
            logger.error(f"Provider {provider_name} initialization blocked: "
                        f"attempts={state.initialization_attempts}, "
                        f"backoff_until={state.backoff_until}")
            return False

        # Acquire initialization lock
        async with state.initialization_lock:
            # Double-check after acquiring lock (another thread might have initialized)
            if state.is_initialized:
                return True
            
            # Check if already initializing in another coroutine
            if state.is_initializing:
                logger.info(f"Provider {provider_name} is already being initialized, waiting...")
                # Wait for initialization to complete (with timeout)
                try:
                    await asyncio.wait_for(
                        self._wait_for_initialization(state), 
                        timeout=30.0
                    )
                    return state.is_initialized
                except asyncio.TimeoutError:
                    logger.error(f"Provider {provider_name} initialization timeout")
                    return False

            # Mark as initializing
            state.is_initializing = True
            
            try:
                # Get provider configuration
                config = self.config_manager.load_provider_config(provider_name)
                provider_instance = self.registry.get_provider(provider_name, config.model_dump())

                logger.info(f"Initializing provider: {provider_name}")
                
                # Initialize the provider
                await provider_instance.initialize(config.model_dump())
                
                # Mark as successfully initialized
                state.record_initialization_success()
                
                logger.info(f"Successfully initialized provider: {provider_name}")
                return True

            except Exception as e:
                # Record the failure
                state.record_initialization_failure(e)
                
                logger.error(f"Failed to initialize provider {provider_name} "
                           f"(attempt {state.initialization_attempts}): {e}")
                
                # If this was the last attempt, mark provider as unhealthy
                if not state.can_retry_initialization():
                    self.load_balancer.update_provider_health(provider_name, False)
                
                return False

    async def _wait_for_initialization(self, state: ProviderState) -> None:
        """Wait for provider initialization to complete."""
        while state.is_initializing and not state.is_initialized:
            await asyncio.sleep(0.1)

    async def _execute_provider_operation(self, provider_name: str, method: str, *args, **kwargs) -> LLMResponse:
        """Execute an operation on a specific provider with enhanced initialization management."""
        # Ensure provider is initialized
        if not await self._ensure_provider_initialized(provider_name):
            state = self._get_provider_state(provider_name)
            error_msg = f"Provider {provider_name} is not available"
            if state.last_error:
                error_msg += f": {state.last_error}"
            raise ProviderUnavailableError(provider_name, error_msg)

        # Get provider instance
        try:
            config = self.config_manager.load_provider_config(provider_name)
            provider_instance = self.registry.get_provider(provider_name, config.model_dump())
        except Exception as e:
            logger.error(f"Failed to get provider instance {provider_name}: {e}")
            raise ProviderError(provider_name, f"Failed to get provider instance: {str(e)}", original_error=e)

        # Log request
        request_id = self.debug_logger.log_request(provider_name, method, kwargs)
        start_time = asyncio.get_event_loop().time()

        try:
            # Execute the operation
            operation = getattr(provider_instance, method)
            if not callable(operation):
                raise ProviderError(provider_name, f"Method {method} not available on provider")
                
            response = await operation(*args, **kwargs)

            # Calculate duration and add metadata
            duration = asyncio.get_event_loop().time() - start_time
            response.duration = duration
            response.request_id = request_id

            # Log successful response
            self.debug_logger.log_response(request_id, response, duration)

            # Record metrics
            tokens = response.usage.get('total_tokens', 0) if response.usage else 0
            self.metrics_collector.record_request(
                provider_name, method, duration, True, tokens, response.model
            )

            # Mark provider as healthy
            self.load_balancer.update_provider_health(provider_name, True)

            return response

        except Exception as e:
            # Calculate duration
            duration = asyncio.get_event_loop().time() - start_time

            # Log error
            self.debug_logger.log_error(request_id, e, {"provider": provider_name, "method": method})

            # Record metrics
            self.metrics_collector.record_request(
                provider_name, method, duration, False, error=str(e)
            )

            # Update provider health
            self.load_balancer.update_provider_health(provider_name, False)

            # If this is a critical error, mark provider for reinitialization
            if self._is_critical_error(e):
                logger.warning(f"Critical error detected for {provider_name}, marking for reinitialization")
                state = self._get_provider_state(provider_name)
                state.is_initialized = False

            raise

    def _is_critical_error(self, error: Exception) -> bool:
        """Determine if an error requires provider reinitialization."""
        critical_error_patterns = [
            "connection",
            "authentication",
            "unauthorized", 
            "invalid_api_key",
            "token",
            "timeout"
        ]
        
        error_str = str(error).lower()
        return any(pattern in error_str for pattern in critical_error_patterns)

    async def cleanup(self) -> None:
        """Enhanced cleanup with proper resource management."""
        if self._shutdown_event.is_set():
            return  # Already cleaning up
            
        logger.info("Starting LLM Manager cleanup...")
        self._shutdown_event.set()

        async with self._manager_lock:
            # Cancel all cleanup tasks
            for task in self.provider_cleanup_tasks:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

            self.provider_cleanup_tasks.clear()

            # Cleanup all providers
            cleanup_errors = []
            for provider_name in self.registry.list_providers():
                try:
                    provider_instance = self.registry.get_provider(provider_name)
                    if hasattr(provider_instance, 'cleanup'):
                        await provider_instance.cleanup()
                    logger.debug(f"Cleaned up provider: {provider_name}")
                except Exception as e:
                    cleanup_errors.append(f"{provider_name}: {e}")
                    logger.warning(f"Cleanup failed for {provider_name}: {e}")

            # Clear provider states
            self.provider_states.clear()

            if cleanup_errors:
                logger.warning(f"Some providers failed to cleanup: {cleanup_errors}")
            else:
                logger.info("LLM Manager cleanup completed successfully")

    def get_provider_status(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed status of all providers."""
        status = {}
        
        for provider_name in self.registry.list_providers():
            state = self._get_provider_state(provider_name)
            status[provider_name] = {
                "is_initialized": state.is_initialized,
                "is_initializing": state.is_initializing,
                "initialization_attempts": state.initialization_attempts,
                "can_retry": state.can_retry_initialization(),
                "last_error": str(state.last_error) if state.last_error else None,
                "last_error_time": state.last_error_time.isoformat() if state.last_error_time else None,
                "backoff_until": state.backoff_until.isoformat() if state.backoff_until else None,
                "health_status": self.load_balancer.provider_health.get(provider_name, True)
            }
        
        return status

    async def reset_provider(self, provider_name: str) -> bool:
        """Reset a provider's state and force reinitialization.
        
        Args:
            provider_name: Name of provider to reset
            
        Returns:
            bool: True if reset successful
        """
        if provider_name not in self.provider_states:
            logger.warning(f"Provider {provider_name} not found in states")
            return False
            
        async with self._manager_lock:
            # Reset state
            state = self.provider_states[provider_name]
            async with state.initialization_lock:
                state.is_initialized = False
                state.is_initializing = False
                state.initialization_attempts = 0
                state.last_error = None
                state.last_error_time = None
                state.backoff_until = None
                
                logger.info(f"Reset provider state: {provider_name}")
                
                # Try to reinitialize
                return await self._ensure_provider_initialized(provider_name)
    """Central orchestrator for LLM providers with fallback and load balancing."""

    def __init__(self,
                 config_manager: Optional[ConfigurationManager] = None,
                 debug_logger: Optional[DebugLogger] = None,
                 metrics_collector: Optional[MetricsCollector] = None,
                 response_normalizer: Optional[ResponseNormalizer] = None,
                 registry: Optional[LLMProviderRegistry] = None):
        """Initialize LLM Manager.

        Args:
            config_manager: Configuration manager instance
            debug_logger: Debug logger instance
            metrics_collector: Metrics collector instance
            response_normalizer: Response normalizer instance
            registry: Provider registry instance
        """
        self.config_manager = config_manager or ConfigurationManager()
        self.debug_logger = debug_logger or get_debug_logger()
        self.metrics_collector = metrics_collector or get_metrics_collector()
        self.response_normalizer = response_normalizer or get_response_normalizer()
        self.registry = registry or get_global_registry()

        self.fallback_strategy = FallbackStrategy(self.debug_logger)
        self.load_balancer = LoadBalancer()

        self.fallback_chain: List[str] = []
        self.default_provider: Optional[str] = None
        self.load_balancing_enabled: bool = False
        self.load_balancing_strategy: str = "round_robin"

        # Initialize providers from configuration
        self._initialize_providers()

    def _initialize_providers(self) -> None:
        """Initialize providers from configuration."""
        try:
            # Import providers to trigger registration
            from .providers import (
                OpenAIProvider,
                OpenRouterProvider,
                DeepSeekProvider,
                AnthropicProvider,
                GeminiProvider
            )

            # Get configured providers
            configured_providers = self.config_manager.list_configured_providers()

            # Initialize each configured provider
            for provider_name in configured_providers:
                try:
                    config = self.config_manager.load_provider_config(provider_name)
                    provider = self.registry.get_provider(provider_name, config.model_dump())

                    # Schedule provider initialization for later (lazy initialization)
                    # The provider will be initialized when first used
                    logger.info(f"Configured provider: {provider_name}")

                except Exception as e:
                    logger.error(f"Failed to configure provider {provider_name}: {e}")

            # Set default provider
            self.default_provider = self.config_manager.get_default_provider()

            # Set fallback chain from configuration
            if not self.fallback_chain:
                self.fallback_chain = self.config_manager.get_fallback_chain()

            logger.info(f"LLM Manager initialized with providers: {configured_providers}")
            logger.info(f"Default provider: {self.default_provider}")
            logger.info(f"Fallback chain: {self.fallback_chain}")

        except Exception as e:
            logger.error(f"Failed to initialize LLM Manager: {e}")
            raise ConfigurationError(f"LLM Manager initialization failed: {str(e)}")

    async def chat(self, messages: List[Message], provider: Optional[str] = None, **kwargs) -> LLMResponse:
        """Send chat request with automatic provider selection and fallback.

        Args:
            messages: List of conversation messages
            provider: Specific provider to use (optional)
            **kwargs: Additional parameters

        Returns:
            LLMResponse: Normalized response
        """
        # Determine provider(s) to use
        providers = self._get_providers_for_request(provider)

        # Define the operation
        async def chat_operation(provider_name: str) -> LLMResponse:
            return await self._execute_provider_operation(
                provider_name, 'chat', messages, **kwargs
            )

        # Execute with fallback
        if len(providers) > 1:
            response = await self.fallback_strategy.execute_with_fallback(
                providers, chat_operation
            )
        else:
            response = await chat_operation(providers[0])

        # Normalize and return response
        return self.response_normalizer.normalize_response(response)

    async def chat_stream(self, messages: List[Message], provider: Optional[str] = None, **kwargs) -> AsyncGenerator[str, None]:
        """Send streaming chat request.

        Args:
            messages: List of conversation messages
            provider: Specific provider to use (optional)
            **kwargs: Additional parameters

        Yields:
            str: Streaming response chunks
        """
        # Determine provider to use (no fallback for streaming)
        providers = self._get_providers_for_request(provider)
        provider_name = providers[0]

        # Get provider instance
        provider_instance = self.registry.get_provider(provider_name)

        # Log request
        request_id = self.debug_logger.log_request(provider_name, 'chat_stream', kwargs)
        start_time = asyncio.get_event_loop().time()

        try:
            async for chunk in provider_instance.chat_stream(messages, **kwargs):
                yield chunk

            # Log successful completion
            duration = asyncio.get_event_loop().time() - start_time
            self.metrics_collector.record_request(
                provider_name, 'chat_stream', duration, True
            )

        except Exception as e:
            # Log error
            duration = asyncio.get_event_loop().time() - start_time
            self.debug_logger.log_error(request_id, e, {"provider": provider_name})
            self.metrics_collector.record_request(
                provider_name, 'chat_stream', duration, False, error=str(e)
            )
            raise

    async def completion(self, prompt: str, provider: Optional[str] = None, **kwargs) -> LLMResponse:
        """Send completion request.

        Args:
            prompt: Text prompt
            provider: Specific provider to use (optional)
            **kwargs: Additional parameters

        Returns:
            LLMResponse: Normalized response
        """
        # Determine provider(s) to use
        providers = self._get_providers_for_request(provider)

        # Define the operation
        async def completion_operation(provider_name: str) -> LLMResponse:
            return await self._execute_provider_operation(
                provider_name, 'completion', prompt, **kwargs
            )

        # Execute with fallback
        if len(providers) > 1:
            response = await self.fallback_strategy.execute_with_fallback(
                providers, completion_operation
            )
        else:
            response = await completion_operation(providers[0])

        # Normalize and return response
        return self.response_normalizer.normalize_response(response)

    async def chat_with_tools(self, messages: List[Message], tools: List[Dict],
                            provider: Optional[str] = None, **kwargs) -> LLMResponse:
        """Send tool-enabled chat request.

        Args:
            messages: List of conversation messages
            tools: List of available tools
            provider: Specific provider to use (optional)
            **kwargs: Additional parameters

        Returns:
            LLMResponse: Normalized response
        """
        # Determine provider(s) to use
        providers = self._get_providers_for_request(provider)

        # Filter providers that support tools
        tool_capable_providers = []
        for p in providers:
            try:
                capabilities = self.registry.get_capabilities(p)
                if ProviderCapability.TOOLS in capabilities:
                    tool_capable_providers.append(p)
                    logger.debug(f"Provider {p} supports tools")
                else:
                    logger.debug(f"Provider {p} does not support tools: {capabilities}")
            except Exception as e:
                logger.warning(f"Failed to check capabilities for provider {p}: {e}")
                continue

        if not tool_capable_providers:
            raise ProviderError(
                "manager",
                "No available providers support tool calls",
                context={"requested_providers": providers, "tools": tools}
            )

        # Define the operation
        async def tools_operation(provider_name: str) -> LLMResponse:
            return await self._execute_provider_operation(
                provider_name, 'chat_with_tools', messages, tools, **kwargs
            )

        # Execute with fallback
        if len(tool_capable_providers) > 1:
            response = await self.fallback_strategy.execute_with_fallback(
                tool_capable_providers, tools_operation
            )
        else:
            response = await tools_operation(tool_capable_providers[0])

        # Normalize and return response
        return self.response_normalizer.normalize_response(response)

    async def _execute_provider_operation(self, provider_name: str, method: str, *args, **kwargs) -> LLMResponse:
        """Execute an operation on a specific provider with monitoring.

        Args:
            provider_name: Name of the provider
            method: Method to call on the provider
            *args, **kwargs: Arguments for the method

        Returns:
            LLMResponse: Provider response
        """
        # Get provider instance with configuration
        config = self.config_manager.load_provider_config(provider_name)
        provider_instance = self.registry.get_provider(provider_name, config.model_dump())

        # Ensure provider is initialized (lazy initialization)
        if not hasattr(provider_instance, '_initialized') or not provider_instance._initialized:
            try:
                await provider_instance.initialize(config.model_dump())
                provider_instance._initialized = True
                logger.info(f"Lazy initialized provider: {provider_name}")
            except Exception as e:
                logger.error(f"Failed to initialize provider {provider_name}: {e}")
                raise ProviderError(provider_name, f"Provider initialization failed: {str(e)}", original_error=e)

        # Log request
        request_id = self.debug_logger.log_request(provider_name, method, kwargs)
        start_time = asyncio.get_event_loop().time()

        try:
            # Execute the operation
            operation = getattr(provider_instance, method)
            response = await operation(*args, **kwargs)

            # Calculate duration
            duration = asyncio.get_event_loop().time() - start_time
            response.duration = duration
            response.request_id = request_id

            # Log successful response
            self.debug_logger.log_response(request_id, response, duration)

            # Record metrics
            tokens = response.usage.get('total_tokens', 0) if response.usage else 0
            self.metrics_collector.record_request(
                provider_name, method, duration, True, tokens, response.model
            )

            return response

        except Exception as e:
            # Calculate duration
            duration = asyncio.get_event_loop().time() - start_time

            # Log error
            self.debug_logger.log_error(request_id, e, {"provider": provider_name, "method": method})

            # Record metrics
            self.metrics_collector.record_request(
                provider_name, method, duration, False, error=str(e)
            )

            # Update load balancer health
            self.load_balancer.update_provider_health(provider_name, False)

            raise

    def _get_providers_for_request(self, requested_provider: Optional[str]) -> List[str]:
        """Get list of providers to use for a request.

        Args:
            requested_provider: Specific provider requested (optional)

        Returns:
            List[str]: List of provider names in order of preference
        """
        if requested_provider:
            # Use specific provider only
            if not self.registry.is_registered(requested_provider):
                raise ConfigurationError(f"Provider '{requested_provider}' not registered")
            return [requested_provider]

        # Use load balancing if enabled and multiple providers available
        if self.load_balancing_enabled and len(self.fallback_chain) > 1:
            primary_provider = self.load_balancer.select_provider(
                self.fallback_chain, self.load_balancing_strategy
            )
            # Return primary provider first, then rest of fallback chain
            fallback_providers = [p for p in self.fallback_chain if p != primary_provider]
            return [primary_provider] + fallback_providers

        # Use fallback chain or default provider
        if self.fallback_chain:
            return self.fallback_chain.copy()
        elif self.default_provider:
            return [self.default_provider]
        else:
            # Use any available provider
            available = self.registry.list_providers()
            if not available:
                raise ConfigurationError("No providers available")
            return available[:1]

    def set_fallback_chain(self, providers: List[str]) -> None:
        """Set fallback provider chain.

        Args:
            providers: List of provider names in fallback order
        """
        # Validate providers
        for provider in providers:
            if not self.registry.is_registered(provider):
                raise ConfigurationError(f"Provider '{provider}' not registered")

        self.fallback_chain = providers.copy()
        logger.info(f"Set fallback chain: {self.fallback_chain}")

    def enable_load_balancing(self, strategy: str = "round_robin") -> None:
        """Enable load balancing with specified strategy.

        Args:
            strategy: Load balancing strategy ('round_robin', 'weighted', 'random')
        """
        valid_strategies = ['round_robin', 'weighted', 'random']
        if strategy not in valid_strategies:
            raise ConfigurationError(f"Invalid load balancing strategy: {strategy}")

        self.load_balancing_enabled = True
        self.load_balancing_strategy = strategy
        logger.info(f"Enabled load balancing with strategy: {strategy}")

    def disable_load_balancing(self) -> None:
        """Disable load balancing."""
        self.load_balancing_enabled = False
        logger.info("Disabled load balancing")

    async def health_check_all(self) -> Dict[str, bool]:
        """Check health of all registered providers.

        Returns:
            Dict[str, bool]: Provider health status
        """
        health_status = {}

        for provider_name in self.registry.list_providers():
            try:
                provider_instance = self.registry.get_provider(provider_name)

                # Try to initialize provider if not already initialized
                if hasattr(provider_instance, 'client') and provider_instance.client is None:
                    try:
                        config = self.config_manager.load_provider_config(provider_name)
                        await provider_instance.initialize(config.model_dump())
                        logger.info(f"Initialized provider {provider_name} for health check")
                    except Exception as init_error:
                        logger.warning(f"Failed to initialize provider {provider_name} for health check: {init_error}")
                        health_status[provider_name] = False
                        self.load_balancer.update_provider_health(provider_name, False)
                        continue

                # Perform health check
                is_healthy = await provider_instance.health_check()
                health_status[provider_name] = is_healthy

                # Update load balancer
                self.load_balancer.update_provider_health(provider_name, is_healthy)

            except Exception as e:
                logger.warning(f"Health check failed for {provider_name}: {e}")
                health_status[provider_name] = False
                self.load_balancer.update_provider_health(provider_name, False)

        return health_status

    async def cleanup(self) -> None:
        """Cleanup all provider resources."""
        for provider_name in self.registry.list_providers():
            try:
                provider_instance = self.registry.get_provider(provider_name)
                await provider_instance.cleanup()
            except Exception as e:
                logger.warning(f"Cleanup failed for {provider_name}: {e}")

        logger.info("LLM Manager cleanup completed")

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics.

        Returns:
            Dict[str, Any]: Manager and provider statistics
        """
        return {
            "manager": {
                "default_provider": self.default_provider,
                "fallback_chain": self.fallback_chain,
                "load_balancing_enabled": self.load_balancing_enabled,
                "load_balancing_strategy": self.load_balancing_strategy,
                "registered_providers": self.registry.list_providers()
            },
            "providers": self.metrics_collector.get_all_stats(),
            "summary": self.metrics_collector.get_summary()
        }


# Global manager instance
_global_manager: Optional[LLMManager] = None


def get_llm_manager() -> LLMManager:
    """Get global LLM manager instance.

    Returns:
        LLMManager: Global manager instance
    """
    global _global_manager
    if _global_manager is None:
        _global_manager = LLMManager()
    return _global_manager


def set_llm_manager(manager: LLMManager) -> None:
    """Set global LLM manager instance.

    Args:
        manager: Manager instance to set as global
    """
    global _global_manager
    _global_manager = manager