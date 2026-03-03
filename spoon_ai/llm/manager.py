"""
LLM Manager - Central orchestrator for managing providers, fallback, and load balancing.
"""

from __future__ import annotations

import asyncio
import random
from typing import Any, AsyncGenerator
from logging import getLogger

from contextlib import asynccontextmanager
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from spoon_ai.schema import Message, LLMResponseChunk
from .interface import LLMProviderInterface, LLMResponse, ProviderCapability
from .registry import LLMProviderRegistry, get_global_registry
from .config import ConfigurationManager
from .monitoring import DebugLogger, MetricsCollector, get_debug_logger, get_metrics_collector
from .response_normalizer import ResponseNormalizer, get_response_normalizer
from .errors import ProviderError, ConfigurationError, ProviderUnavailableError
from spoon_ai.callbacks.base import BaseCallbackHandler
from spoon_ai.callbacks.manager import CallbackManager

logger = getLogger(__name__)


@dataclass
class ProviderState:
    """Track provider initialization and health state."""
    is_initializing: bool = False
    is_initialized: bool = False
    initialization_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_error: Exception | None = None
    last_error_time: datetime | None = None
    initialization_attempts: int = 0
    max_attempts: int = 3
    backoff_until: datetime | None = None

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

    async def execute_with_fallback(self, providers: list[str], operation, *args, **kwargs) -> LLMResponse:
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
        self.provider_weights: dict[str, float] = {}
        self.provider_health: dict[str, bool] = {}

    def select_provider(self, providers: list[str], strategy: str = "round_robin") -> str:
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

    def _round_robin_selection(self, providers: list[str]) -> str:
        """Simple round-robin selection."""
        if not hasattr(self, '_round_robin_index'):
            self._round_robin_index = 0

        provider = providers[self._round_robin_index % len(providers)]
        self._round_robin_index += 1
        return provider

    def _weighted_selection(self, providers: list[str]) -> str:
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
                 config_manager: ConfigurationManager | None = None,
                 debug_logger: DebugLogger | None = None,
                 metrics_collector: MetricsCollector | None = None,
                 response_normalizer: ResponseNormalizer | None = None,
                 registry: LLMProviderRegistry | None = None):
        """Initialize LLM Manager with enhanced provider state tracking."""
        self.config_manager = config_manager or ConfigurationManager()
        self.debug_logger = debug_logger or get_debug_logger()
        self.metrics_collector = metrics_collector or get_metrics_collector()
        self.response_normalizer = response_normalizer or get_response_normalizer()
        self.registry = registry or get_global_registry()

        self.fallback_strategy = FallbackStrategy(self.debug_logger)
        self.load_balancer = LoadBalancer()

        # Enhanced provider state management
        self.provider_states: dict[str, ProviderState] = {}
        self.provider_cleanup_tasks: set[asyncio.Task] = set()
        self._manager_lock = asyncio.Lock()
        self._shutdown_event = asyncio.Event()

        # Existing configuration
        self.fallback_chain: list[str] = []
        self.default_provider: str | None = None
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
                # Try to get the running event loop (Python 3.10+ safe)
                try:
                    loop = asyncio.get_running_loop()
                    # If we're inside a running loop, schedule cleanup as a task (best-effort).
                    # We intentionally do NOT block here.
                    asyncio.create_task(self.cleanup())
                except RuntimeError:
                    # No running loop (common during interpreter shutdown).
                    #
                    # IMPORTANT:
                    # Do not create a new event loop here. Some third-party SDKs
                    # (notably google-genai) schedule async cleanup tasks in
                    # `__del__` when a loop is running. Spinning up a loop solely
                    # for atexit cleanup can therefore cause noisy shutdown warnings:
                    # "Task was destroyed but it is pending!" / "aclose was never awaited".
                    #
                    # Users who need deterministic cleanup should call
                    # `await get_llm_manager().cleanup()` explicitly.
                    return
            except Exception as e:
                # Silently skip cleanup on shutdown errors
                logger.debug(f"Cleanup error (safe to ignore at shutdown): {e}")

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

            # Cleanup only initialized providers (those with instances or states)
            cleanup_errors = []
            # Only cleanup providers that have been initialized (have instances or states)
            providers_to_cleanup = set()

            # Add providers that have instances (safely access private attribute)
            registry_instances = getattr(self.registry, '_instances', {})
            if registry_instances:
                providers_to_cleanup.update(registry_instances.keys())

            # Add providers that have states (initialized)
            providers_to_cleanup.update(self.provider_states.keys())

            for provider_name in providers_to_cleanup:
                try:
                    # Only try to cleanup if provider has an instance
                    if provider_name in registry_instances:
                        provider_instance = registry_instances[provider_name]
                        if hasattr(provider_instance, 'cleanup'):
                            await provider_instance.cleanup()
                        logger.debug(f"Cleaned up provider: {provider_name}")
                except Exception as e:
                    # Only track and log non-configuration errors
                    error_str = str(e).lower()
                    if "configuration error" in error_str or "not configured" in error_str:
                        logger.debug(f"Skipping cleanup for unconfigured provider {provider_name}")
                    elif "event loop is closed" in error_str:
                        # During interpreter shutdown, the event loop may already be closed.
                        # Cleanup is best-effort; don't spam warnings in this case.
                        logger.debug(f"Skipping cleanup for provider {provider_name}: event loop is closed")
                    else:
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
    def _initialize_providers(self) -> None:
        """Initialize providers from configuration."""
        try:
            # Import providers to trigger registration
            from .providers import (
                OpenAIProvider,
                OpenRouterProvider,
                DeepSeekProvider,
                AnthropicProvider,
                GeminiProvider,
                OllamaProvider,
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
                configured_chain = self.config_manager.get_fallback_chain()
                self.fallback_chain = self._sanitize_provider_chain(configured_chain)

            logger.info(f"LLM Manager initialized with providers: {configured_providers}")
            logger.info(f"Default provider: {self.default_provider}")
            logger.info(f"Fallback chain: {self.fallback_chain}")

        except Exception as e:
            logger.error(f"Failed to initialize LLM Manager: {e}")
            raise ConfigurationError(f"LLM Manager initialization failed: {str(e)}")

    async def chat(self, messages: list[Message], provider: str | None = None, **kwargs) -> LLMResponse:
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

    async def chat_stream(self,messages: list[Message],provider: str | None = None,callbacks: Optional[list[BaseCallbackHandler]] = None,**kwargs) -> AsyncGenerator[LLMResponseChunk, None]:
        """Send streaming chat request with callback support.
        Args:
            messages: List of conversation messages
            provider: Specific provider to use (optional)
            callbacks: Optional callback handlers for monitoring
            **kwargs: Additional parameters

        Yields:
            LLMResponseChunk: Structured streaming response chunks
        """
        # Determine provider to use (no fallback for streaming)
        providers = self._get_providers_for_request(provider)
        provider_name = providers[0]
        await self._ensure_provider_initialized(provider_name)

        # Get provider instance
        provider_instance = self.registry.get_provider(provider_name)

        # Create callback manager with internal monitoring callbacks
        internal_callbacks = self._get_internal_callbacks()
        all_callbacks = internal_callbacks + (callbacks or [])
        callback_manager = CallbackManager.from_callbacks(all_callbacks)

        # Log request
        request_id = self.debug_logger.log_request(provider_name, 'chat_stream', kwargs)
        start_time = asyncio.get_event_loop().time()

        try:
            # Stream from provider with callbacks
            async for chunk in provider_instance.chat_stream(messages,callbacks=all_callbacks,**kwargs):
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

    def _get_internal_callbacks(self) -> list[BaseCallbackHandler]:
        """Get internal monitoring callbacks."""
        # For now, return empty list
        return []

    async def completion(self, prompt: str, provider: str | None = None, **kwargs) -> LLMResponse:
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

    async def chat_with_tools(self, messages: list[Message], tools: list[dict],
                            provider: str | None = None, **kwargs) -> LLMResponse:
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

    def _sanitize_provider_chain(self, providers: list[str] | None) -> list[str]:
        """Remove duplicates and unknown providers while preserving order."""
        sanitized: list[str] = []
        if not providers:
            return sanitized

        for provider in providers:
            if not provider:
                continue
            if not self.registry.is_registered(provider):
                logger.warning(f"Provider '{provider}' referenced in fallback chain is not registered; skipping")
                continue
            if provider not in sanitized:
                sanitized.append(provider)
        return sanitized

    def _resolve_fallback_candidates(self) -> list[str]:
        """Determine fallback candidates from config or available providers."""
        if self.fallback_chain:
            return self.fallback_chain.copy()

        dynamic_chain = self.config_manager.get_available_providers_by_priority()
        if dynamic_chain:
            return self._sanitize_provider_chain(dynamic_chain)

        return []

    def _build_provider_chain(self) -> list[str]:
        """Construct ordered provider chain starting with the default provider."""
        providers: list[str] = []

        if self.default_provider:
            if self.registry.is_registered(self.default_provider):
                providers.append(self.default_provider)
            else:
                logger.warning(f"Default provider '{self.default_provider}' is not registered; ignoring")

        for candidate in self._resolve_fallback_candidates():
            if candidate not in providers:
                if not self.registry.is_registered(candidate):
                    logger.warning(f"Provider '{candidate}' is not registered; skipping")
                    continue
                providers.append(candidate)

        if not providers:
            available = self.registry.list_providers()
            if not available:
                raise ConfigurationError("No providers available")
            providers.append(available[0])

        return providers

    def _get_providers_for_request(self, requested_provider: str | None) -> list[str]:
        """Get list of providers to use for a request.

        Args:
            requested_provider: Specific provider requested (optional)

        Returns:
            list[str]: List of provider names in order of preference
        """
        if requested_provider:
            # Use specific provider only
            if not self.registry.is_registered(requested_provider):
                raise ConfigurationError(f"Provider '{requested_provider}' not registered")
            return [requested_provider]

        providers = self._build_provider_chain()

        # Use load balancing if enabled and multiple providers available
        if self.load_balancing_enabled and len(providers) > 1:
            primary_provider = self.load_balancer.select_provider(
                providers, self.load_balancing_strategy
            )
            fallback_providers = [p for p in providers if p != primary_provider]
            return [primary_provider] + fallback_providers

        return providers

    def set_fallback_chain(self, providers: list[str]) -> None:
        """Set fallback provider chain.

        Args:
            providers: List of provider names in fallback order
        """
        # Validate providers
        sanitized: list[str] = []
        for provider in providers:
            if not self.registry.is_registered(provider):
                raise ConfigurationError(f"Provider '{provider}' not registered")
            if provider not in sanitized:
                sanitized.append(provider)

        self.fallback_chain = sanitized
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

    async def health_check_all(self) -> dict[str, bool]:
        """Check health of all registered providers.

        Returns:
            dict[str, bool]: Provider health status
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
_global_manager: LLMManager | None = None


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
