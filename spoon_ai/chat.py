from logging import getLogger
from typing import List, Optional, Union, Dict, Any, Tuple
import asyncio

from spoon_ai.schema import Message, LLMResponse
from spoon_ai.llm.manager import get_llm_manager
from spoon_ai.llm.errors import ConfigurationError
from spoon_ai.memory.short_term_manager import (
    ShortTermMemoryManager,
    TrimStrategy,
    MessageTokenCounter,
)
from spoon_ai.memory.remove_message import (
    RemoveMessage,
    REMOVE_ALL_MESSAGES,
)
from pydantic import BaseModel, Field

logger = getLogger(__name__)


class ShortTermMemoryConfig(BaseModel):
    """Configuration for short-term memory management."""
    
    enabled: bool = True
    """Enable automatic short-term memory management."""
    
    max_tokens: int = 8000
    """Maximum token count before triggering trimming/summarization."""
    
    strategy: str = "summarize"  # "summarize" or "trim"
    """Strategy to use when exceeding max_tokens: 'summarize' or 'trim'."""
    
    messages_to_keep: int = 5
    """Number of recent messages to keep when summarizing."""
    
    trim_strategy: TrimStrategy = TrimStrategy.FROM_END
    """Trimming strategy when using 'trim' mode."""
    
    keep_system_messages: bool = True
    """Always keep system messages during trimming."""
    
    auto_checkpoint: bool = False
    """Automatically save checkpoints before trimming/summarization."""
    
    checkpoint_thread_id: Optional[str] = None
    """Thread ID for checkpoint management."""
    
    summary_model: Optional[str] = None
    """Model to use for summarization (defaults to ChatBot's model)."""


class Memory(BaseModel):

    messages: List[Message] = Field(default_factory=list)
    max_messages: int = 100

    model_config = {"arbitrary_types_allowed": True,}

    def __init__(self, **data):
        super().__init__(**data)
        # Enforce max messages limit when adding new messages
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    def add_message(self, message: Message) -> None:
        self.messages.append(message)
        if len(self.messages) > self.max_messages:
            self.messages.pop(0)

    def get_messages(self) -> List[Message]:
        return self.messages.copy()

    def clear(self) -> None:
        self.messages.clear()

def to_dict(message: Message) -> dict:
    messages = {"role": message.role}
    if message.content:
        messages["content"] = message.content
    if message.tool_calls:
        messages["tool_calls"] = [tool_call.model_dump() for tool_call in message.tool_calls]
    if message.name:
        messages["name"] = message.name
    if message.tool_call_id:
        messages["tool_call_id"] = message.tool_call_id
    return messages


class ChatBot:
    def __init__(self, use_llm_manager: bool = True, model_name: str = None, llm_provider: str = None, api_key: str = None, base_url: str = None, enable_short_term_memory: bool = True,short_term_memory_config: Optional[Union[Dict[str, Any], ShortTermMemoryConfig]] = None,token_counter: Optional[MessageTokenCounter] = None, **kwargs):
        """Initialize ChatBot with hierarchical configuration priority system.

        Configuration Priority System:
        1. Full manual override (highest priority) - all params provided
        2. Partial override with config fallback - llm_provider provided, api_key from config
        3. Full config-based loading - only use_llm_manager=True, loads from config.json

        Args:
            use_llm_manager: Enable LLM manager architecture (default: True)
            model_name: Model name override
            llm_provider: Provider name override
            api_key: API key override
            base_url: Base URL override
            enable_short_term_memory: Enable short-term memory management (default: True)
            short_term_memory_config: Configuration dict or ShortTermMemoryConfig instance
            token_counter: Optional custom token counter instance
            **kwargs: Additional parameters
        """
        self.use_llm_manager = use_llm_manager
        self.model_name = model_name
        self.llm_provider = llm_provider
        self.api_key = api_key
        self.base_url = base_url
        self.llm_manager = None
        self._latest_summary_text: Optional[str] = None
        self._latest_removals: List[RemoveMessage] = []

        # Store original parameters for priority mode detection
        self._original_llm_provider = llm_provider
        self._original_api_key = api_key

        if not self.use_llm_manager:
            logger.warning("use_llm_manager=False is deprecated. LLM Manager architecture is now required.")
            # Force use of LLM manager for compatibility
            self.use_llm_manager = True

        # Initialize based on configuration priority
        self._initialize_with_priority()

        # Initialize short-term memory configuration
        self.short_term_memory_enabled = enable_short_term_memory
        self.short_term_memory_manager = None
        self.short_term_memory_config = None
        
        if enable_short_term_memory:
            # Parse configuration
            if isinstance(short_term_memory_config, ShortTermMemoryConfig):
                self.short_term_memory_config = short_term_memory_config
            elif isinstance(short_term_memory_config, dict):
                self.short_term_memory_config = ShortTermMemoryConfig(**short_term_memory_config)
            else:
                self.short_term_memory_config = ShortTermMemoryConfig()
            
            # Initialize manager
            self.short_term_memory_manager = ShortTermMemoryManager(
                token_counter=token_counter
            )
            
            logger.info(
                f"Short-term memory manager enabled with config: "
                f"max_tokens={self.short_term_memory_config.max_tokens}, "
                f"strategy={self.short_term_memory_config.strategy}, "
                f"messages_to_keep={self.short_term_memory_config.messages_to_keep}"
            )

        logger.info(f"ChatBot initialized with LLM Manager architecture (priority mode: {self._get_priority_mode()})")

    def _get_priority_mode(self) -> str:
        """Determine which priority mode is being used."""
        # Use original parameters to determine mode (before config loading)
        if self._original_api_key and self._original_llm_provider:
            return "full_manual_override"
        elif self._original_llm_provider and not self._original_api_key:
            return "partial_override_with_config_fallback"
        elif self.use_llm_manager and not self._original_llm_provider:
            return "full_config_based_loading"
        else:
            return "default_config_loading"

    def _initialize_with_priority(self) -> None:
        """Initialize ChatBot based on configuration priority system."""
        from spoon_ai.llm.config import ConfigurationManager

        # Always initialize LLM manager
        self.llm_manager = get_llm_manager()

        # Priority 1: Full manual override (highest priority)
        if self.api_key and self.llm_provider:
            logger.info("Using full manual override mode")
            self._apply_manual_override()
            return

        # Priority 2: Partial override with config fallback
        if self.llm_provider and not self.api_key:
            logger.info("Using partial override with config fallback mode")
            self._apply_partial_override_with_config()
            return

        # Priority 3: Full config-based loading
        if self.use_llm_manager and not self.llm_provider:
            logger.info("Using full config-based loading mode")
            self._apply_full_config_loading()
            return

        # Default: Use config with any provided overrides
        logger.info("Using default config loading with overrides")
        self._apply_default_config_loading()

    def _apply_manual_override(self) -> None:
        """Apply full manual override configuration."""
        # All parameters provided manually - highest priority
        self._update_provider_config(
            provider=self.llm_provider,
            api_key=self.api_key,
            base_url=self.base_url,
            model_name=self.model_name
        )
        logger.info(f"Applied manual override for provider: {self.llm_provider}")

    def _apply_partial_override_with_config(self) -> None:
        """Apply partial override with config fallback."""
        from spoon_ai.llm.config import ConfigurationManager

        config_manager = ConfigurationManager()

        # Get config values for the specified provider
        try:
            provider_config = config_manager._get_provider_config_dict(self.llm_provider)

            # Use config values for missing parameters
            config_api_key = provider_config.get('api_key')
            config_base_url = provider_config.get('base_url')
            config_model = provider_config.get('model')

            # Validate provider consistency: ensure API key matches requested provider
            final_api_key = self.api_key or config_api_key
            if final_api_key and not self._validate_provider_api_key_match(self.llm_provider, final_api_key):
                # Get available providers for helpful error message
                available_providers = config_manager.list_configured_providers()
                logger.error(f"Provider/API key mismatch detected: requested '{self.llm_provider}' but API key appears to be for different provider")
                raise ConfigurationError(
                    f"API key mismatch for provider '{self.llm_provider}'. "
                    f"Please ensure the API key in your configuration matches the requested provider. "
                    f"Available providers: {available_providers}",
                    config_key=self.llm_provider,
                    context={
                        "requested_provider": self.llm_provider,
                        "available_providers": available_providers,
                        "api_key_prefix": final_api_key[:10] + "..." if final_api_key else None
                    }
                )

            # Apply configuration with manual overrides taking priority
            self._update_provider_config(
                provider=self.llm_provider,
                api_key=final_api_key,
                base_url=self.base_url or config_base_url,
                model_name=self.model_name or config_model
            )

            logger.info(f"Applied partial override with config fallback for provider: {self.llm_provider}")

        except Exception as e:
            logger.error(f"Failed to load config for provider {self.llm_provider}: {e}")
            # Fallback to manual values only
            self._update_provider_config(
                provider=self.llm_provider,
                api_key=self.api_key,
                base_url=self.base_url,
                model_name=self.model_name
            )

    def _apply_full_config_loading(self) -> None:
        """Apply full config-based loading using default provider and fallback chain."""
        from spoon_ai.llm.config import ConfigurationManager

        config_manager = ConfigurationManager()

        try:
            # Use default provider from config
            default_provider = config_manager.get_default_provider()
            fallback_chain = config_manager.get_fallback_chain()

            if default_provider:
                self.llm_provider = default_provider
                logger.info(f"Using default provider from config: {default_provider}")

            if fallback_chain:
                self.llm_manager.set_fallback_chain(fallback_chain)
                logger.info(f"Set fallback chain from config: {fallback_chain}")

        except Exception as e:
            logger.error(f"Failed to load full config: {e}")
            # Let LLM manager handle provider selection
            pass

    def _apply_default_config_loading(self) -> None:
        """Apply default config loading with any provided overrides."""
        # Apply any manual overrides if provided
        if self.api_key or self.base_url or self.model_name:
            self._update_provider_config(
                provider=self.llm_provider,
                api_key=self.api_key,
                base_url=self.base_url,
                model_name=self.model_name
            )

    def _update_provider_config(self, provider: str, api_key: str = None, base_url: str = None, model_name: str = None):
        """Update provider configuration in the LLM manager."""
        if not provider:
            logger.warning("No provider specified for configuration update")
            return

        try:
            # Get the current configuration manager
            config_manager = self.llm_manager.config_manager

            # Create a temporary configuration update
            config_updates = {}
            if api_key:
                config_updates['api_key'] = api_key
            if base_url:
                config_updates['base_url'] = base_url
            if model_name:
                config_updates['model'] = model_name

            # Update the provider configuration in memory
            if hasattr(config_manager, '_provider_configs'):
                if provider in config_manager._provider_configs:
                    # Update existing config
                    existing_config = config_manager._provider_configs[provider]
                    for key, value in config_updates.items():
                        setattr(existing_config, key, value)
                    logger.info(f"Updated existing provider config for {provider}")
                else:
                    # Create new config
                    from spoon_ai.llm.config import ProviderConfig
                    new_config = ProviderConfig(
                        name=provider,
                        api_key=api_key or '',
                        base_url=base_url,
                        model=model_name or '',
                        max_tokens=4096,
                        temperature=0.3,
                        timeout=30,
                        retry_attempts=3,
                        custom_headers={},
                        extra_params={}
                    )
                    config_manager._provider_configs[provider] = new_config
                    logger.info(f"Created new provider config for {provider}")

        except Exception as e:
            logger.error(f"Failed to update provider configuration: {e}")

    def _validate_provider_api_key_match(self, provider_name: str, api_key: str) -> bool:
        """Validate that an API key belongs to the specified provider family.

        Args:
            provider_name: Name of the provider (e.g., 'openai', 'anthropic')
            api_key: API key to validate

        Returns:
            bool: True if API key matches provider, False otherwise
        """
        if not api_key or not provider_name:
            return True  # Skip validation for empty values

        # Provider family mapping based on API key prefixes
        provider_families = {
            'openai': ['sk-'],
            'anthropic': ['sk-ant-'],
            'openrouter': ['sk-or-'],
            'deepseek': ['sk-'],
            'gemini': ['AIza']
        }

        expected_prefixes = provider_families.get(provider_name.lower(), [])
        if not expected_prefixes:
            # Unknown provider - skip validation
            logger.debug(f"Unknown provider '{provider_name}', skipping API key validation")
            return True

        # Check if API key starts with any expected prefix
        is_match = any(api_key.startswith(prefix) for prefix in expected_prefixes)

        if not is_match:
            logger.debug(f"API key validation failed: provider '{provider_name}' expects prefixes {expected_prefixes}, got key starting with '{api_key[:10]}...'")
        else:
            logger.debug(f"API key validation passed for provider '{provider_name}'")

        return is_match

    async def _apply_short_term_memory_strategy(
        self,
        messages: List[Message],
        model: Optional[str] = None,
    ) -> List[Message]:
        """Apply short-term memory strategy before sending to LLM.
        
        This method is automatically called by ask() and ask_tool() when
        short-term memory is enabled. It ensures messages stay within token limits.
        
        Args:
            messages: Messages to process
            model: Model name for token counting
            
        Returns:
            List[Message]: Processed messages within token limits
        """
        if not self.short_term_memory_enabled or not self.short_term_memory_manager:
            return messages
        
        config = self.short_term_memory_config
        if not config or not config.enabled:
            return messages
        
        # Use specified model or fall back to ChatBot's model
        token_model = config.summary_model or model or self.model_name
        
        # Save checkpoint if enabled
        if config.auto_checkpoint and config.checkpoint_thread_id:
            try:
                self.short_term_memory_manager.save_checkpoint(
                    thread_id=config.checkpoint_thread_id,
                    messages=messages,
                    metadata={"pre_strategy": True}
                )
                logger.debug(f"Auto-checkpoint saved for thread {config.checkpoint_thread_id}")
            except Exception as e:
                logger.warning(f"Failed to save auto-checkpoint: {e}")
        
        # Apply strategy based on configuration
        try:
            if config.strategy == "summarize":
                llm_ready_messages, removals, summary = await self.short_term_memory_manager.summarize_messages(
                    messages=messages,
                    max_tokens_before_summary=config.max_tokens,
                    messages_to_keep=config.messages_to_keep,
                    summary_model=token_model,
                    llm_manager=self.llm_manager,
                    existing_summary=self._latest_summary_text or "",
                )

                if summary:
                    logger.info("Short-term memory: Generated summary of conversation history")
                    self._latest_summary_text = summary
                if removals:
                    logger.info("Short-term memory: Emitted %d removal directives", len(removals))
                    self._latest_removals = removals
                else:
                    self._latest_removals = []

                processed_messages = llm_ready_messages or messages

            elif config.strategy == "trim":
                processed_messages = await self.short_term_memory_manager.trim_messages(
                    messages=messages,
                    max_tokens=config.max_tokens,
                    strategy=config.trim_strategy,
                    keep_system=config.keep_system_messages,
                    model=token_model
                )
                logger.info(
                    f"Short-term memory: Trimmed {len(messages)} -> {len(processed_messages)} messages"
                )
            else:
                logger.warning(f"Unknown short-term memory strategy: {config.strategy}")
                processed_messages = messages

            return processed_messages

        except Exception as e:
            logger.error(f"Failed to apply short-term memory strategy: {e}", exc_info=True)
            # Return original messages on error to avoid breaking the flow
            return messages

    async def ask(self, messages: List[Union[dict, Message]], system_msg: Optional[str] = None, output_queue: Optional[asyncio.Queue] = None) -> str:
        """Ask method using the LLM manager architecture.
        
        Automatically applies short-term memory strategy if enabled.
        """
        # Convert messages to the expected format
        formatted_messages = []
        if system_msg:
            formatted_messages.append(Message(role="system", content=system_msg))

        for message in messages:
            if isinstance(message, dict):
                formatted_messages.append(Message(**message))
            elif isinstance(message, Message):
                formatted_messages.append(message)
            else:
                raise ValueError(f"Invalid message type: {type(message)}")
        
        # Apply short-term memory strategy before sending to LLM
        processed_messages = await self._apply_short_term_memory_strategy(
            formatted_messages,
            model=self.model_name
        )

        # Use LLM manager for the request
        response = await self.llm_manager.chat(
            messages=processed_messages,
            provider=self.llm_provider
        )

        return response.content

    async def ask_tool(self, messages: List[Union[dict, Message]], system_msg: Optional[str] = None, tools: Optional[List[dict]] = None, tool_choice: Optional[str] = None, output_queue: Optional[asyncio.Queue] = None, **kwargs) -> LLMResponse:
        """Ask tool method using the LLM manager architecture.
        
        Automatically applies short-term memory strategy if enabled.
        """
        # Convert messages to the expected format
        formatted_messages = []
        if system_msg:
            formatted_messages.append(Message(role="system", content=system_msg))

        for message in messages:
            if isinstance(message, dict):
                formatted_messages.append(Message(**message))
            elif isinstance(message, Message):
                formatted_messages.append(message)
            else:
                raise ValueError(f"Invalid message type: {type(message)}")
        
        # Apply short-term memory strategy before sending to LLM
        processed_messages = await self._apply_short_term_memory_strategy(
            formatted_messages,
            model=self.model_name
        )

        # Use LLM manager for the tool request
        response = await self.llm_manager.chat_with_tools(
            messages=processed_messages,
            tools=tools or [],
            provider=self.llm_provider,
            **kwargs
        )

        return response

    # Short-term memory management convenience methods

    async def trim_messages(
        self,
        messages: List[Message],
        max_tokens: int,
        strategy: TrimStrategy = TrimStrategy.FROM_END,
        keep_system: bool = True,
        model: Optional[str] = None,
    ) -> List[Message]:
        """Trim messages to stay within the token budget.

        Args:
            messages: List of messages to trim
            max_tokens: Maximum token count to retain
            strategy: Trimming strategy (from_start or from_end)
            keep_system: Whether to always keep the leading system message
            model: Model name for token counting

        Returns:
            List[Message]: Trimmed messages list
        """
        if not self.short_term_memory_manager:
            raise RuntimeError("Short-term memory manager not enabled")

        return await self.short_term_memory_manager.trim_messages(
            messages, max_tokens, strategy, keep_system, model
        )

    def remove_message(self, message_id: str, **kwargs: Any) -> "RemoveMessage":
        """Construct a removal instruction for the message with the given ID."""
        if not message_id:
            raise ValueError("message_id must be provided for removal.")
        return RemoveMessage(id=message_id, **kwargs)

    def remove_all_messages(self) -> "RemoveMessage":
        """Construct a removal instruction that clears the entire history."""
        return RemoveMessage(id=REMOVE_ALL_MESSAGES)

    async def summarize_messages(
        self,
        messages: List[Message],
        max_tokens_before_summary: int,
        messages_to_keep: int = 5,
        summary_model: Optional[str] = None,
        existing_summary: str = "",
    ) -> Tuple[List[Message], List[RemoveMessage], Optional[str]]:
        """Summarize earlier messages and emit removal directives.

        Returns a tuple ``(messages_for_llm, removals, summary_text)`` where
        ``messages_for_llm`` are the messages that should be sent to the language
        model for the next turn, ``removals`` contains ``RemoveMessage``
        directives that should be applied to the stored history, and
        ``summary_text`` is the newly generated summary (if any).

        Args:
            messages: List of messages to process
            max_tokens_before_summary: Token threshold for triggering summary
            messages_to_keep: Number of recent messages to keep uncompressed
            summary_model: Model to use for summarization
            existing_summary: Previously stored summary text

        """
        if not self.short_term_memory_manager:
            raise RuntimeError("Short-term memory manager not enabled")

        return await self.short_term_memory_manager.summarize_messages(
            messages=messages,
            max_tokens_before_summary=max_tokens_before_summary,
            messages_to_keep=messages_to_keep,
            summary_model=summary_model,
            llm_manager=self.llm_manager,
            existing_summary=existing_summary,
        )

    @property
    def latest_summary(self) -> Optional[str]:
        """Return the most recent summary generated by short-term memory."""
        return self._latest_summary_text

    @property
    def latest_removals(self) -> List[RemoveMessage]:
        """Return the most recent removal directives emitted by summarization."""
        return list(self._latest_removals)
    def save_checkpoint(
        self,
        thread_id: str,
        messages: List[Message],
        metadata: Optional[dict] = None,
    ) -> str:
        """Save current message state to checkpoint.

        Args:
            thread_id: Thread identifier
            messages: Messages to save
            metadata: Optional metadata to store

        Returns:
            str: Checkpoint ID
        """
        if not self.short_term_memory_manager:
            raise RuntimeError("Short-term memory manager not enabled")

        return self.short_term_memory_manager.save_checkpoint(thread_id, messages, metadata)

    def restore_checkpoint(
        self,
        thread_id: str,
        checkpoint_id: Optional[str] = None,
    ) -> Optional[List[Message]]:
        """Restore messages from checkpoint.

        Args:
            thread_id: Thread identifier
            checkpoint_id: Optional specific checkpoint ID

        Returns:
            Optional[List[Message]]: Restored messages, or None if checkpoint not found
        """
        if not self.short_term_memory_manager:
            raise RuntimeError("Short-term memory manager not enabled")

        return self.short_term_memory_manager.restore_checkpoint(thread_id, checkpoint_id)

    def list_checkpoints(self, thread_id: str) -> List[dict]:
        """List all checkpoints for a thread.

        Args:
            thread_id: Thread identifier

        Returns:
            List[dict]: List of checkpoint metadata
        """
        if not self.short_term_memory_manager:
            raise RuntimeError("Short-term memory manager not enabled")

        return self.short_term_memory_manager.list_checkpoints(thread_id)

    def clear_checkpoints(self, thread_id: str) -> None:
        """Clear all checkpoints for a thread.

        Args:
            thread_id: Thread identifier
        """
        if not self.short_term_memory_manager:
            raise RuntimeError("Short-term memory manager not enabled")

        self.short_term_memory_manager.clear_checkpoints(thread_id)
