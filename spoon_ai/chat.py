import os
from logging import getLogger
from typing import List, Optional, Union
import json

from spoon_ai.schema import Message, LLMResponse, ToolCall
from spoon_ai.utils.config_manager import ConfigManager
from spoon_ai.llm.factory import LLMFactory

# Import providers to register them
import spoon_ai.llm.providers

from pydantic import BaseModel, Field
import asyncio

logger = getLogger(__name__)

class Memory(BaseModel):
    messages: List[Message] = Field(default_factory=list)
    max_messages: int = 100

    def add_message(self, message:  Message) -> None:
        self.messages.append(message)
        if len(self.messages) > self.max_messages:
            self.messages.pop(0)

    def get_messages(self) -> List[Message]:
        return self.messages

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
    """Simplified ChatBot class using LLMFactory for provider management"""
    
    def __init__(self, model_name: str = None, llm_config: dict = None, llm_provider: str = None, 
                 api_key: str = None, base_url: str = None, enable_prompt_cache: bool = True):
        """Initialize ChatBot with factory-based provider selection
        
        Args:
            model_name: Model name override
            llm_config: Legacy parameter (ignored, kept for compatibility)
            llm_provider: Provider name (openai, anthropic, deepseek, gemini)
            api_key: API key override
            base_url: Base URL override
            enable_prompt_cache: Enable prompt caching (handled by providers)
        """
        # Initialize configuration manager
        self.config_manager = ConfigManager()
        
        # Initialize memory system
        self.memory = Memory()
        
        # Determine provider using existing ConfigManager logic
        self.llm_provider = llm_provider or self.config_manager.get_llm_provider()
        
        # If still no provider, use the same fallback logic as before
        if not self.llm_provider:
            self.llm_provider = self._determine_provider_fallback()
        
        # Create LLM instance using factory
        try:
            self.llm = LLMFactory.create(self.llm_provider)
        except Exception as e:
            logger.error(f"Failed to create LLM provider '{self.llm_provider}': {str(e)}")
            raise ValueError(f"Failed to initialize LLM provider '{self.llm_provider}': {str(e)}")
        
        # Store configuration for backward compatibility
        self.model_name = model_name or self.llm.config.model
        self.api_key = api_key  # Stored but not used (providers handle this)
        self.base_url = base_url  # Stored but not used (providers handle this)
        self.enable_prompt_cache = enable_prompt_cache  # Stored but handled by providers
        self.llm_config = llm_config  # Legacy parameter
        
        # Store initial cache metrics for backward compatibility
        self._local_cache_metrics = {
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "total_input_tokens": 0
        }
        
        logger.info(f"Initialized ChatBot with provider: {self.llm_provider}, model: {self.model_name}")
    
    @property
    def cache_metrics(self) -> dict:
        """Get current cache performance metrics (dynamic property)"""
        if hasattr(self.llm, 'cache_metrics'):
            return self.llm.cache_metrics
        return self._local_cache_metrics
    
    def _determine_provider_fallback(self) -> str:
        """Fallback provider determination logic (matches original behavior)"""
        # Get configured providers from config.json
        config_data = self.config_manager._load_config()
        configured_providers = []
        api_keys = config_data.get("api_keys", {})
        
        if "anthropic" in api_keys and not self.config_manager._is_placeholder_value(api_keys["anthropic"]):
            configured_providers.append("anthropic")
        if "openai" in api_keys and not self.config_manager._is_placeholder_value(api_keys["openai"]):
            configured_providers.append("openai")
        if "deepseek" in api_keys and not self.config_manager._is_placeholder_value(api_keys["deepseek"]):
            configured_providers.append("deepseek")
        if "gemini" in api_keys and not self.config_manager._is_placeholder_value(api_keys["gemini"]):
            configured_providers.append("gemini")
        
        # If config.json has explicit providers, use those
        if configured_providers:
            if "anthropic" in configured_providers:
                logger.info("Using Anthropic API from config")
                return "anthropic"
            elif "openai" in configured_providers:
                logger.info("Using OpenAI API from config")
                return "openai"
            elif "deepseek" in configured_providers:
                logger.info("Using DeepSeek API from config")
                return "deepseek"
            elif "gemini" in configured_providers:
                logger.info("Using Gemini API from config")
                return "gemini"
        else:
            # Fallback to environment variables
            if os.getenv("ANTHROPIC_API_KEY"):
                logger.info("Using Anthropic API from environment")
                return "anthropic"
            elif os.getenv("OPENAI_API_KEY"):
                logger.info("Using OpenAI API from environment")
                return "openai"
            elif os.getenv("DEEPSEEK_API_KEY"):
                logger.info("Using DeepSeek API from environment")
                return "deepseek"
            elif os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
                logger.info("Using Gemini API from environment")
                return "gemini"
            else:
                raise ValueError("No API key found in config or environment. Please configure API keys in config.json or set API key environment variables")
    
    def _log_cache_metrics(self, usage_data) -> None:
        """Log cache metrics - delegated to provider if available"""
        if hasattr(self.llm, '_log_cache_metrics'):
            self.llm._log_cache_metrics(usage_data)
    
    def get_cache_metrics(self) -> dict:
        """Get current cache performance metrics"""
        if hasattr(self.llm, 'get_cache_metrics'):
            return self.llm.get_cache_metrics()
        elif hasattr(self.llm, 'cache_metrics'):
            return self.llm.cache_metrics.copy()
        return self.cache_metrics.copy()

    async def ask(self, messages: List[Union[dict, Message]], system_msg: Optional[str] = None, 
                  output_queue: Optional[asyncio.Queue] = None) -> str:
        """Simple chat request using factory provider
        
        Args:
            messages: List of messages
            system_msg: System message
            output_queue: Output queue for streaming (if supported)
            
        Returns:
            str: Response content
        """
        # Convert to Message objects
        formatted_messages = []
        for message in messages:
            if isinstance(message, dict):
                formatted_messages.append(Message(**message))
            elif isinstance(message, Message):
                formatted_messages.append(message)
            else:
                raise ValueError(f"Invalid message type: {type(message)}")
        
        # Create system messages if provided
        system_msgs = [Message(role="system", content=system_msg)] if system_msg else None
        
        # Use provider's chat method
        response = await self.llm.chat(messages=formatted_messages, system_msgs=system_msgs)
        
        # Update memory
        if system_msgs:
            for msg in system_msgs:
                self.memory.add_message(msg)
        for msg in formatted_messages:
            self.memory.add_message(msg)
        self.memory.add_message(Message(role="assistant", content=response.content))
        
        return response.content or response.text or ""

    async def ask_tool(self, messages: List[Union[dict, Message]], system_msg: Optional[str] = None, 
                      tools: Optional[List[dict]] = None, tool_choice: Optional[str] = None, 
                      output_queue: Optional[asyncio.Queue] = None, **kwargs) -> LLMResponse:
        """Tool-enabled chat request using factory provider
        
        Args:
            messages: List of messages
            system_msg: System message
            tools: Available tools
            tool_choice: Tool choice mode (auto, none, required)
            output_queue: Output queue for streaming (if supported)
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse: Structured response with content, tool_calls, etc.
        """
        if tool_choice not in ["auto", "none", "required"]:
            tool_choice = "auto"
        
        # Convert to Message objects
        formatted_messages = []
        for message in messages:
            if isinstance(message, dict):
                formatted_messages.append(Message(**message))
            elif isinstance(message, Message):
                formatted_messages.append(message)
            else:
                raise ValueError(f"Invalid message type: {type(message)}")
        
        # Create system messages if provided
        system_msgs = [Message(role="system", content=system_msg)] if system_msg else None
        
        try:
            # Use provider's chat_with_tools method
            response = await self.llm.chat_with_tools(
                messages=formatted_messages,
                system_msgs=system_msgs,
                tools=tools,
                tool_choice=tool_choice,
                output_queue=output_queue,
                **kwargs
            )
            
            # Update memory
            if system_msgs:
                for msg in system_msgs:
                    self.memory.add_message(msg)
            for msg in formatted_messages:
                self.memory.add_message(msg)
            
            # Add assistant response to memory
            assistant_message = Message(
                role="assistant", 
                content=response.content,
                tool_calls=response.tool_calls
            )
            self.memory.add_message(assistant_message)
            
            return response
            
        except Exception as e:
            logger.error(f"Error during tool call: {e}")
            raise e
