"""
OpenAI Provider implementation for the unified LLM interface.
"""

import asyncio
import json
from typing import List, Dict, Any, Optional, AsyncGenerator
from logging import getLogger

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from spoon_ai.schema import Message, ToolCall, Function
from ..interface import LLMProviderInterface, LLMResponse, ProviderMetadata, ProviderCapability
from ..errors import ProviderError, AuthenticationError, RateLimitError, ModelNotFoundError, NetworkError
from ..registry import register_provider

logger = getLogger(__name__)


@register_provider("openai", [
    ProviderCapability.CHAT,
    ProviderCapability.COMPLETION,
    ProviderCapability.TOOLS,
    ProviderCapability.STREAMING
])
class OpenAIProvider(LLMProviderInterface):
    """OpenAI provider implementation."""
    
    def __init__(self):
        self.client: Optional[AsyncOpenAI] = None
        self.config: Dict[str, Any] = {}
        self.model: str = ""
        self.max_tokens: int = 4096
        self.temperature: float = 0.3
        
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the OpenAI provider with configuration."""
        try:
            self.config = config
            self.model = config.get('model', 'gpt-4.1')
            self.max_tokens = config.get('max_tokens', 4096)
            self.temperature = config.get('temperature', 0.3)
            
            api_key = config.get('api_key')
            if not api_key:
                raise AuthenticationError("openai", context={"config": config})
            
            base_url = config.get('base_url')
            timeout = config.get('timeout', 30)
            
            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout
            )
            
            logger.info(f"OpenAI provider initialized with model: {self.model}")
            
        except Exception as e:
            if isinstance(e, (AuthenticationError, ProviderError)):
                raise
            raise ProviderError("openai", f"Failed to initialize: {str(e)}", original_error=e)
    
    def _convert_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert Message objects to OpenAI format with validation."""
        openai_messages = []
        
        for i, message in enumerate(messages):
            msg_dict = {"role": message.role}
            
            if message.content:
                msg_dict["content"] = message.content
            
            if message.tool_calls:
                msg_dict["tool_calls"] = [
                    {
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    }
                    for tool_call in message.tool_calls
                ]
            
            if message.name:
                msg_dict["name"] = message.name
            
            if message.tool_call_id:
                msg_dict["tool_call_id"] = message.tool_call_id
            
            # Validate tool message placement
            if message.role == "tool":
                # Check if previous message is assistant with tool_calls
                if i == 0 or openai_messages[-1]["role"] != "assistant" or "tool_calls" not in openai_messages[-1]:
                    logger.warning(f"Tool message at index {i} is not preceded by assistant message with tool_calls. Skipping.")
                    continue
            
            openai_messages.append(msg_dict)
        
        return self._validate_and_fix_message_sequence(openai_messages)
    
    def _validate_and_fix_message_sequence(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and fix message sequence to comply with OpenAI API requirements."""
        if not messages:
            return messages
        
        fixed_messages = []
        i = 0
        
        while i < len(messages):
            current_msg = messages[i]
            
            # Handle tool messages
            if current_msg["role"] == "tool":
                # Find the preceding assistant message with tool_calls
                assistant_msg_idx = -1
                for j in range(len(fixed_messages) - 1, -1, -1):
                    if fixed_messages[j]["role"] == "assistant" and "tool_calls" in fixed_messages[j]:
                        assistant_msg_idx = j
                        break
                
                if assistant_msg_idx == -1:
                    # No preceding assistant message with tool_calls found, skip this tool message
                    logger.warning(f"Skipping tool message without preceding assistant tool_calls: {current_msg}")
                    i += 1
                    continue
                
                # Check if this tool message corresponds to any tool_call in the assistant message
                assistant_msg = fixed_messages[assistant_msg_idx]
                tool_call_ids = [tc["id"] for tc in assistant_msg.get("tool_calls", [])]
                
                if current_msg.get("tool_call_id") not in tool_call_ids:
                    logger.warning(f"Tool message tool_call_id {current_msg.get('tool_call_id')} not found in assistant tool_calls. Skipping.")
                    i += 1
                    continue
            
            # Handle system messages - they should be at the beginning
            elif current_msg["role"] == "system":
                # If we already have messages and this is not the first, move it to the beginning
                if fixed_messages and fixed_messages[0]["role"] != "system":
                    # Insert at the beginning, but after any existing system messages
                    insert_idx = 0
                    while insert_idx < len(fixed_messages) and fixed_messages[insert_idx]["role"] == "system":
                        insert_idx += 1
                    fixed_messages.insert(insert_idx, current_msg)
                    i += 1
                    continue
            
            # Handle consecutive messages with the same role (except tool messages)
            elif (fixed_messages and 
                  fixed_messages[-1]["role"] == current_msg["role"] and 
                  current_msg["role"] != "tool"):
                
                # Merge content if both have content
                if (fixed_messages[-1].get("content") and current_msg.get("content")):
                    fixed_messages[-1]["content"] += "\n" + current_msg["content"]
                    i += 1
                    continue
                # If current message has content but previous doesn't, replace
                elif current_msg.get("content") and not fixed_messages[-1].get("content"):
                    fixed_messages[-1] = current_msg
                    i += 1
                    continue
            
            fixed_messages.append(current_msg)
            i += 1
        
        return fixed_messages
    
    def _convert_response(self, response: ChatCompletion, duration: float) -> LLMResponse:
        """Convert OpenAI response to standardized LLMResponse."""
        choice = response.choices[0]
        message = choice.message
        
        # Convert tool calls
        tool_calls = []
        if message.tool_calls:
            for tool_call in message.tool_calls:
                tool_calls.append(ToolCall(
                    id=tool_call.id,
                    type=tool_call.type,
                    function=Function(
                        name=tool_call.function.name,
                        arguments=tool_call.function.arguments
                    )
                ))
        
        # Map finish reasons
        finish_reason = choice.finish_reason
        if finish_reason == "stop":
            standardized_finish_reason = "stop"
        elif finish_reason == "length":
            standardized_finish_reason = "length"
        elif finish_reason == "tool_calls":
            standardized_finish_reason = "tool_calls"
        elif finish_reason == "content_filter":
            standardized_finish_reason = "content_filter"
        else:
            standardized_finish_reason = finish_reason
        
        # Extract usage information
        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        
        return LLMResponse(
            content=message.content or "",
            provider="openai",
            model=response.model,
            finish_reason=standardized_finish_reason,
            native_finish_reason=finish_reason,
            tool_calls=tool_calls,
            usage=usage,
            duration=duration,
            metadata={
                "response_id": response.id,
                "created": response.created,
                "system_fingerprint": getattr(response, 'system_fingerprint', None)
            }
        )
    
    async def chat(self, messages: List[Message], **kwargs) -> LLMResponse:
        """Send chat request to OpenAI."""
        if not self.client:
            raise ProviderError("openai", "Provider not initialized")
        
        try:
            start_time = asyncio.get_event_loop().time()
            
            openai_messages = self._convert_messages(messages)
            
            # Extract parameters
            model = kwargs.get('model', self.model)
            max_tokens = kwargs.get('max_tokens', self.max_tokens)
            temperature = kwargs.get('temperature', self.temperature)
            
            response = await self.client.chat.completions.create(
                model=model,
                messages=openai_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False,
                **{k: v for k, v in kwargs.items() if k not in ['model', 'max_tokens', 'temperature']}
            )
            
            duration = asyncio.get_event_loop().time() - start_time
            return self._convert_response(response, duration)
            
        except Exception as e:
            await self._handle_error(e)
    
    async def chat_stream(self, messages: List[Message], **kwargs) -> AsyncGenerator[str, None]:
        """Send streaming chat request to OpenAI."""
        if not self.client:
            raise ProviderError("openai", "Provider not initialized")
        
        try:
            openai_messages = self._convert_messages(messages)
            
            # Extract parameters
            model = kwargs.get('model', self.model)
            max_tokens = kwargs.get('max_tokens', self.max_tokens)
            temperature = kwargs.get('temperature', self.temperature)
            
            stream = await self.client.chat.completions.create(
                model=model,
                messages=openai_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                **{k: v for k, v in kwargs.items() if k not in ['model', 'max_tokens', 'temperature']}
            )
            
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            await self._handle_error(e)
    
    async def completion(self, prompt: str, **kwargs) -> LLMResponse:
        """Send completion request to OpenAI."""
        # Convert to chat format
        messages = [Message(role="user", content=prompt)]
        return await self.chat(messages, **kwargs)
    
    async def chat_with_tools(self, messages: List[Message], tools: List[Dict], **kwargs) -> LLMResponse:
        """Send chat request with tools to OpenAI."""
        if not self.client:
            raise ProviderError("openai", "Provider not initialized")
        
        try:
            start_time = asyncio.get_event_loop().time()
            
            openai_messages = self._convert_messages(messages)
            
            # Extract parameters
            model = kwargs.get('model', self.model)
            max_tokens = kwargs.get('max_tokens', self.max_tokens)
            temperature = kwargs.get('temperature', self.temperature)
            tool_choice = kwargs.get('tool_choice', 'auto')
            
            response = await self.client.chat.completions.create(
                model=model,
                messages=openai_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                tools=tools,
                tool_choice=tool_choice,
                stream=False,
                **{k: v for k, v in kwargs.items() if k not in ['model', 'max_tokens', 'temperature', 'tool_choice']}
            )
            
            duration = asyncio.get_event_loop().time() - start_time
            return self._convert_response(response, duration)
            
        except Exception as e:
            await self._handle_error(e)
    
    def get_metadata(self) -> ProviderMetadata:
        """Get OpenAI provider metadata."""
        return ProviderMetadata(
            name="openai",
            version="1.0.0",
            capabilities=[
                ProviderCapability.CHAT,
                ProviderCapability.COMPLETION,
                ProviderCapability.TOOLS,
                ProviderCapability.STREAMING
            ],
            max_tokens=128000,  # GPT-4 context limit
            supports_system_messages=True,
            rate_limits={
                "requests_per_minute": 3500,
                "tokens_per_minute": 90000
            }
        )
    
    async def health_check(self) -> bool:
        """Check if OpenAI provider is healthy."""
        if not self.client:
            return False
        
        try:
            # Simple test request
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1
            )
            return True
        except Exception as e:
            logger.warning(f"OpenAI health check failed: {e}")
            return False
    
    async def cleanup(self) -> None:
        """Cleanup OpenAI provider resources."""
        if self.client:
            await self.client.close()
            self.client = None
        logger.info("OpenAI provider cleaned up")
    
    async def _handle_error(self, error: Exception) -> None:
        """Handle and convert OpenAI errors to standardized errors."""
        error_str = str(error).lower()
        
        if "authentication" in error_str or "api key" in error_str:
            raise AuthenticationError("openai", context={"original_error": str(error)})
        elif "rate limit" in error_str:
            raise RateLimitError("openai", context={"original_error": str(error)})
        elif "model" in error_str and "not found" in error_str:
            raise ModelNotFoundError("openai", self.model, context={"original_error": str(error)})
        elif "timeout" in error_str or "connection" in error_str:
            raise NetworkError("openai", "Network error", original_error=error)
        else:
            raise ProviderError("openai", f"Request failed: {str(error)}", original_error=error)