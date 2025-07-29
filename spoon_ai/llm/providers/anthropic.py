import os
import json
import asyncio
from typing import List, Optional, Dict, Literal, Any
from logging import getLogger

from anthropic import AsyncAnthropic
from httpx import AsyncClient

from spoon_ai.llm.base import LLMBase
from spoon_ai.llm.factory import LLMFactory
from spoon_ai.schema import Message, LLMConfig, LLMResponse, ToolCall, Function
from spoon_ai.utils.config_manager import ConfigManager

logger = getLogger(__name__)
MAX_TOOLS_CACHED = 10

class AnthropicConfig(LLMConfig):
    """Anthropic-specific configuration"""
    model: str = "claude-sonnet-4-20250514"
    enable_prompt_cache: bool = True

@LLMFactory.register("anthropic")
class AnthropicProvider(LLMBase):
    """Anthropic provider implementation with streaming and prompt caching support"""
    
    def __init__(self, config_path: str = "config.json", config_name: str = "llm"):
        # Use ConfigManager for all configuration (no TOML)
        self.config_manager = ConfigManager()
        
        # Load configuration using ConfigManager instead of TOML
        self.config = self._load_config_from_json()
        
        # Get API key with config.json -> environment fallback
        api_key = self.config_manager.get_api_key("anthropic") or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key not found in config.json or ANTHROPIC_API_KEY environment variable")
        
        # Initialize cache metrics tracking
        self.cache_metrics = {
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "total_input_tokens": 0
        }
        
        # Initialize Anthropic client with official endpoint (ignore env overrides)
        http_client = AsyncClient(follow_redirects=True)
        self.client = AsyncAnthropic(
            api_key=api_key,
            base_url="https://api.anthropic.com",  # Force official endpoint
            http_client=http_client
        )
    
    def _load_config_from_json(self) -> AnthropicConfig:
        """Load configuration from config.json via ConfigManager"""
        # Get model from config.json or use default
        model_name = self.config_manager.get("model_name") or "claude-sonnet-4-20250514"
        
        return AnthropicConfig(
            model=model_name,
            enable_prompt_cache=True,
            max_tokens=4096,
            temperature=0.3
        )
    
    def _load_config(self, config_path: str, config_name: str) -> AnthropicConfig:
        """Load Anthropic-specific configuration (for compatibility only)"""
        # This method is for compatibility with LLMBase interface
        # We use _load_config_from_json() instead in the constructor
        return self._load_config_from_json()
    
    def _log_cache_metrics(self, usage_data) -> None:
        """Log cache metrics from Anthropic API response usage data"""
        if self.config.enable_prompt_cache and usage_data:
            if hasattr(usage_data, 'cache_creation_input_tokens') and usage_data.cache_creation_input_tokens:
                self.cache_metrics["cache_creation_input_tokens"] += usage_data.cache_creation_input_tokens
                logger.info(f"Cache creation tokens: {usage_data.cache_creation_input_tokens}")
            if hasattr(usage_data, 'cache_read_input_tokens') and usage_data.cache_read_input_tokens:
                self.cache_metrics["cache_read_input_tokens"] += usage_data.cache_read_input_tokens
                logger.info(f"Cache read tokens: {usage_data.cache_read_input_tokens}")
            if hasattr(usage_data, 'input_tokens') and usage_data.input_tokens:
                self.cache_metrics["total_input_tokens"] += usage_data.input_tokens
    
    def get_cache_metrics(self) -> dict:
        """Get current cache performance metrics"""
        return self.cache_metrics.copy()
    
    def _format_system_message_with_cache(self, system_msg: str) -> Any:
        """Format system message with cache control if applicable"""
        if not system_msg or system_msg.strip() == "":
            return None
            
        # Use ~4000 chars to ensure we hit 1024 tokens (rough approximation: 1 token ≈ 4 chars)
        if self.config.enable_prompt_cache and len(system_msg) >= 4000:
            logger.info(f"Applied cache_control to system message ({len(system_msg)} chars)")
            return [
                {
                    "type": "text",
                    "text": system_msg,
                    "cache_control": {"type": "ephemeral"}
                }
            ]
        return system_msg

    async def chat(
        self,
        messages: List[Message],
        system_msgs: Optional[List[Message]] = None,
        **kwargs
    ) -> LLMResponse:
        """Send chat request to Anthropic"""
        # Format system message with cache control
        system_formatted = None
        if system_msgs:
            system_content = " ".join([msg.content for msg in system_msgs])
            system_formatted = self._format_system_message_with_cache(system_content)
        
        # Format messages for Anthropic API (exclude system messages)
        anthropic_messages = []
        for message in messages:
            if message.role != "system":  # Anthropic handles system separately
                anthropic_messages.append({
                    "role": message.role,
                    "content": message.content
                })
        
        # Anthropic requires at least one message
        if not anthropic_messages:
            if system_formatted:
                # If we have system message but no user messages, create a minimal user message
                anthropic_messages.append({
                    "role": "user",
                    "content": "Hello"
                })
            else:
                # No messages at all, return empty response
                return LLMResponse(content="", text="")
        
        try:
            # Only include system parameter if we have system content
            create_params = {
                "model": self.config.model,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "messages": anthropic_messages,
                **kwargs
            }
            
            if system_formatted:
                create_params["system"] = system_formatted
            
            logger.debug(f"Anthropic API call params: {create_params}")
            response = await self.client.messages.create(**create_params)
            logger.debug(f"Anthropic API response: {response}")
            
            # Log cache metrics
            if hasattr(response, 'usage'):
                self._log_cache_metrics(response.usage)
            
            # Parse response content properly
            content = ""
            if hasattr(response, 'content') and response.content:
                if isinstance(response.content, list) and len(response.content) > 0:
                    # response.content is a list of content blocks
                    first_block = response.content[0]
                    if hasattr(first_block, 'text'):
                        content = first_block.text
                    else:
                        content = str(first_block)
                elif isinstance(response.content, str):
                    # Sometimes content might be a string directly
                    content = response.content
                else:
                    content = str(response.content)
            else:
                content = ""
                
            return LLMResponse(content=content, text=content)
            
        except Exception as e:
            logger.error(f"Anthropic API request failed: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return LLMResponse(
                content=f"API request failed: {str(e)}", 
                text=f"API request failed: {str(e)}"
            )

    async def completion(self, prompt: str, **kwargs) -> LLMResponse:
        """Send text completion request to Anthropic"""
        message = Message(role="user", content=prompt)
        return await self.chat(messages=[message], **kwargs)

    async def chat_with_tools(
        self,
        messages: List[Message],
        system_msgs: Optional[List[Message]] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: Literal["none", "auto", "required"] = "auto",
        output_queue: Optional[asyncio.Queue] = None,
        **kwargs
    ) -> LLMResponse:
        """Send chat request with tools to Anthropic with streaming support"""
        # Format system message with cache control
        system_content = ""
        if system_msgs:
            system_content = " ".join([msg.content for msg in system_msgs])
        
        system_formatted = self._format_system_message_with_cache(system_content)
        
        def to_anthropic_tools(tools: List[dict]) -> List[dict]:
            """Convert OpenAI tool format to Anthropic format with cache control"""
            anthropic_tools = []
            count = 0
            for tool in tools or []:
                anthropic_tool = {
                    "name": tool["function"]["name"],
                    "description": tool["function"].get("description", ""),
                    "input_schema": tool["function"]["parameters"]
                }
                count += 1
                # Apply cache control only to first MAX_TOOLS_CACHED tools to avoid hitting limits
                if count <= MAX_TOOLS_CACHED and self.config.enable_prompt_cache:
                    anthropic_tool["cache_control"] = {"type": "ephemeral"}
                anthropic_tools.append(anthropic_tool)
            return anthropic_tools

        # Convert message format to Anthropic format
        anthropic_messages = []
        for message in messages:
            role = message.role
            
            # Anthropic only supports user and assistant roles
            if role == "system":
                # System messages are handled separately, skip here
                continue
            elif role == "tool":
                # Tool messages are converted to user messages with tool_result
                anthropic_messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": message.tool_call_id,
                        "content": message.content
                    }]
                })
            elif role == "assistant":
                content = None
                if message.tool_calls:
                    content = []
                    for tool_call in message.tool_calls:
                        tool_fn = tool_call.function
                        try:
                            arguments = json.loads(tool_fn.arguments) if isinstance(tool_fn.arguments, str) else tool_fn.arguments
                        except:
                            arguments = {}

                        content.append({
                            "type": "tool_use",
                            "id": tool_call.id,
                            "name": tool_fn.name,
                            "input": arguments
                        })
                else:
                    content = message.content

                anthropic_messages.append({
                    "role": "assistant",
                    "content": content
                })
            elif role == "user":
                anthropic_messages.append({
                    "role": "user",
                    "content": message.content
                })

        # Convert tools to Anthropic format
        anthropic_tools = to_anthropic_tools(tools) if tools else []
        
        try:
            content = ""
            buffer = ""
            buffer_type = ""
            current_tool = None
            tool_calls = []
            finish_reason = None
            native_finish_reason = None
            output_index = 0

            async with self.client.messages.stream(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                system=system_formatted,
                messages=anthropic_messages,
                tools=anthropic_tools,
                **kwargs
            ) as stream:
                async for chunk in stream:
                    if chunk.type == "message_start":
                        # Log cache metrics from streaming message_start event
                        if hasattr(chunk, 'message') and hasattr(chunk.message, 'usage'):
                            self._log_cache_metrics(chunk.message.usage)
                        continue
                    elif chunk.type == "message_delta":
                        # Extract finish_reason from message delta
                        if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'stop_reason'):
                            finish_reason = chunk.delta.stop_reason
                            native_finish_reason = chunk.delta.stop_reason
                        continue
                    elif chunk.type == "message_stop":
                        # Extract finish_reason from message stop
                        if hasattr(chunk, 'message') and hasattr(chunk.message, 'stop_reason'):
                            finish_reason = chunk.message.stop_reason
                            native_finish_reason = chunk.message.stop_reason
                        continue
                    elif chunk.type in ["text", "input_json"]:
                        continue
                    elif chunk.type == "content_block_start":
                        buffer_type = chunk.content_block.type
                        if output_queue:
                            await output_queue.put({"type": "start", "content_block": chunk.content_block.model_dump(), "index": output_index})
                        if buffer_type == "tool_use":
                            current_tool = {
                                "id": chunk.content_block.id,
                                "function": {
                                    "name": chunk.content_block.name,
                                    "arguments": {}
                                }
                            }
                        continue
                    elif chunk.type == "content_block_delta" and chunk.delta.type == "text_delta":
                        buffer += chunk.delta.text
                        if output_queue:
                            await output_queue.put({"type": "text_delta", "delta": chunk.delta.text, "index": output_index})
                        continue
                    elif chunk.type == "content_block_delta" and chunk.delta.type == "input_json_delta":
                        buffer += chunk.delta.partial_json
                        if output_queue:
                            await output_queue.put({"type": "input_json_delta", "delta": chunk.delta.partial_json, "index": output_index})
                    elif chunk.type == "content_block_stop":
                        content += buffer
                        if buffer_type == "tool_use":
                            current_tool["function"]["arguments"] = buffer
                            current_tool = ToolCall(**current_tool)
                            tool_calls.append(current_tool)
                        buffer = ""
                        buffer_type = ""
                        current_tool = None
                        if output_queue:
                            await output_queue.put({"type": "stop", "content_block": chunk.content_block.model_dump(), "index": output_index})
                        output_index += 1

            # Map Anthropic stop reasons to standard finish reasons
            if finish_reason == "end_turn":
                finish_reason = "stop"
            elif finish_reason == "max_tokens":
                finish_reason = "length"
            elif finish_reason == "tool_use":
                finish_reason = "tool_calls"

            return LLMResponse(
                content=content,
                text=content,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
                native_finish_reason=native_finish_reason
            )
            
        except Exception as e:
            logger.error(f"Anthropic API request failed: {str(e)}")
            return LLMResponse(
                content=f"API request failed: {str(e)}",
                text=f"API request failed: {str(e)}",
                tool_calls=[],
                finish_reason="error"
            ) 