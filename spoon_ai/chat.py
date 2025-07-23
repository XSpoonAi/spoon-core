import os
from logging import getLogger
from typing import List, Optional, Union
import json

from spoon_ai.schema import Message, LLMResponse, ToolCall
from spoon_ai.utils.config_manager import ConfigManager

from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from httpx import AsyncClient
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_random_exponential
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
    def __init__(self, model_name: str = None, llm_config: dict = None, llm_provider: str = None, api_key: str = None, base_url: str = None, enable_prompt_cache: bool = True):
        # Initialize configuration manager
        config_manager = ConfigManager()
        
        # Configure prompt caching
        self.enable_prompt_cache = enable_prompt_cache
        self.cache_metrics = {
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "total_input_tokens": 0
        }

        # Use parameters provided by user first, then fall back to config, then environment
        self.model_name = model_name or config_manager.get_model_name()
        self.llm_provider = llm_provider or config_manager.get_llm_provider()
        self.base_url = base_url or config_manager.get_base_url() or os.getenv("BASE_URL")
        self.api_key = api_key
        self.llm_config = llm_config
        self.output_index = 0

        # If llm_provider is still not specified, determine it from config first, then environment variables
        if self.llm_provider is None:
            # Get configured providers from config.json (ignore environment variables)
            config_data = config_manager._load_config()
            configured_providers = []

            # Only consider providers that are explicitly configured in config.json
            api_keys = config_data.get("api_keys", {})

            if "anthropic" in api_keys and not config_manager._is_placeholder_value(api_keys["anthropic"]):
                configured_providers.append("anthropic")
            if "openai" in api_keys and not config_manager._is_placeholder_value(api_keys["openai"]):
                configured_providers.append("openai")
            if "deepseek" in api_keys and not config_manager._is_placeholder_value(api_keys["deepseek"]):
                configured_providers.append("deepseek")

            # If config.json has explicit providers, only use those (ignore environment)
            if configured_providers:
                if "anthropic" in configured_providers:
                    logger.info("Using Anthropic API from config")
                    self.model_name = self.model_name or "claude-sonnet-4-20250514"
                    self.llm_provider = "anthropic"
                elif "openai" in configured_providers:
                    logger.info("Using OpenAI API from config")
                    self.model_name = self.model_name or "gpt-4.1"
                    self.llm_provider = "openai"
                elif "deepseek" in configured_providers:
                    logger.info("Using DeepSeek API from config")
                    self.model_name = self.model_name or "deepseek-chat"
                    self.llm_provider = "openai"  # DeepSeek uses OpenAI-compatible API
            else:
                # Fallback to environment variables only if no config providers
                if os.getenv("ANTHROPIC_API_KEY"):
                    logger.info("Using Anthropic API from environment")
                    self.model_name = self.model_name or "claude-sonnet-4-20250514"
                    self.llm_provider = "anthropic"
                elif os.getenv("OPENAI_API_KEY"):
                    logger.info("Using OpenAI API from environment")
                    self.model_name = self.model_name or "gpt-4.1"
                    self.llm_provider = "openai"
                else:
                    raise ValueError("No API key found in config or environment. Please configure API keys in config.json or set OPENAI_API_KEY/ANTHROPIC_API_KEY environment variables")

        # Get API key from config if not provided
        # When using a custom base_url (like OpenRouter), use the openai API key
        # since these services typically use OpenAI-compatible APIs
        if not self.api_key:
            if self.base_url:
                # For custom base URLs (like OpenRouter), use openai API key
                self.api_key = config_manager.get_api_key("openai")
            else:
                # For native APIs, use the provider-specific API key
                self.api_key = config_manager.get_api_key(self.llm_provider)

        # Set default model names if still not specified
        if not self.model_name:
            if self.llm_provider == "openai":
                self.model_name = "gpt-4.1"
            elif self.llm_provider == "anthropic":
                self.model_name = "claude-sonnet-4-20250514"

        logger.info(f"Initializing ChatBot with provider: {self.llm_provider}, model: {self.model_name}, base_url: {self.base_url}")

        # Determine API logic to use based on base_url and provider
        # If base_url is specified (like OpenRouter), use OpenAI-compatible API regardless of model name
        # Only use native Anthropic API when using official Anthropic endpoint
        if self.base_url or self.llm_provider == "openai":
            # Use OpenAI-compatible API (works for OpenAI, OpenRouter, and other compatible providers)
            self.api_logic = "openai"
            self.llm = AsyncOpenAI(
                api_key=self.api_key or os.getenv("OPENAI_API_KEY"),
                base_url=self.base_url
            )
        elif self.llm_provider == "anthropic" and not self.base_url:
            # Use native Anthropic API only when no custom base_url is specified
            self.api_logic = "anthropic"
            http_client = AsyncClient(follow_redirects=True)
            self.llm = AsyncAnthropic(
                api_key=self.api_key or os.getenv("ANTHROPIC_API_KEY"),
                http_client=http_client
            )
        else:
            raise ValueError(f"Invalid LLM provider: {llm_provider}")

    def _log_cache_metrics(self, usage_data) -> None:
        """Log cache metrics from Anthropic API response usage data"""
        if self.llm_provider == "anthropic" and self.enable_prompt_cache and usage_data:
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

    async def ask(self, messages: List[Union[dict, Message]], system_msg: Optional[str] = None, output_queue: Optional[asyncio.Queue] = None) -> str:
        formatted_messages = [] if system_msg is None else [{"role": "system", "content": system_msg}]
        for message in messages:
            if isinstance(message, dict):
                formatted_messages.append(message)
            elif isinstance(message, Message):
                formatted_messages.append(to_dict(message))
            else:
                raise ValueError(f"Invalid message type: {type(message)}")

        if self.api_logic == "openai":
            response = await self.llm.chat.completions.create(messages=formatted_messages, model=self.model_name, max_tokens=4096, temperature=0.3, stream=False)
            return response.choices[0].message.content
        elif self.api_logic == "anthropic":
            # Format system message with cache control for Anthropic models if it's long enough
            system_content = system_msg
            # Use ~4000 chars to ensure we hit 1024 tokens (rough approximation: 1 token ≈ 4 chars)
            if system_msg and self.llm_provider == "anthropic" and self.enable_prompt_cache and len(system_msg) >= 4000:
                system_content = [
                    {
                        "type": "text",
                        "text": system_msg,
                        "cache_control": {"type": "ephemeral"}
                    }
                ]
                logger.info(f"Applied cache_control to system message ({len(system_msg)} chars)")
            
            response = await self.llm.messages.create(
                model=self.model_name,
                max_tokens=4096,
                temperature=0.3,
                system=system_content,
                messages=[m for m in formatted_messages if m.get("role") != "system"]
            )
            # Log cache metrics for non-streaming response
            if hasattr(response, 'usage'):
                self._log_cache_metrics(response.usage)
            return response.content[0].text

    # @retry(stop=stop_after_attempt(3), wait=wait_random_exponential(min=1, max=60))
    async def ask_tool(self,messages: List[Union[dict, Message]], system_msg: Optional[str] = None, tools: Optional[List[dict]] = None, tool_choice: Optional[str] = None, output_queue: Optional[asyncio.Queue] = None, **kwargs):
        if tool_choice not in ["auto", "none", "required"]:
            tool_choice = "auto"

        formatted_messages = [] if system_msg is None else [{"role": "system", "content": system_msg}]
        for message in messages:
            if isinstance(message, dict):
                formatted_messages.append(message)
            elif isinstance(message, Message):
                formatted_messages.append(to_dict(message))
            else:
                raise ValueError(f"Invalid message type: {type(message)}")

        try:
            if self.api_logic == "openai":
                response = await self.llm.chat.completions.create(
                    messages=formatted_messages,
                    model=self.model_name,
                    max_tokens=4096,
                    temperature=0.3,
                    stream=False,
                    tools=tools,
                    tool_choice=tool_choice,
                    **kwargs
                )

                # Extract message and finish_reason from OpenAI response
                message = response.choices[0].message
                finish_reason = response.choices[0].finish_reason

                # Convert OpenAI tool calls to our ToolCall format
                tool_calls = []
                if message.tool_calls:
                    from spoon_ai.schema import Function
                    for tool_call in message.tool_calls:
                        tool_calls.append(ToolCall(
                            id=tool_call.id,
                            type=tool_call.type,
                            function=Function(
                                name=tool_call.function.name,
                                arguments=tool_call.function.arguments
                            )
                        ))

                # Map OpenAI finish reasons to standardized values
                standardized_finish_reason = finish_reason
                if finish_reason == "stop":
                    standardized_finish_reason = "stop"
                elif finish_reason == "length":
                    standardized_finish_reason = "length"
                elif finish_reason == "tool_calls":
                    standardized_finish_reason = "tool_calls"
                elif finish_reason == "content_filter":
                    standardized_finish_reason = "content_filter"

                # Return consistent LLMResponse object
                return LLMResponse(
                    content=message.content or "",
                    tool_calls=tool_calls,
                    finish_reason=standardized_finish_reason,
                    native_finish_reason=finish_reason
                )
            elif self.api_logic == "anthropic":
                def to_anthropic_tools(tools: List[dict]) -> List[dict]:
                    anthropic_tools = []
                    for tool in tools:
                        anthropic_tool = {
                            "name": tool["function"]["name"], 
                            "description": tool["function"]["description"], 
                            "input_schema": tool["function"]["parameters"]
                        }
                        # Add cache control for Anthropic models if tools are substantial
                        if self.llm_provider == "anthropic" and self.enable_prompt_cache and len(tools) > 1:
                            anthropic_tool["cache_control"] = {"type": "ephemeral"}
                        anthropic_tools.append(anthropic_tool)
                    return anthropic_tools

                # Convert message format to Anthropic format
                anthropic_messages = []
                
                # Format system message with cache control for Anthropic models if it's long enough
                system_content = system_msg or ""
                # Use ~4000 chars to ensure we hit 1024 tokens (rough approximation: 1 token ≈ 4 chars)
                if system_msg and self.llm_provider == "anthropic" and self.enable_prompt_cache and len(system_msg) >= 4000:
                    system_content = [
                        {
                            "type": "text",
                            "text": system_msg,
                            "cache_control": {"type": "ephemeral"}
                        }
                    ]

                for message in formatted_messages:
                    role = message.get("role")

                    # Anthropic only supports user and assistant roles
                    if role == "system":
                        # System messages are handled above, skip here
                        continue
                    elif role == "tool":
                        # Tool messages are converted to user messages, content contains tool_result
                        anthropic_messages.append({
                            "role": "user",
                            "content": [{
                                "type": "tool_result",
                                "tool_use_id": message.get("tool_call_id"),
                                "content": message.get("content")
                            }]
                        })
                    elif role == "assistant":
                        content = None
                        if message.get("tool_calls"):
                            content = []
                            for tool_call in message.get("tool_calls", []):
                                tool_fn = tool_call.get("function", {})
                                try:
                                    arguments = json.loads(tool_fn.get("arguments", "{}"))
                                except:
                                    arguments = {}

                                content.append({
                                    "type": "tool_use",
                                    "id": tool_call.get("id"),
                                    "name": tool_fn.get("name"),
                                    "input": arguments
                                })
                        else:
                            content = message.get("content")

                        anthropic_messages.append({
                            "role": "assistant",
                            "content": content
                        })
                    elif role == "user":
                        anthropic_messages.append({
                            "role": "user",
                            "content": message.get("content")
                        })

                content = ""
                buffer = ""
                buffer_type = ""
                current_tool = None
                tool_calls = []
                finish_reason = None
                native_finish_reason = None

                async with self.llm.messages.stream(
                    model=self.model_name,
                    max_tokens=4096,
                    temperature=0.3,
                    system=system_content,
                    messages=anthropic_messages,
                    tools=to_anthropic_tools(tools),
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
                                    output_queue.put_nowait({"type": "start", "content_block": chunk.content_block.model_dump(), "index": self.output_index})
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
                                output_queue.put_nowait({"type": "text_delta", "delta": chunk.delta.text, "index": self.output_index})
                            continue
                        elif chunk.type == "content_block_delta" and chunk.delta.type == "input_json_delta":
                            buffer += chunk.delta.partial_json
                            if output_queue:
                                output_queue.put_nowait({"type": "input_json_delta", "delta": chunk.delta.partial_json, "index": self.output_index})

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
                                output_queue.put_nowait({"type": "stop", "content_block": chunk.content_block.model_dump(), "index": self.output_index})
                            self.output_index += 1

                # Map Anthropic stop reasons to standard finish reasons
                if finish_reason == "end_turn":
                    finish_reason = "stop"
                elif finish_reason == "max_tokens":
                    finish_reason = "length"
                elif finish_reason == "tool_use":
                    finish_reason = "tool_calls"

                return LLMResponse(
                    content=content,
                    tool_calls=tool_calls,
                    finish_reason=finish_reason,
                    native_finish_reason=native_finish_reason
                )
        except Exception as e:
            logger.error(f"Error during tool call: {e}")
            raise e
