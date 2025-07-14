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
    # def __init__(self, model_name: str = "gpt-4.5-preview", llm_config: dict = None, llm_provider: str = "openai", api_key: str = None):
    def __init__(self, model_name: str = None, llm_config: dict = None, llm_provider: str = None, api_key: str = None, base_url: str = None):
        # Initialize configuration manager
        config_manager = ConfigManager()

        # Use parameters provided by user first, then fall back to config, then environment
        self.model_name = model_name or config_manager.get_model_name()
        self.llm_provider = llm_provider or config_manager.get_llm_provider()
        self.base_url = base_url or config_manager.get_base_url() or os.getenv("BASE_URL")
        self.api_key = api_key
        self.llm_config = llm_config
        self.output_index = 0

        # If llm_provider is still not specified, determine it from environment variables
        if self.llm_provider is None:
            if os.getenv("OPENAI_API_KEY"):
                logger.info("Using OpenAI API")
                self.model_name = self.model_name or "gpt-4.1"
                self.llm_provider = "openai"
            elif os.getenv("ANTHROPIC_API_KEY"):
                logger.info("Using Anthropic API")
                self.model_name = self.model_name or "claude-3-7-sonnet-20250219"
                self.llm_provider = "anthropic"
            else:
                raise ValueError("No API key provided, please set OPENAI_API_KEY or ANTHROPIC_API_KEY")

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
                self.model_name = "claude-3-7-sonnet-20250219"

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
            response = await self.llm.messages.create(
                model=self.model_name,
                max_tokens=4096,
                temperature=0.3,
                system=system_msg,
                messages=[m for m in formatted_messages if m.get("role") != "system"]
            )
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
                return response.choices[0].message
            elif self.api_logic == "anthropic":
                def to_anthropic_tools(tools: List[dict]) -> List[dict]:
                    return [{"name": tool["function"]["name"], "description": tool["function"]["description"], "input_schema": tool["function"]["parameters"]} for tool in tools]

                # 转换消息格式为 Anthropic 格式
                anthropic_messages = []
                system_content = system_msg or ""

                for message in formatted_messages:
                    role = message.get("role")

                    # Anthropic 只支持 user 和 assistant 角色
                    if role == "system":
                        # 系统消息已在上面处理，这里跳过
                        continue
                    elif role == "tool":
                        # 工具消息转换为用户消息，内容包含tool_result
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
