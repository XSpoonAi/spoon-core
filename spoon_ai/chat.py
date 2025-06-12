import os
from logging import getLogger
from typing import List, Optional, Union
import json

from spoon_ai.schema import Message, LLMResponse, ToolCall

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
        # Use parameters provided by user first
        self.llm_provider = llm_provider
        self.model_name = model_name
        self.api_key = api_key
        self.llm_config = llm_config
        self.output_index = 0
        self.base_url = base_url

        # If llm_provider is not specified, determine it from environment variables
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

        if self.llm_provider == "openai":
            self.llm = AsyncOpenAI(api_key=api_key if api_key else os.getenv("OPENAI_API_KEY"), base_url=base_url if base_url else os.getenv("BASE_URL"))
        elif self.llm_provider == "anthropic":
            http_client = AsyncClient(follow_redirects=True)
            self.llm = AsyncAnthropic(api_key=api_key if api_key else os.getenv("ANTHROPIC_API_KEY"), http_client=http_client)
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
        
        if self.llm_provider == "openai":
            response = await self.llm.chat.completions.create(messages=formatted_messages, model=self.model_name, max_tokens=4096, temperature=0.3, stream=False)
            return response.choices[0].message.content
        elif self.llm_provider == "anthropic":
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
            if self.llm_provider == "openai":
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
            elif self.llm_provider == "anthropic":
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
                        if chunk.type in ["message_start", "text", "input_json", "message_delta", "message_stop"]:
                            continue
                        if chunk.type == "content_block_start":
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
                        if chunk.type == "content_block_delta" and chunk.delta.type == "text_delta":
                            buffer += chunk.delta.text
                            if output_queue:
                                output_queue.put_nowait({"type": "text_delta", "delta": chunk.delta.text, "index": self.output_index})
                            continue
                        if chunk.type == "content_block_delta" and chunk.delta.type == "input_json_delta":
                            buffer += chunk.delta.partial_json
                            if output_queue:
                                output_queue.put_nowait({"type": "input_json_delta", "delta": chunk.delta.partial_json, "index": self.output_index})

                        if chunk.type == "content_block_stop":
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
                return LLMResponse(content=content, tool_calls=tool_calls)
        except Exception as e:
            logger.error(f"Error during tool call: {e}")
            raise e
