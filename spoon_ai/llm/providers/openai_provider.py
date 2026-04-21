"""
OpenAI Provider implementation for the unified LLM interface.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from logging import getLogger
from typing import Any, AsyncIterator, Dict, List, Optional
from uuid import uuid4

from spoon_ai.callbacks.base import BaseCallbackHandler
from spoon_ai.callbacks.manager import CallbackManager
from spoon_ai.schema import LLMResponseChunk, Message, ToolCall, Function
from spoon_ai.utils.streaming import build_output_queue_event

from ..errors import ProviderError
from ..interface import LLMResponse, ProviderMetadata, ProviderCapability
from ..registry import register_provider
from .openai_compatible_provider import OpenAICompatibleProvider

logger = getLogger(__name__)


@register_provider("openai", [
    ProviderCapability.CHAT,
    ProviderCapability.COMPLETION,
    ProviderCapability.TOOLS,
    ProviderCapability.STREAMING
])
class OpenAIProvider(OpenAICompatibleProvider):
    """OpenAI provider implementation."""

    def __init__(self):
        super().__init__()
        self.provider_name = "openai"
        self.default_base_url = "https://api.openai.com/v1"
        self.default_model = "gpt-4.1"

    @staticmethod
    def _stringify_responses_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        if content is None:
            return ""
        if isinstance(content, (dict, list)):
            try:
                return json.dumps(content, ensure_ascii=False)
            except TypeError:
                return str(content)
        return str(content)

    def _supports_responses_reasoning(
        self,
        messages: List[Message],
        tools: List[Dict[str, Any]] | None,
        kwargs: Dict[str, Any],
    ) -> bool:
        if not (kwargs.get("thinking") or kwargs.get("reasoning_effort")):
            return False
        try:
            openai_messages = self._convert_messages(messages)
            self._convert_messages_to_responses_input(openai_messages)
            if tools:
                self._convert_tools_to_responses(tools)
            self._convert_tool_choice_to_responses(kwargs.get("tool_choice", "auto"))
        except ValueError:
            return False
        return True

    def _convert_messages_to_responses_input(
        self,
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        input_items: List[Dict[str, Any]] = []

        for message in messages:
            role = str(message.get("role") or "")
            content = message.get("content")
            tool_calls = message.get("tool_calls") or []

            if role == "tool":
                tool_call_id = message.get("tool_call_id")
                if not tool_call_id:
                    raise ValueError("Responses API requires tool_call_id for tool outputs")
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": tool_call_id,
                        "output": self._stringify_responses_content(content),
                    }
                )
                continue

            if content is not None:
                if not isinstance(content, str):
                    raise ValueError("Responses reasoning path currently supports text-only messages")
                input_items.append(
                    {
                        "type": "message",
                        "role": role,
                        "content": content,
                    }
                )

            if role == "assistant" and tool_calls:
                for tool_call in tool_calls:
                    function_payload = tool_call.get("function") or {}
                    input_items.append(
                        {
                            "type": "function_call",
                            "call_id": tool_call.get("id"),
                            "name": function_payload.get("name") or "unknown",
                            "arguments": function_payload.get("arguments") or "{}",
                        }
                    )

        return input_items

    @staticmethod
    def _convert_tools_to_responses(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        converted: List[Dict[str, Any]] = []
        for tool in tools:
            if tool.get("type") != "function":
                raise ValueError("Responses reasoning path currently supports function tools only")
            function_payload = tool.get("function") or {}
            converted.append(
                {
                    "type": "function",
                    "name": function_payload.get("name") or "unknown",
                    "description": function_payload.get("description"),
                    "parameters": function_payload.get("parameters") or {"type": "object", "properties": {}},
                }
            )
        return converted

    @staticmethod
    def _convert_tool_choice_to_responses(tool_choice: Any) -> Any:
        if tool_choice in {None, "auto", "required", "none"}:
            return tool_choice or "auto"
        if isinstance(tool_choice, dict):
            if tool_choice.get("type") == "function":
                function_payload = tool_choice.get("function") or {}
                name = function_payload.get("name") or tool_choice.get("name")
                if name:
                    return {"type": "function", "name": name}
            choice_type = tool_choice.get("type")
            if choice_type in {"auto", "required", "none"}:
                return choice_type
        return tool_choice

    def _build_responses_reasoning(self, kwargs: Dict[str, Any]) -> Dict[str, Any] | None:
        if not (kwargs.get("thinking") or kwargs.get("reasoning_effort")):
            return None

        reasoning: Dict[str, Any] = {"summary": "detailed"}
        effort = kwargs.get("reasoning_effort")
        if effort:
            reasoning["effort"] = effort
        return reasoning

    def _build_responses_request_kwargs(
        self,
        messages: List[Message],
        kwargs: Dict[str, Any],
        *,
        tools: List[Dict[str, Any]] | None = None,
        stream: bool,
    ) -> Dict[str, Any]:
        openai_messages = self._convert_messages(messages)
        model = kwargs.get("model", self.model)
        max_tokens = kwargs.get("max_completion_tokens", kwargs.get("max_tokens", self.max_tokens))
        temperature = kwargs.get("temperature", self.temperature)

        request_kwargs: Dict[str, Any] = {
            "model": model,
            "input": self._convert_messages_to_responses_input(openai_messages),
            "stream": stream,
            "max_output_tokens": max_tokens,
        }
        if self._supports_temperature(model, kwargs.get("reasoning_effort")):
            request_kwargs["temperature"] = temperature

        reasoning = self._build_responses_reasoning(kwargs)
        if reasoning:
            request_kwargs["reasoning"] = reasoning

        if tools:
            request_kwargs["tools"] = self._convert_tools_to_responses(tools)
            request_kwargs["tool_choice"] = self._convert_tool_choice_to_responses(
                kwargs.get("tool_choice", "auto")
            )

        extra_keys = {
            "model",
            "max_tokens",
            "max_completion_tokens",
            "temperature",
            "thinking",
            "reasoning_effort",
            "tools",
            "tool_choice",
            "callbacks",
            "output_queue",
        }
        request_kwargs.update({k: v for k, v in kwargs.items() if k not in extra_keys})
        return request_kwargs

    @staticmethod
    def _extract_responses_text(response: Any) -> str:
        direct = getattr(response, "output_text", None)
        if isinstance(direct, str) and direct:
            return direct

        texts: List[str] = []
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) != "message":
                continue
            for content_part in getattr(item, "content", []) or []:
                if getattr(content_part, "type", None) == "output_text":
                    text = getattr(content_part, "text", None)
                    if text:
                        texts.append(str(text))
        return "".join(texts)

    @staticmethod
    def _extract_responses_reasoning(response: Any) -> str:
        parts: List[str] = []
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) != "reasoning":
                continue
            for summary_item in getattr(item, "summary", []) or []:
                text = getattr(summary_item, "text", None)
                if text:
                    parts.append(str(text))
        return "\n\n".join(parts)

    @staticmethod
    def _extract_responses_tool_calls(response: Any) -> List[ToolCall]:
        tool_calls: List[ToolCall] = []
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) != "function_call":
                continue
            tool_calls.append(
                ToolCall(
                    id=getattr(item, "call_id", None) or getattr(item, "id", ""),
                    type="function",
                    function=Function(
                        name=getattr(item, "name", None) or "unknown",
                        arguments=getattr(item, "arguments", None) or "{}",
                    ),
                )
            )
        return tool_calls

    @staticmethod
    def _responses_finish_reason(response: Any, tool_calls: List[ToolCall]) -> str:
        if tool_calls:
            return "tool_calls"
        incomplete = getattr(response, "incomplete_details", None)
        reason = getattr(incomplete, "reason", None)
        if reason == "max_output_tokens":
            return "length"
        if reason == "content_filter":
            return "content_filter"
        return "stop"

    @staticmethod
    def _responses_usage(response: Any) -> Dict[str, int] | None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return None
        prompt_tokens = getattr(usage, "input_tokens", None)
        completion_tokens = getattr(usage, "output_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)
        if prompt_tokens is None and completion_tokens is None and total_tokens is None:
            return None
        return {
            "prompt_tokens": prompt_tokens or 0,
            "completion_tokens": completion_tokens or 0,
            "total_tokens": total_tokens or 0,
        }

    def _convert_responses_response(self, response: Any, duration: float) -> LLMResponse:
        tool_calls = self._extract_responses_tool_calls(response)
        reasoning_text = self._extract_responses_reasoning(response)
        finish_reason = self._responses_finish_reason(response, tool_calls)
        metadata = {
            "response_id": getattr(response, "id", ""),
            "created": getattr(response, "created_at", None),
        }
        if reasoning_text:
            metadata["reasoning"] = reasoning_text

        return LLMResponse(
            content=self._extract_responses_text(response),
            provider=self.get_provider_name(),
            model=getattr(response, "model", self.model),
            finish_reason=finish_reason,
            native_finish_reason=finish_reason,
            tool_calls=tool_calls,
            usage=self._responses_usage(response),
            duration=duration,
            metadata=metadata,
        )

    async def chat(self, messages: List[Message], **kwargs) -> LLMResponse:
        if not self.client or not self._supports_responses_reasoning(messages, None, kwargs):
            return await super().chat(messages, **kwargs)

        start_time = asyncio.get_event_loop().time()
        response = await self.client.responses.create(
            **self._build_responses_request_kwargs(messages, kwargs, stream=False)
        )
        duration = asyncio.get_event_loop().time() - start_time
        return self._convert_responses_response(response, duration)

    def _build_responses_stream_chunk(
        self,
        *,
        content: str,
        delta: str,
        model: str,
        finish_reason: str | None,
        tool_calls: List[ToolCall],
        tool_call_chunks: List[Dict[str, Any]] | None = None,
        usage: Dict[str, int] | None = None,
        metadata: Dict[str, Any] | None = None,
        chunk_index: int = 0,
    ) -> LLMResponseChunk:
        return LLMResponseChunk(
            content=content,
            delta=delta,
            provider=self.get_provider_name(),
            model=model,
            finish_reason=finish_reason,
            tool_calls=tool_calls,
            tool_call_chunks=tool_call_chunks,
            usage=usage,
            metadata=metadata or {},
            chunk_index=chunk_index,
            timestamp=datetime.now().isoformat(),
        )

    async def chat_stream(
        self,
        messages: List[Message],
        callbacks: Optional[List[BaseCallbackHandler]] = None,
        **kwargs,
    ) -> AsyncIterator[LLMResponseChunk]:
        tools = kwargs.get("tools")
        if not self.client or not self._supports_responses_reasoning(messages, tools, kwargs):
            async for chunk in super().chat_stream(messages, callbacks=callbacks, **kwargs):
                yield chunk
            return

        callback_manager = CallbackManager.from_callbacks(callbacks)
        run_id = uuid4()
        model = kwargs.get("model", self.model)
        start_time = asyncio.get_event_loop().time()

        await callback_manager.on_llm_start(
            run_id=run_id,
            messages=messages,
            model=model,
            provider=self.get_provider_name(),
        )

        try:
            request_kwargs = self._build_responses_request_kwargs(
                messages,
                kwargs,
                tools=tools,
                stream=True,
            )
            stream = await self.client.responses.create(**request_kwargs)

            full_content = ""
            full_reasoning = ""
            chunk_index = 0
            tool_calls: List[ToolCall] = []
            latest_response = None

            async for event in stream:
                event_type = getattr(event, "type", None)
                if event_type == "response.reasoning_summary_text.delta":
                    delta = getattr(event, "delta", "") or ""
                    if not delta:
                        continue
                    full_reasoning += delta
                    yield self._build_responses_stream_chunk(
                        content=full_reasoning,
                        delta=delta,
                        model=model,
                        finish_reason=None,
                        tool_calls=tool_calls,
                        metadata={
                            "type": "thinking",
                            "phase": "think",
                            "provider": self.get_provider_name(),
                            "channel": "thinking",
                        },
                        chunk_index=chunk_index,
                    )
                    chunk_index += 1
                    continue

                if event_type == "response.output_text.delta":
                    delta = getattr(event, "delta", "") or ""
                    if not delta:
                        continue
                    full_content += delta
                    response_chunk = self._build_responses_stream_chunk(
                        content=full_content,
                        delta=delta,
                        model=model,
                        finish_reason=None,
                        tool_calls=tool_calls,
                        metadata={
                            "provider": self.get_provider_name(),
                            "channel": "text",
                        },
                        chunk_index=chunk_index,
                    )
                    await callback_manager.on_llm_new_token(
                        token=delta,
                        chunk=response_chunk,
                        run_id=run_id,
                    )
                    yield response_chunk
                    chunk_index += 1
                    continue

                if event_type == "response.output_item.done":
                    item = getattr(event, "item", None)
                    if getattr(item, "type", None) != "function_call":
                        continue
                    tool_calls = [
                        *tool_calls,
                        ToolCall(
                            id=getattr(item, "call_id", None) or getattr(item, "id", ""),
                            type="function",
                            function=Function(
                                name=getattr(item, "name", None) or "unknown",
                                arguments=getattr(item, "arguments", None) or "{}",
                            ),
                        ),
                    ]
                    yield self._build_responses_stream_chunk(
                        content=full_content,
                        delta="",
                        model=model,
                        finish_reason=None,
                        tool_calls=tool_calls,
                        tool_call_chunks=[
                            {
                                "index": getattr(event, "output_index", len(tool_calls) - 1),
                                "id": tool_calls[-1].id,
                                "type": "function",
                                "function": {
                                    "name": tool_calls[-1].function.name,
                                    "arguments": tool_calls[-1].function.arguments,
                                },
                            }
                        ],
                        metadata={
                            "provider": self.get_provider_name(),
                            "channel": "tool",
                        },
                        chunk_index=chunk_index,
                    )
                    chunk_index += 1
                    continue

                if event_type == "response.completed":
                    latest_response = getattr(event, "response", None)

            if latest_response is None:
                raise RuntimeError("OpenAI Responses stream completed without a final response")

            final_content = self._extract_responses_text(latest_response)
            if final_content and len(final_content) >= len(full_content):
                full_content = final_content

            final_tool_calls = self._extract_responses_tool_calls(latest_response) or tool_calls
            finish_reason = self._responses_finish_reason(latest_response, final_tool_calls)
            usage = self._responses_usage(latest_response)
            final_metadata: Dict[str, Any] = {
                "response_id": getattr(latest_response, "id", ""),
                "created": getattr(latest_response, "created_at", None),
            }
            if full_reasoning:
                final_metadata["reasoning"] = full_reasoning

            final_chunk = self._build_responses_stream_chunk(
                content=full_content,
                delta="",
                model=getattr(latest_response, "model", model),
                finish_reason=finish_reason,
                tool_calls=final_tool_calls,
                usage=usage,
                metadata=final_metadata,
                chunk_index=chunk_index,
            )
            yield final_chunk

            duration = asyncio.get_event_loop().time() - start_time
            await callback_manager.on_llm_end(
                response=LLMResponse(
                    content=full_content,
                    provider=self.get_provider_name(),
                    model=getattr(latest_response, "model", model),
                    finish_reason=finish_reason,
                    native_finish_reason=finish_reason,
                    tool_calls=final_tool_calls,
                    usage=usage,
                    duration=duration,
                    metadata=final_metadata,
                ),
                run_id=run_id,
            )
        except Exception as e:
            await callback_manager.on_llm_error(
                error=e,
                run_id=run_id,
            )
            await self._handle_error(e)

    async def chat_with_tools(self, messages: List[Message], tools: List[Dict], **kwargs) -> LLMResponse:
        if not self.client or not self._supports_responses_reasoning(messages, tools, kwargs):
            return await super().chat_with_tools(messages, tools, **kwargs)

        start_time = asyncio.get_event_loop().time()
        output_queue = kwargs.get("output_queue")
        request_kwargs = self._build_responses_request_kwargs(
            messages,
            kwargs,
            tools=tools,
            stream=output_queue is not None,
        )

        if output_queue is None:
            response = await self.client.responses.create(**request_kwargs)
            duration = asyncio.get_event_loop().time() - start_time
            return self._convert_responses_response(response, duration)

        stream = await self.client.responses.create(**request_kwargs)
        latest_response = None
        full_reasoning = ""

        async for event in stream:
            event_type = getattr(event, "type", None)
            if event_type == "response.reasoning_summary_text.delta":
                delta = getattr(event, "delta", "") or ""
                if delta:
                    full_reasoning += delta
                    try:
                        output_queue.put_nowait(
                            build_output_queue_event(
                                event_type="thinking",
                                delta=delta,
                                metadata={
                                    "phase": "think",
                                    "provider": self.get_provider_name(),
                                    "channel": "thinking",
                                },
                            )
                        )
                    except Exception:
                        pass
            elif event_type == "response.output_text.delta":
                delta = getattr(event, "delta", "") or ""
                if delta:
                    try:
                        output_queue.put_nowait(
                            build_output_queue_event(
                                event_type="content",
                                delta=delta,
                                metadata={
                                    "provider": self.get_provider_name(),
                                    "channel": "text",
                                },
                            )
                        )
                    except Exception:
                        pass
            elif event_type == "response.completed":
                latest_response = getattr(event, "response", None)

        if latest_response is None:
            raise RuntimeError("OpenAI Responses stream completed without a final response")

        duration = asyncio.get_event_loop().time() - start_time
        result = self._convert_responses_response(latest_response, duration)
        if full_reasoning:
            result.metadata["reasoning"] = full_reasoning
        result.metadata["streamed_content"] = bool(result.content)
        result.metadata["stream_chunk_count"] = 0
        return result

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
