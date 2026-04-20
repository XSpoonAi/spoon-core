"""Focused regressions for real-time tool-call streaming output."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from spoon_ai.chat import ChatBot
from spoon_ai.schema import Message
from spoon_ai.llm.interface import LLMResponse
from spoon_ai.llm.providers.anthropic_provider import AnthropicProvider
from spoon_ai.llm.providers.gemini_provider import GeminiProvider
from spoon_ai.llm.providers.ollama_provider import OllamaProvider
from spoon_ai.llm.providers.openai_compatible_provider import OpenAICompatibleProvider
from spoon_ai.llm.providers.openai_provider import OpenAIProvider
from spoon_ai.llm.providers.openrouter_provider import OpenRouterProvider


class _AsyncItems:
    def __init__(self, items):
        self._items = list(items)

    def __aiter__(self):
        self._iter = iter(self._items)
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


class _DelayedAsyncItems:
    def __init__(self, items, delay: float):
        self._items = list(items)
        self._delay = delay

    def __aiter__(self):
        self._iter = iter(self._items)
        return self

    async def __anext__(self):
        await asyncio.sleep(self._delay)
        try:
            return next(self._iter)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


class _AsyncStreamContext:
    def __init__(self, items):
        self._items = list(items)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    def __aiter__(self):
        self._iter = iter(self._items)
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


def _tool_spec() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "echo_tool",
                "description": "Echo text",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            },
        }
    ]


@pytest.mark.asyncio
async def test_chatbot_ask_tool_forwards_output_queue():
    mock_manager = SimpleNamespace(chat_with_tools=AsyncMock())
    mock_manager.chat_with_tools.return_value = LLMResponse(
        content="ok",
        provider="openai",
        model="gpt-4.1",
        finish_reason="stop",
        native_finish_reason="stop",
    )

    with patch("spoon_ai.chat.get_llm_manager", return_value=mock_manager):
        bot = ChatBot(use_llm_manager=True, llm_provider="openai")
        q: asyncio.Queue = asyncio.Queue()
        await bot.ask_tool([{"role": "user", "content": "hi"}], tools=_tool_spec(), output_queue=q)

    call = mock_manager.chat_with_tools.call_args
    assert call.kwargs["output_queue"] is q


@pytest.mark.asyncio
async def test_chatbot_ask_tool_strips_disabled_thinking_flag():
    mock_manager = SimpleNamespace(chat_with_tools=AsyncMock())
    mock_manager.chat_with_tools.return_value = LLMResponse(
        content="ok",
        provider="anthropic",
        model="claude-sonnet-4-20250514",
        finish_reason="stop",
        native_finish_reason="stop",
    )

    with patch("spoon_ai.chat.get_llm_manager", return_value=mock_manager):
        bot = ChatBot(use_llm_manager=True, llm_provider="anthropic")
        await bot.ask_tool(
            [{"role": "user", "content": "hi"}],
            tools=_tool_spec(),
            thinking=False,
        )

    call = mock_manager.chat_with_tools.call_args
    assert "thinking" not in call.kwargs


@pytest.mark.asyncio
async def test_chatbot_ask_tool_normalizes_anthropic_boolean_thinking():
    mock_manager = SimpleNamespace(chat_with_tools=AsyncMock())
    mock_manager.chat_with_tools.return_value = LLMResponse(
        content="ok",
        provider="anthropic",
        model="claude-sonnet-4-20250514",
        finish_reason="stop",
        native_finish_reason="stop",
    )

    with patch("spoon_ai.chat.get_llm_manager", return_value=mock_manager):
        bot = ChatBot(use_llm_manager=True, llm_provider="anthropic")
        await bot.ask_tool(
            [{"role": "user", "content": "hi"}],
            tools=_tool_spec(),
            thinking=True,
        )

    call = mock_manager.chat_with_tools.call_args
    assert call.kwargs["thinking"] == {
        "type": "enabled",
        "budget_tokens": 1024,
    }


@pytest.mark.asyncio
async def test_chatbot_ask_tool_normalizes_anthropic_reasoning_effort():
    mock_manager = SimpleNamespace(chat_with_tools=AsyncMock())
    mock_manager.chat_with_tools.return_value = LLMResponse(
        content="ok",
        provider="anthropic",
        model="claude-sonnet-4.6",
        finish_reason="stop",
        native_finish_reason="stop",
    )

    with patch("spoon_ai.chat.get_llm_manager", return_value=mock_manager):
        bot = ChatBot(
            use_llm_manager=True,
            llm_provider="anthropic",
            model_name="claude-sonnet-4.6",
        )
        await bot.ask_tool(
            [{"role": "user", "content": "hi"}],
            tools=_tool_spec(),
            thinking=True,
            reasoning_effort="high",
        )

    call = mock_manager.chat_with_tools.call_args
    assert call.kwargs["thinking"] == {"type": "adaptive"}
    assert call.kwargs["output_config"] == {"effort": "high"}


@pytest.mark.asyncio
async def test_chatbot_ask_tool_enables_anthropic_adaptive_thinking_from_reasoning_effort_alone():
    mock_manager = SimpleNamespace(chat_with_tools=AsyncMock())
    mock_manager.chat_with_tools.return_value = LLMResponse(
        content="ok",
        provider="anthropic",
        model="claude-sonnet-4.6",
        finish_reason="stop",
        native_finish_reason="stop",
    )

    with patch("spoon_ai.chat.get_llm_manager", return_value=mock_manager):
        bot = ChatBot(
            use_llm_manager=True,
            llm_provider="anthropic",
            model_name="claude-sonnet-4.6",
        )
        await bot.ask_tool(
            [{"role": "user", "content": "hi"}],
            tools=_tool_spec(),
            reasoning_effort="high",
        )

    call = mock_manager.chat_with_tools.call_args
    assert call.kwargs["thinking"] == {"type": "adaptive"}
    assert call.kwargs["output_config"] == {"effort": "high"}


@pytest.mark.asyncio
async def test_chatbot_ask_tool_maps_anthropic_xhigh_to_max_for_opus_46():
    mock_manager = SimpleNamespace(chat_with_tools=AsyncMock())
    mock_manager.chat_with_tools.return_value = LLMResponse(
        content="ok",
        provider="anthropic",
        model="claude-opus-4.6",
        finish_reason="stop",
        native_finish_reason="stop",
    )

    with patch("spoon_ai.chat.get_llm_manager", return_value=mock_manager):
        bot = ChatBot(
            use_llm_manager=True,
            llm_provider="anthropic",
            model_name="anthropic/claude-opus-4.6",
        )
        await bot.ask_tool(
            [{"role": "user", "content": "hi"}],
            tools=_tool_spec(),
            thinking=True,
            reasoning_effort="xhigh",
        )

    call = mock_manager.chat_with_tools.call_args
    assert call.kwargs["thinking"] == {"type": "adaptive"}
    assert call.kwargs["output_config"] == {"effort": "max"}


@pytest.mark.asyncio
async def test_chatbot_ask_tool_normalizes_gemini_boolean_thinking():
    mock_manager = SimpleNamespace(chat_with_tools=AsyncMock())
    mock_manager.chat_with_tools.return_value = LLMResponse(
        content="ok",
        provider="gemini",
        model="gemini-3-flash-preview",
        finish_reason="stop",
        native_finish_reason="stop",
    )

    with patch("spoon_ai.chat.get_llm_manager", return_value=mock_manager):
        bot = ChatBot(use_llm_manager=True, llm_provider="gemini")
        await bot.ask_tool(
            [{"role": "user", "content": "hi"}],
            tools=_tool_spec(),
            thinking=True,
        )

    call = mock_manager.chat_with_tools.call_args
    assert "thinking" not in call.kwargs
    assert call.kwargs["thinking_budget"] == 32


@pytest.mark.asyncio
async def test_chatbot_ask_tool_normalizes_gemini_reasoning_effort():
    mock_manager = SimpleNamespace(chat_with_tools=AsyncMock())
    mock_manager.chat_with_tools.return_value = LLMResponse(
        content="ok",
        provider="gemini",
        model="gemini-3-flash-preview",
        finish_reason="stop",
        native_finish_reason="stop",
    )

    with patch("spoon_ai.chat.get_llm_manager", return_value=mock_manager):
        bot = ChatBot(
            use_llm_manager=True,
            llm_provider="gemini",
            model_name="gemini-3-flash-preview",
        )
        await bot.ask_tool(
            [{"role": "user", "content": "hi"}],
            tools=_tool_spec(),
            thinking=True,
            reasoning_effort="high",
        )

    call = mock_manager.chat_with_tools.call_args
    assert "thinking" not in call.kwargs
    assert call.kwargs["thinking_config"] == {"thinking_level": "high"}


@pytest.mark.asyncio
async def test_chatbot_ask_tool_passes_openai_reasoning_effort_through():
    mock_manager = SimpleNamespace(chat_with_tools=AsyncMock())
    mock_manager.chat_with_tools.return_value = LLMResponse(
        content="ok",
        provider="openai",
        model="gpt-5.2",
        finish_reason="stop",
        native_finish_reason="stop",
    )

    with patch("spoon_ai.chat.get_llm_manager", return_value=mock_manager):
        bot = ChatBot(
            use_llm_manager=True,
            llm_provider="openai",
            model_name="gpt-5.2",
        )
        await bot.ask_tool(
            [{"role": "user", "content": "hi"}],
            tools=_tool_spec(),
            reasoning_effort="high",
        )

    call = mock_manager.chat_with_tools.call_args
    assert call.kwargs["reasoning_effort"] == "high"


@pytest.mark.asyncio
async def test_chatbot_astream_normalizes_anthropic_reasoning_effort():
    mock_manager = SimpleNamespace(chat_stream=MagicMock(return_value=_AsyncItems([])))

    with patch("spoon_ai.chat.get_llm_manager", return_value=mock_manager):
        bot = ChatBot(
            use_llm_manager=True,
            llm_provider="anthropic",
            model_name="claude-sonnet-4.6",
        )
        chunks = [chunk async for chunk in bot.astream(
            [{"role": "user", "content": "hi"}],
            thinking=True,
            reasoning_effort="high",
        )]

    assert chunks == []
    call = mock_manager.chat_stream.call_args
    assert call.kwargs["thinking"] == {"type": "adaptive"}
    assert call.kwargs["output_config"] == {"effort": "high"}


@pytest.mark.asyncio
async def test_openai_chat_with_tools_streams_deltas_to_output_queue():
    provider = OpenAICompatibleProvider()
    provider.model = "gpt-4.1"

    stream_items = [
        SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="Hel", tool_calls=None), finish_reason=None)],
            usage=None,
        ),
        SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="lo", tool_calls=None), finish_reason="stop")],
            usage=None,
        ),
    ]
    provider.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=AsyncMock(return_value=_AsyncItems(stream_items))),
        )
    )

    q: asyncio.Queue = asyncio.Queue()
    response = await provider.chat_with_tools(
        messages=[Message(role="user", content="hi")],
        tools=_tool_spec(),
        output_queue=q,
    )

    chunks: list[dict] = []
    while not q.empty():
        chunks.append(await q.get())

    assert [chunk["content"] for chunk in chunks] == ["Hel", "lo"]
    assert all(chunk["type"] == "content" for chunk in chunks)
    assert all(chunk["metadata"]["channel"] == "text" for chunk in chunks)
    assert all("phase" not in chunk["metadata"] for chunk in chunks)
    assert response.content == "Hello"
    assert response.metadata.get("streamed_content") is True


def test_openai_supports_temperature_for_gpt_54_only_without_reasoning():
    provider = OpenAICompatibleProvider()

    assert provider._supports_temperature("gpt-5.4", reasoning_effort="none") is True
    assert provider._supports_temperature("gpt-5.4", reasoning_effort=None) is True
    assert provider._supports_temperature("gpt-5.4", reasoning_effort="high") is False
    assert provider._supports_temperature("gpt-5.4", reasoning_effort="medium") is False


@pytest.mark.asyncio
async def test_openai_chat_with_tools_uses_responses_reasoning_summary_when_effort_requested():
    provider = OpenAIProvider()
    provider.model = "gpt-5.4"

    completed_response = SimpleNamespace(
        id="resp_123",
        created_at=123.0,
        model="gpt-5.4",
        output=[
            SimpleNamespace(
                type="reasoning",
                summary=[
                    SimpleNamespace(
                        text="Plan: inspect the latest game state before choosing a move."
                    )
                ],
            ),
            SimpleNamespace(
                type="function_call",
                id="fc_123",
                call_id="call_123",
                name="echo_tool",
                arguments='{"text":"hello"}',
            ),
        ],
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=7,
            total_tokens=17,
        ),
    )
    stream_items = [
        SimpleNamespace(
            type="response.reasoning_summary_text.delta",
            delta="Plan: inspect the latest game state before choosing a move.",
            output_index=0,
            summary_index=0,
            item_id="rs_123",
        ),
        SimpleNamespace(
            type="response.output_item.done",
            output_index=1,
            item=SimpleNamespace(
                type="function_call",
                id="fc_123",
                call_id="call_123",
                name="echo_tool",
                arguments='{"text":"hello"}',
            ),
        ),
        SimpleNamespace(
            type="response.completed",
            response=completed_response,
        ),
    ]
    provider.client = SimpleNamespace(
        responses=SimpleNamespace(create=AsyncMock(return_value=_AsyncItems(stream_items))),
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=AsyncMock()),
        ),
    )

    q: asyncio.Queue = asyncio.Queue()
    response = await provider.chat_with_tools(
        messages=[Message(role="user", content="hi")],
        tools=_tool_spec(),
        output_queue=q,
        reasoning_effort="high",
    )

    streamed_events: list[dict] = []
    while not q.empty():
        streamed_events.append(await q.get())

    assert streamed_events == [
        {
            "type": "thinking",
            "delta": "Plan: inspect the latest game state before choosing a move.",
            "content": "Plan: inspect the latest game state before choosing a move.",
            "metadata": {
                "phase": "think",
                "provider": "openai",
                "channel": "thinking",
            },
        }
    ]
    assert response.content == ""
    assert response.tool_calls[0].id == "call_123"
    assert response.tool_calls[0].function.name == "echo_tool"
    assert response.tool_calls[0].function.arguments == '{"text":"hello"}'
    assert response.metadata["reasoning"] == (
        "Plan: inspect the latest game state before choosing a move."
    )
    assert provider.client.responses.create.await_count == 1
    assert provider.client.chat.completions.create.await_count == 0
    request_kwargs = provider.client.responses.create.await_args.kwargs
    assert request_kwargs["reasoning"] == {"effort": "high", "summary": "detailed"}
    assert "temperature" not in request_kwargs
    assert request_kwargs["tools"] == [
        {
            "type": "function",
            "name": "echo_tool",
            "description": "Echo text",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        }
    ]


@pytest.mark.asyncio
async def test_openai_chat_stream_uses_responses_reasoning_summary_when_effort_requested():
    provider = OpenAIProvider()
    provider.model = "gpt-5.4"

    completed_response = SimpleNamespace(
        id="resp_stream_123",
        created_at=456.0,
        model="gpt-5.4",
        output=[
            SimpleNamespace(
                type="reasoning",
                summary=[
                    SimpleNamespace(
                        text="Plan: inspect the wallet before attempting to join."
                    )
                ],
            ),
            SimpleNamespace(
                type="function_call",
                id="fc_stream_123",
                call_id="call_stream_123",
                name="echo_tool",
                arguments='{"text":"hello"}',
            ),
        ],
        usage=SimpleNamespace(
            input_tokens=12,
            output_tokens=8,
            total_tokens=20,
        ),
    )
    stream_items = [
        SimpleNamespace(
            type="response.reasoning_summary_text.delta",
            delta="Plan: inspect the wallet before attempting to join.",
            output_index=0,
            summary_index=0,
            item_id="rs_stream_123",
        ),
        SimpleNamespace(
            type="response.output_text.delta",
            delta="Wallet looks ready.",
        ),
        SimpleNamespace(
            type="response.output_item.done",
            output_index=1,
            item=SimpleNamespace(
                type="function_call",
                id="fc_stream_123",
                call_id="call_stream_123",
                name="echo_tool",
                arguments='{"text":"hello"}',
            ),
        ),
        SimpleNamespace(
            type="response.completed",
            response=completed_response,
        ),
    ]
    provider.client = SimpleNamespace(
        responses=SimpleNamespace(create=AsyncMock(return_value=_AsyncItems(stream_items))),
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=AsyncMock()),
        ),
    )

    chunks = [
        chunk
        async for chunk in provider.chat_stream(
            messages=[Message(role="user", content="hi")],
            tools=_tool_spec(),
            reasoning_effort="high",
        )
    ]

    assert chunks[0].delta == "Plan: inspect the wallet before attempting to join."
    assert chunks[0].metadata == {
        "type": "thinking",
        "phase": "think",
        "provider": "openai",
        "channel": "thinking",
    }
    assert chunks[1].delta == "Wallet looks ready."
    assert chunks[1].metadata == {
        "provider": "openai",
        "channel": "text",
    }
    assert chunks[2].tool_calls[0].id == "call_stream_123"
    assert chunks[2].tool_calls[0].function.name == "echo_tool"
    assert chunks[2].tool_call_chunks == [
        {
            "index": 1,
            "id": "call_stream_123",
            "type": "function",
            "function": {
                "name": "echo_tool",
                "arguments": '{"text":"hello"}',
            },
        }
    ]
    assert chunks[-1].finish_reason == "tool_calls"
    assert chunks[-1].metadata["reasoning"] == (
        "Plan: inspect the wallet before attempting to join."
    )
    assert provider.client.responses.create.await_count == 1
    assert provider.client.chat.completions.create.await_count == 0
    request_kwargs = provider.client.responses.create.await_args.kwargs
    assert request_kwargs["reasoning"] == {"effort": "high", "summary": "detailed"}
    assert "temperature" not in request_kwargs
    assert request_kwargs["tools"] == [
        {
            "type": "function",
            "name": "echo_tool",
            "description": "Echo text",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        }
    ]


@pytest.mark.asyncio
async def test_openrouter_chat_with_tools_streams_reasoning_to_output_queue():
    provider = OpenRouterProvider()
    provider.model = "google/gemini-3-flash-preview"

    create_mock = AsyncMock()
    stream_items = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content="",
                        reasoning="Plan: inspect files first.",
                        reasoning_content=None,
                        reasoning_details=[
                            SimpleNamespace(
                                type="reasoning.text",
                                text="Plan: inspect files first.",
                            )
                        ],
                        tool_calls=None,
                    ),
                    finish_reason=None,
                )
            ],
            usage=None,
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content="Done.",
                        reasoning=None,
                        reasoning_content=None,
                        reasoning_details=None,
                        tool_calls=None,
                    ),
                    finish_reason="stop",
                )
            ],
            usage=None,
        ),
    ]
    create_mock.return_value = _AsyncItems(stream_items)
    provider.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create_mock),
        )
    )

    q: asyncio.Queue = asyncio.Queue()
    response = await provider.chat_with_tools(
        messages=[Message(role="user", content="hi")],
        tools=_tool_spec(),
        output_queue=q,
        thinking=True,
    )

    streamed_events: list[dict] = []
    while not q.empty():
        streamed_events.append(await q.get())

    assert streamed_events == [
        {
            "type": "thinking",
            "delta": "Plan: inspect files first.",
            "content": "Plan: inspect files first.",
            "metadata": {
                "phase": "think",
                "provider": "openrouter",
                "channel": "thinking",
            },
        },
        {
            "type": "content",
            "delta": "Done.",
            "content": "Done.",
            "metadata": {
                "provider": "openrouter",
                "channel": "text",
            },
        },
    ]
    assert create_mock.call_args.kwargs["extra_body"]["reasoning"] == {"effort": "low"}


@pytest.mark.asyncio
async def test_openrouter_chat_with_tools_maps_reasoning_effort_to_extra_body():
    provider = OpenRouterProvider()
    provider.model = "openai/gpt-5"
    provider.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=AsyncMock(
                    return_value=SimpleNamespace(
                        id="resp_123",
                        choices=[
                            SimpleNamespace(
                                message=SimpleNamespace(content="done", tool_calls=None),
                                finish_reason="stop",
                            )
                        ],
                        created=123,
                        usage=None,
                        model="openai/gpt-5",
                    )
                )
            )
        )
    )

    await provider.chat_with_tools(
        messages=[Message(role="user", content="hi")],
        tools=_tool_spec(),
        reasoning_effort="high",
    )

    assert provider.client.chat.completions.create.call_args.kwargs["extra_body"]["reasoning"] == {
        "effort": "high"
    }
    assert "reasoning_effort" not in provider.client.chat.completions.create.call_args.kwargs


@pytest.mark.asyncio
async def test_openai_compatible_chat_with_tools_drops_top_level_reasoning_effort_for_non_openrouter_requests():
    provider = OpenAICompatibleProvider()
    provider.model = "gpt-4.1"
    provider.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=AsyncMock(
                    return_value=SimpleNamespace(
                        id="resp_456",
                        choices=[
                            SimpleNamespace(
                                message=SimpleNamespace(content="done", tool_calls=None),
                                finish_reason="stop",
                            )
                        ],
                        created=456,
                        usage=None,
                        model="gpt-4.1",
                    )
                )
            )
        )
    )

    await provider.chat_with_tools(
        messages=[Message(role="user", content="hi")],
        tools=_tool_spec(),
        reasoning_effort="high",
    )

    request_kwargs = provider.client.chat.completions.create.call_args.kwargs
    assert "reasoning_effort" not in request_kwargs
    assert "extra_body" not in request_kwargs


@pytest.mark.asyncio
async def test_openai_chat_with_tools_merges_streamed_tool_call_fragments_without_repeated_id():
    provider = OpenAICompatibleProvider()
    provider.model = "gpt-4.1"

    stream_items = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content="",
                        tool_calls=[
                            SimpleNamespace(
                                id="call_abc",
                                index=0,
                                type="function",
                                function=SimpleNamespace(name="get_weather", arguments='{"city":"'),
                            )
                        ],
                    ),
                    finish_reason=None,
                )
            ],
            usage=None,
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content="",
                        tool_calls=[
                            SimpleNamespace(
                                id=None,
                                index=0,
                                type=None,
                                function=SimpleNamespace(name=None, arguments='Paris"}'),
                            )
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
            usage=None,
        ),
    ]
    provider.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=AsyncMock(return_value=_AsyncItems(stream_items))),
        )
    )

    q: asyncio.Queue = asyncio.Queue()
    response = await provider.chat_with_tools(
        messages=[Message(role="user", content="weather?")],
        tools=_tool_spec(),
        output_queue=q,
    )

    assert response.finish_reason == "tool_calls"
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].id == "call_abc"
    assert response.tool_calls[0].function.name == "get_weather"
    assert response.tool_calls[0].function.arguments == '{"city":"Paris"}'


@pytest.mark.asyncio
async def test_anthropic_chat_with_tools_streams_deltas_to_output_queue():
    provider = AnthropicProvider()
    provider.model = "claude-sonnet-4-20250514"

    chunks = [
        SimpleNamespace(type="content_block_start", content_block=SimpleNamespace(type="text")),
        SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(type="text_delta", text="Hel")),
        SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(type="text_delta", text="lo")),
        SimpleNamespace(type="content_block_stop"),
        SimpleNamespace(type="message_delta", delta=SimpleNamespace(stop_reason="end_turn")),
        SimpleNamespace(type="message_stop", message=SimpleNamespace(stop_reason="end_turn")),
    ]
    provider.client = SimpleNamespace(
        messages=SimpleNamespace(stream=lambda **_: _AsyncStreamContext(chunks))
    )

    q: asyncio.Queue = asyncio.Queue()
    response = await provider.chat_with_tools(
        messages=[Message(role="user", content="hi")],
        tools=_tool_spec(),
        output_queue=q,
    )

    deltas: list[dict] = []
    while not q.empty():
        deltas.append(await q.get())

    assert [chunk["content"] for chunk in deltas] == ["Hel", "lo"]
    assert all(chunk["type"] == "content" for chunk in deltas)
    assert all(chunk["metadata"]["channel"] == "text" for chunk in deltas)
    assert all("phase" not in chunk["metadata"] for chunk in deltas)
    assert response.content == "Hello"
    assert response.finish_reason == "stop"
    assert response.native_finish_reason == "end_turn"
    assert response.metadata.get("streamed_content") is True


@pytest.mark.asyncio
async def test_anthropic_chat_with_tools_streams_provider_thinking_to_output_queue():
    provider = AnthropicProvider()
    provider.model = "claude-sonnet-4-20250514"

    chunks = [
        SimpleNamespace(type="content_block_start", content_block=SimpleNamespace(type="thinking")),
        SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(type="thinking_delta", thinking="Plan: inspect files.")),
        SimpleNamespace(type="content_block_stop"),
        SimpleNamespace(type="content_block_start", content_block=SimpleNamespace(type="tool_use", id="call_1", name="shell")),
        SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(type="input_json_delta", partial_json='{"command":"pwd"}')),
        SimpleNamespace(type="content_block_stop"),
        SimpleNamespace(type="message_delta", delta=SimpleNamespace(stop_reason="tool_use")),
        SimpleNamespace(type="message_stop", message=SimpleNamespace(stop_reason="tool_use")),
    ]
    provider.client = SimpleNamespace(
        messages=SimpleNamespace(stream=lambda **_: _AsyncStreamContext(chunks))
    )

    q: asyncio.Queue = asyncio.Queue()
    response = await provider.chat_with_tools(
        messages=[Message(role="user", content="hi")],
        tools=_tool_spec(),
        output_queue=q,
    )

    streamed_events: list[dict] = []
    while not q.empty():
        streamed_events.append(await q.get())

    assert streamed_events == [
        {
            "type": "thinking",
            "delta": "Plan: inspect files.",
            "content": "Plan: inspect files.",
            "metadata": {
                "phase": "think",
                "provider": "anthropic",
                "channel": "thinking",
            },
        }
    ]
    assert response.finish_reason == "tool_calls"
    assert response.tool_calls[0].function.arguments == '{"command":"pwd"}'


@pytest.mark.asyncio
async def test_anthropic_chat_with_tools_normalizes_boolean_thinking_before_request():
    provider = AnthropicProvider()
    provider.model = "claude-sonnet-4-20250514"

    captured_kwargs: dict = {}
    chunks = [
        SimpleNamespace(type="content_block_start", content_block=SimpleNamespace(type="text")),
        SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(type="text_delta", text="ok")),
        SimpleNamespace(type="content_block_stop"),
        SimpleNamespace(type="message_delta", delta=SimpleNamespace(stop_reason="end_turn")),
        SimpleNamespace(type="message_stop", message=SimpleNamespace(stop_reason="end_turn")),
    ]

    def _stream(**request_kwargs):
        captured_kwargs.update(request_kwargs)
        return _AsyncStreamContext(chunks)

    provider.client = SimpleNamespace(
        messages=SimpleNamespace(stream=_stream)
    )

    response = await provider.chat_with_tools(
        messages=[Message(role="user", content="hi")],
        tools=_tool_spec(),
        thinking=True,
    )

    assert captured_kwargs["thinking"] == {
        "type": "enabled",
        "budget_tokens": 1024,
    }
    assert response.content == "ok"


@pytest.mark.asyncio
async def test_anthropic_chat_with_tools_strips_temperature_top_k_and_forced_tool_choice_when_thinking_enabled():
    provider = AnthropicProvider()
    provider.model = "claude-opus-4.7"

    captured_kwargs: dict = {}
    chunks = [
        SimpleNamespace(type="content_block_start", content_block=SimpleNamespace(type="text")),
        SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(type="text_delta", text="ok")),
        SimpleNamespace(type="content_block_stop"),
        SimpleNamespace(type="message_delta", delta=SimpleNamespace(stop_reason="end_turn")),
        SimpleNamespace(type="message_stop", message=SimpleNamespace(stop_reason="end_turn")),
    ]

    def _stream(**request_kwargs):
        captured_kwargs.update(request_kwargs)
        return _AsyncStreamContext(chunks)

    provider.client = SimpleNamespace(
        messages=SimpleNamespace(stream=_stream)
    )

    response = await provider.chat_with_tools(
        messages=[Message(role="user", content="hi")],
        tools=_tool_spec(),
        thinking=True,
        temperature=0.7,
        top_k=5,
        tool_choice="required",
    )

    assert captured_kwargs["thinking"] == {"type": "adaptive"}
    assert "temperature" not in captured_kwargs
    assert "top_k" not in captured_kwargs
    assert "tool_choice" not in captured_kwargs
    assert response.content == "ok"


@pytest.mark.asyncio
async def test_gemini_chat_with_tools_streams_deltas_to_output_queue():
    provider = GeminiProvider()
    provider.api_key = "test-key"
    provider.model = "gemini-2.5-pro"

    responses = [
        SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(parts=[SimpleNamespace(text="Hel")]),
                    finish_reason=None,
                )
            ]
        ),
        SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(parts=[SimpleNamespace(text="lo")]),
                    finish_reason="STOP",
                )
            ]
        ),
    ]

    class _FakeGeminiClient:
        def __init__(self, api_key: str):
            self.aio = SimpleNamespace(
                models=SimpleNamespace(
                    generate_content_stream=AsyncMock(return_value=_AsyncItems(responses))
                ),
                aclose=AsyncMock(return_value=None),
            )

        def close(self):
            return None

    with patch("spoon_ai.llm.providers.gemini_provider.genai.Client", _FakeGeminiClient):
        q: asyncio.Queue = asyncio.Queue()
        response = await provider.chat_with_tools(
            messages=[Message(role="user", content="hi")],
            tools=_tool_spec(),
            output_queue=q,
        )

    deltas: list[str] = []
    while not q.empty():
        deltas.append((await q.get())["content"])

    assert deltas == ["Hel", "lo"]
    assert response.content == "Hello"
    assert response.finish_reason == "stop"
    assert response.native_finish_reason == "STOP"
    assert response.metadata.get("streamed_content") is True


@pytest.mark.asyncio
async def test_gemini_chat_with_tools_streams_provider_thinking_and_preserves_tool_call_signature():
    provider = GeminiProvider()
    provider.api_key = "test-key"
    provider.model = "gemini-2.5-pro"

    signature = b"sig-123"
    responses = [
        SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(
                        parts=[
                            SimpleNamespace(text="Plan: inspect files.", thought=True),
                            SimpleNamespace(
                                function_call=SimpleNamespace(
                                    name="get_weather",
                                    args={"city": "Paris"},
                                ),
                                thought_signature=signature,
                            ),
                        ]
                    ),
                    finish_reason="STOP",
                )
            ]
        ),
    ]

    class _FakeGeminiClient:
        captured_config = None

        def __init__(self, api_key: str):
            self.aio = SimpleNamespace(
                models=SimpleNamespace(
                    generate_content_stream=AsyncMock(side_effect=self._generate_content_stream)
                ),
                aclose=AsyncMock(return_value=None),
            )

        async def _generate_content_stream(self, *, config, **kwargs):
            type(self).captured_config = config
            return _AsyncItems(responses)

        def close(self):
            return None

    with patch("spoon_ai.llm.providers.gemini_provider.genai.Client", _FakeGeminiClient):
        q: asyncio.Queue = asyncio.Queue()
        response = await provider.chat_with_tools(
            messages=[Message(role="user", content="hi")],
            tools=_tool_spec(),
            output_queue=q,
            thinking_budget=128,
        )

    streamed_events: list[dict] = []
    while not q.empty():
        streamed_events.append(await q.get())

    assert streamed_events == [
        {
            "type": "thinking",
            "delta": "Plan: inspect files.",
            "content": "Plan: inspect files.",
            "metadata": {
                "phase": "think",
                "provider": "gemini",
                "channel": "thinking",
            },
        }
    ]
    assert _FakeGeminiClient.captured_config.thinking_config.include_thoughts is True
    assert response.finish_reason == "tool_calls"
    assert response.metadata["reasoning"] == "Plan: inspect files."
    assert response.tool_calls[0].function.name == "get_weather"
    assert response.tool_calls[0].metadata == {
        "thought_signature": "c2lnLTEyMw=="
    }


@pytest.mark.asyncio
async def test_gemini_chat_with_tools_emits_first_chunk_before_completion():
    provider = GeminiProvider()
    provider.api_key = "test-key"
    provider.model = "gemini-2.5-pro"

    responses = [
        SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(parts=[SimpleNamespace(text="Hel")]),
                    finish_reason=None,
                )
            ]
        ),
        SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(parts=[SimpleNamespace(text="lo")]),
                    finish_reason="STOP",
                )
            ]
        ),
    ]

    class _FakeGeminiClient:
        def __init__(self, api_key: str):
            self.aio = SimpleNamespace(
                models=SimpleNamespace(
                    generate_content_stream=AsyncMock(
                        return_value=_DelayedAsyncItems(responses, delay=0.05)
                    )
                ),
                aclose=AsyncMock(return_value=None),
            )

        def close(self):
            return None

    with patch("spoon_ai.llm.providers.gemini_provider.genai.Client", _FakeGeminiClient):
        q: asyncio.Queue = asyncio.Queue()
        task = asyncio.create_task(
            provider.chat_with_tools(
                messages=[Message(role="user", content="hi")],
                tools=_tool_spec(),
                output_queue=q,
            )
        )

        first_chunk = await asyncio.wait_for(q.get(), timeout=0.2)
        assert first_chunk["content"] == "Hel"
        assert task.done() is False

        response = await task

    remaining = [first_chunk["content"]]
    while not q.empty():
        remaining.append((await q.get())["content"])

    assert remaining == ["Hel", "lo"]
    assert response.content == "Hello"
    assert response.metadata.get("streamed_content") is True


def test_gemini_convert_messages_for_tools_reuses_thought_signature():
    provider = GeminiProvider()

    system_content, gemini_messages = provider._convert_messages_for_tools(
        [
            Message(
                role="assistant",
                content="Working on it.",
                tool_calls=[
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city":"Paris"}',
                        },
                        "metadata": {
                            "thought_signature": "c2lnLTEyMw==",
                        },
                    }
                ],
            )
        ]
    )

    assert system_content == ""
    assert gemini_messages[0].parts[1].function_call.name == "get_weather"
    assert gemini_messages[0].parts[1].thought_signature == b"sig-123"


@pytest.mark.asyncio
async def test_gemini_chat_stream_yields_incrementally():
    provider = GeminiProvider()
    provider.api_key = "test-key"
    provider.model = "gemini-2.5-pro"

    responses = [
        SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(parts=[SimpleNamespace(text="Hel")]),
                    finish_reason=None,
                )
            ],
            usage_metadata=None,
        ),
        SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(parts=[SimpleNamespace(text="lo")]),
                    finish_reason="STOP",
                )
            ],
            usage_metadata=None,
        ),
    ]

    class _FakeGeminiClient:
        def __init__(self, api_key: str):
            self.aio = SimpleNamespace(
                models=SimpleNamespace(
                    generate_content_stream=AsyncMock(
                        return_value=_DelayedAsyncItems(responses, delay=0.05)
                    )
                ),
                aclose=AsyncMock(return_value=None),
            )

        def close(self):
            return None

    with patch("spoon_ai.llm.providers.gemini_provider.genai.Client", _FakeGeminiClient):
        stream = provider.chat_stream(messages=[Message(role="user", content="hi")])
        first = await asyncio.wait_for(anext(stream), timeout=0.2)
        second = await asyncio.wait_for(anext(stream), timeout=0.2)

    assert first.delta == "Hel"
    assert second.delta == "lo"
    assert second.finish_reason == "stop"


@pytest.mark.asyncio
async def test_gemini_chat_stream_yields_thinking_before_visible_content():
    provider = GeminiProvider()
    provider.api_key = "test-key"
    provider.model = "gemini-2.5-pro"

    responses = [
        SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(
                        parts=[
                            SimpleNamespace(text="Plan: inspect.", thought=True),
                            SimpleNamespace(text="Done."),
                        ]
                    ),
                    finish_reason="STOP",
                )
            ],
            usage_metadata=None,
        ),
    ]

    class _FakeGeminiClient:
        def __init__(self, api_key: str):
            self.aio = SimpleNamespace(
                models=SimpleNamespace(
                    generate_content_stream=AsyncMock(return_value=_AsyncItems(responses))
                ),
                aclose=AsyncMock(return_value=None),
            )

        def close(self):
            return None

    with patch("spoon_ai.llm.providers.gemini_provider.genai.Client", _FakeGeminiClient):
        chunks = [
            chunk
            async for chunk in provider.chat_stream(
                messages=[Message(role="user", content="hi")],
                thinking_budget=128,
            )
        ]

    assert chunks[0].delta == "Plan: inspect."
    assert chunks[0].metadata == {
        "chunk_index": 0,
        "type": "thinking",
        "phase": "think",
        "provider": "gemini",
        "channel": "thinking",
    }
    assert chunks[1].delta == "Done."
    assert chunks[1].metadata == {
        "chunk_index": 1,
        "finish_reason": "stop",
    }


@pytest.mark.asyncio
async def test_gemini_chat_with_tools_keeps_tool_calls_finish_reason_when_native_is_stop():
    provider = GeminiProvider()
    provider.api_key = "test-key"
    provider.model = "gemini-2.5-pro"

    responses = [
        SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(
                        parts=[
                            SimpleNamespace(
                                function_call=SimpleNamespace(
                                    name="get_weather",
                                    args={"city": "Paris"},
                                )
                            )
                        ]
                    ),
                    finish_reason="STOP",
                )
            ]
        ),
    ]

    class _FakeGeminiClient:
        def __init__(self, api_key: str):
            self.aio = SimpleNamespace(
                models=SimpleNamespace(
                    generate_content_stream=AsyncMock(return_value=_AsyncItems(responses))
                ),
                aclose=AsyncMock(return_value=None),
            )

        def close(self):
            return None

    with patch("spoon_ai.llm.providers.gemini_provider.genai.Client", _FakeGeminiClient):
        response = await provider.chat_with_tools(
            messages=[Message(role="user", content="weather?")],
            tools=_tool_spec(),
            output_queue=asyncio.Queue(),
        )

    assert response.finish_reason == "tool_calls"
    assert response.native_finish_reason == "STOP"
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].function.name == "get_weather"


@pytest.mark.asyncio
async def test_ollama_chat_with_tools_streams_deltas_to_output_queue():
    provider = OllamaProvider()
    provider.model = "llama3.2"

    lines = [
        json.dumps({"message": {"content": "Hel"}, "done": False}),
        json.dumps(
            {
                "message": {"content": "lo"},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 1,
                "eval_count": 2,
            }
        ),
    ]

    class _FakeResponse:
        def raise_for_status(self):
            return None

        async def aiter_lines(self):
            for line in lines:
                yield line

    class _FakeStreamContext:
        async def __aenter__(self):
            return _FakeResponse()

        async def __aexit__(self, exc_type, exc, tb):
            return None

    provider.client = SimpleNamespace(stream=lambda *_, **__: _FakeStreamContext())

    q: asyncio.Queue = asyncio.Queue()
    response = await provider.chat_with_tools(
        messages=[Message(role="user", content="hi")],
        tools=[],
        output_queue=q,
    )

    deltas: list[str] = []
    while not q.empty():
        deltas.append((await q.get())["content"])

    assert deltas == ["Hel", "lo"]
    assert response.content == "Hello"
    assert response.metadata.get("streamed_content") is True
