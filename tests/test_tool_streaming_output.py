"""Focused regressions for real-time tool-call streaming output."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from spoon_ai.chat import ChatBot
from spoon_ai.schema import Message
from spoon_ai.llm.interface import LLMResponse
from spoon_ai.llm.providers.anthropic_provider import AnthropicProvider
from spoon_ai.llm.providers.gemini_provider import GeminiProvider
from spoon_ai.llm.providers.ollama_provider import OllamaProvider
from spoon_ai.llm.providers.openai_compatible_provider import OpenAICompatibleProvider


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

    chunks: list[str] = []
    while not q.empty():
        chunks.append((await q.get())["content"])

    assert chunks == ["Hel", "lo"]
    assert response.content == "Hello"
    assert response.metadata.get("streamed_content") is True


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

    deltas: list[str] = []
    while not q.empty():
        deltas.append((await q.get())["content"])

    assert deltas == ["Hel", "lo"]
    assert response.content == "Hello"
    assert response.finish_reason == "stop"
    assert response.native_finish_reason == "end_turn"
    assert response.metadata.get("streamed_content") is True


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
