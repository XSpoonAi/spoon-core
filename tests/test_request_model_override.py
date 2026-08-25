import inspect
from unittest.mock import AsyncMock, Mock, patch

import pytest

from spoon_ai.agents.spoon_react import SpoonReactAI
from spoon_ai.agents.spoon_react_skill import SpoonReactSkill
from spoon_ai.agents.toolcall import ToolCallAgent
from spoon_ai.chat import ChatBot
from spoon_ai.llm.interface import LLMResponse as ManagerLLMResponse
from spoon_ai.llm.manager import LLMManager
from spoon_ai.schema import LLMResponse
from spoon_ai.tools import ToolManager


def _manager_response(model: str) -> ManagerLLMResponse:
    return ManagerLLMResponse(
        content="ok",
        provider="openrouter",
        model=model,
        finish_reason="stop",
        native_finish_reason="stop",
    )


def test_spoon_react_run_contract_accepts_request_model() -> None:
    assert "model" in inspect.signature(SpoonReactAI.run).parameters
    assert "model" in inspect.signature(SpoonReactSkill.run).parameters


@pytest.fixture
def chatbot_and_manager():
    manager = Mock(spec=LLMManager)
    manager.chat = AsyncMock(return_value=_manager_response("startup-model"))
    manager.chat_with_tools = AsyncMock(return_value=_manager_response("startup-model"))
    with patch("spoon_ai.chat.get_llm_manager", return_value=manager):
        chatbot = ChatBot(use_llm_manager=True, llm_provider="openrouter")
    chatbot.model_name = "startup-model"
    return chatbot, manager


@pytest.mark.asyncio
async def test_chatbot_uses_default_and_request_models(chatbot_and_manager):
    chatbot, manager = chatbot_and_manager

    await chatbot.ask([{"role": "user", "content": "default"}])
    assert manager.chat.await_args.kwargs["model"] == "startup-model"

    await chatbot.ask(
        [{"role": "user", "content": "override"}],
        model="request-model",
    )
    assert manager.chat.await_args.kwargs["model"] == "request-model"

    await chatbot.ask_tool(
        [{"role": "user", "content": "tools"}],
        tools=[],
        model="tool-model",
    )
    assert manager.chat_with_tools.await_args.kwargs["model"] == "tool-model"


@pytest.mark.asyncio
async def test_chatbot_stream_uses_request_model(chatbot_and_manager):
    chatbot, manager = chatbot_and_manager
    captured: dict = {}

    async def chat_stream(**kwargs):
        captured.update(kwargs)
        if False:
            yield None

    manager.chat_stream = chat_stream

    chunks = [
        chunk
        async for chunk in chatbot.astream(
            [{"role": "user", "content": "stream"}],
            model="stream-model",
        )
    ]

    assert chunks == []
    assert captured["model"] == "stream-model"


@pytest.mark.asyncio
async def test_toolcall_agent_forwards_request_model():
    chatbot = Mock(spec=ChatBot)
    chatbot.ask = AsyncMock()
    chatbot.ask_tool = AsyncMock(
        return_value=LLMResponse(
            content="done",
            tool_calls=[],
            finish_reason="stop",
            native_finish_reason="stop",
        )
    )
    agent = ToolCallAgent(
        name="model-test",
        llm=chatbot,
        available_tools=ToolManager([]),
        max_steps=1,
    )

    assert await agent.run("work", model="request-model") == "done"
    assert chatbot.ask_tool.await_args.kwargs["model"] == "request-model"


@pytest.mark.asyncio
async def test_toolcall_middleware_preserves_request_model():
    chatbot = Mock(spec=ChatBot)
    chatbot.ask = AsyncMock()
    chatbot.ask_tool = AsyncMock(
        return_value=LLMResponse(
            content="done",
            tool_calls=[],
            finish_reason="stop",
            native_finish_reason="stop",
        )
    )
    agent = ToolCallAgent(
        name="middleware-model-test",
        llm=chatbot,
        available_tools=ToolManager([]),
        max_steps=1,
    )

    class PassThroughPipeline:
        async def awrap_model_call(self, request, handler):
            assert request.model == "middleware-model"
            return await handler(request.override(system_prompt="wrapped"))

    agent._middleware_pipeline = PassThroughPipeline()
    await agent.add_message("user", "work")

    assert await agent.think(model="middleware-model") is False
    assert chatbot.ask_tool.await_args.kwargs["model"] == "middleware-model"
