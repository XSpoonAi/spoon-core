from __future__ import annotations

import sys
from types import SimpleNamespace
from uuid import uuid4

import pytest

from spoon_ai.middleware.base import (
    AgentRuntime,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
    ToolCallResult,
)
from spoon_ai.middleware.openviking_memory import OpenVikingMemoryMiddleware
from spoon_ai.schema import Message


class FakeSession:
    def __init__(self) -> None:
        self.messages = []
        self.commits = 0

    def batch_add_messages(self, messages):
        self.messages.extend(messages)

    def commit(self):
        self.commits += 1


class FakeClient:
    def __init__(self, *, fail=False) -> None:
        self.fail = fail
        self.initialized = False
        self.searches = []
        self.sessions = {}

    def initialize(self):
        if self.fail:
            raise RuntimeError("offline")
        self.initialized = True

    def get_session(self, session_id, *, auto_create=False):
        self.sessions.setdefault(session_id, FakeSession())
        return {"session_id": session_id}

    def search(self, **kwargs):
        self.searches.append(kwargs)
        return {"memories": [{"abstract": "User prefers concise answers"}]}

    def session(self, session_id):
        return self.sessions.setdefault(session_id, FakeSession())

    def close(self):
        self.initialized = False


def make_runtime(messages):
    return AgentRuntime(
        agent_name="researcher",
        run_id=uuid4(),
        thread_id="thread-1",
        state={},
        messages=messages,
    )


@pytest.mark.asyncio
async def test_recall_is_injected_and_run_is_committed():
    client = FakeClient()
    middleware = OpenVikingMemoryMiddleware(client=client, session_id="session-1")
    runtime = make_runtime([Message(role="user", content="What do I prefer?")])

    middleware.before_agent({}, runtime)

    async def model_handler(request):
        assert "User prefers concise answers" in request.system_prompt
        assert "potentially stale background" in request.system_prompt
        return ModelResponse(content="Concise answers.")

    await middleware.awrap_model_call(
        ModelRequest(system_prompt="Be helpful", runtime=runtime), model_handler
    )
    runtime.messages.append(Message(role="assistant", content="Concise answers."))
    middleware.after_agent({}, runtime)

    session = client.sessions["session-1"]
    assert client.searches[0]["session_id"] == "session-1"
    assert [message["role"] for message in session.messages] == ["user", "assistant"]
    assert session.commits == 1


@pytest.mark.asyncio
async def test_tool_events_are_bounded_and_sensitive_values_redacted():
    client = FakeClient()
    middleware = OpenVikingMemoryMiddleware(
        client=client, session_id="session-1", max_event_chars=200
    )
    runtime = make_runtime([Message(role="user", content="Call it")])
    middleware.before_agent({}, runtime)

    async def tool_handler(request):
        return ToolCallResult(output="done")

    await middleware.awrap_tool_call(
        ToolCallRequest(
            tool_name="service",
            arguments={"api_key": "secret-value", "query": "safe"},
            tool_call_id="call-1",
            runtime=runtime,
        ),
        tool_handler,
    )
    runtime.messages.append(
        Message(role="tool", content="raw Spoon tool result", tool_call_id="call-1")
    )
    middleware.after_agent({}, runtime)

    messages = client.sessions["session-1"].messages
    assert all(message["role"] in {"user", "assistant"} for message in messages)
    captured_message = messages[-1]
    captured = captured_message["parts"][0]
    assert captured_message["role"] == "assistant"
    assert captured["type"] == "tool"
    assert captured["tool_id"] == "call-1"
    assert captured["tool_status"] == "completed"
    assert "secret-value" not in str(captured["tool_input"])
    assert captured["tool_input"]["api_key"] == "[REDACTED]"
    assert len(captured["tool_output"]) <= 200


@pytest.mark.asyncio
async def test_openviking_failure_does_not_stop_agent():
    middleware = OpenVikingMemoryMiddleware(client=FakeClient(fail=True))
    runtime = make_runtime([Message(role="user", content="Continue")])

    assert middleware.before_agent({}, runtime) is None

    async def handler(request):
        return ModelResponse(content="still running")

    response = await middleware.awrap_model_call(ModelRequest(runtime=runtime), handler)
    assert response.content == "still running"
    assert middleware.after_agent({}, runtime) is None


def test_identity_configuration_is_forwarded_to_sdk(monkeypatch):
    created = {}

    class Client:
        def __init__(self, **kwargs):
            created.update(kwargs)

    monkeypatch.setitem(
        sys.modules, "openviking_sdk", SimpleNamespace(SyncHTTPClient=Client)
    )
    OpenVikingMemoryMiddleware(
        url="https://memory.example",
        api_key="key",
        account="team",
        user="alice",
        actor_peer_id="research-agent",
    )

    assert created == {
        "url": "https://memory.example",
        "api_key": "key",
        "account": "team",
        "user": "alice",
        "actor_peer_id": "research-agent",
    }


def test_recall_and_commit_can_be_disabled():
    client = FakeClient()
    middleware = OpenVikingMemoryMiddleware(
        client=client,
        session_id="session-1",
        auto_recall=False,
        auto_commit=False,
    )
    runtime = make_runtime([Message(role="user", content="Do not recall")])

    middleware.before_agent({}, runtime)
    middleware.after_agent({}, runtime)

    assert client.searches == []
    assert client.sessions["session-1"].commits == 0
