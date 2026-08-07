from __future__ import annotations

import asyncio
import sys
import time
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
    def __init__(self, *, fail_commit=False) -> None:
        self.messages = []
        self.batch_sizes = []
        self.commits = 0
        self.fail_commit = fail_commit

    def batch_add_messages(self, messages):
        self.batch_sizes.append(len(messages))
        self.messages.extend(messages)

    def commit(self):
        if self.fail_commit:
            raise RuntimeError("commit failed")
        self.commits += 1


class FakeClient:
    def __init__(self, *, fail=False, fail_commit=False, search_result=None) -> None:
        self.fail = fail
        self.fail_commit = fail_commit
        self.search_result = search_result
        self.initialized = False
        self.closes = 0
        self.searches = []
        self.sessions = {}

    def initialize(self):
        if self.fail:
            raise RuntimeError("offline")
        self.initialized = True

    def get_session(self, session_id, *, auto_create=False):
        self.sessions.setdefault(session_id, FakeSession(fail_commit=self.fail_commit))
        return {"session_id": session_id}

    def search(self, **kwargs):
        self.searches.append(kwargs)
        return self.search_result or {
            "memories": [{"abstract": "User prefers concise answers"}]
        }

    def session(self, session_id):
        return self.sessions.setdefault(
            session_id, FakeSession(fail_commit=self.fail_commit)
        )

    def close(self):
        self.closes += 1
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
        assert request.system_prompt.startswith("Be helpful\n\n")
        assert "User prefers concise answers" in request.system_prompt
        assert "potentially stale background" in request.system_prompt
        return ModelResponse(content="Concise answers.")

    await middleware.awrap_model_call(
        ModelRequest(system_prompt="Be helpful", runtime=runtime), model_handler
    )
    runtime.messages.append(Message(role="assistant", content="Concise answers."))
    middleware.after_agent({}, runtime)
    middleware.close()

    session = client.sessions["session-1"]
    assert client.searches[0]["session_id"] == "session-1"
    assert [message["role"] for message in session.messages] == ["user", "assistant"]
    assert session.commits == 1


@pytest.mark.asyncio
async def test_tool_events_are_bounded_and_sensitive_values_redacted():
    client = FakeClient()
    middleware = OpenVikingMemoryMiddleware(
        client=client, session_id="session-1", max_event_chars=120
    )
    runtime = make_runtime([Message(role="user", content="Call it")])
    middleware.before_agent({}, runtime)

    async def tool_handler(request):
        return ToolCallResult(
            output=(
                '{"token":"json-secret","api_key":"api-secret"} '
                "{'private_key': 'python-secret'} "
                "token=output-secret refresh_token=second-secret " + "y" * 200
            )
        )

    await middleware.awrap_tool_call(
        ToolCallRequest(
            tool_name="service",
            arguments={
                "api_key": "secret-value",
                "nested": {"accessToken": "nested-secret"},
                "query": "token=input-secret " + "x" * 200,
            },
            tool_call_id="call-1",
            runtime=runtime,
        ),
        tool_handler,
    )
    runtime.messages.append(
        Message(role="tool", content="raw Spoon tool result", tool_call_id="call-1")
    )
    middleware.after_agent({}, runtime)
    middleware.close()

    messages = client.sessions["session-1"].messages
    assert all(message["role"] in {"user", "assistant"} for message in messages)
    captured = messages[-1]["parts"][0]
    assert captured["type"] == "tool"
    assert captured["tool_id"] == "call-1"
    assert captured["tool_status"] == "completed"
    assert "secret-value" not in str(captured["tool_input"])
    assert "nested-secret" not in str(captured["tool_input"])
    assert "input-secret" not in str(captured["tool_input"])
    assert captured["tool_input"] == {"truncated": True}
    assert "output-secret" not in captured["tool_output"]
    assert "second-secret" not in captured["tool_output"]
    assert "json-secret" not in captured["tool_output"]
    assert "api-secret" not in captured["tool_output"]
    assert "python-secret" not in captured["tool_output"]
    assert len(str(captured["tool_input"])) <= 122
    assert len(captured["tool_output"]) <= 120


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
    middleware.close()


def test_identity_configuration_is_forwarded_to_sdk(monkeypatch):
    created = {}

    class Client:
        def __init__(self, **kwargs):
            created.update(kwargs)

        def close(self):
            created["closed"] = True

    monkeypatch.setitem(
        sys.modules, "openviking_sdk", SimpleNamespace(SyncHTTPClient=Client)
    )
    middleware = OpenVikingMemoryMiddleware(
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
        "timeout": 5.0,
    }
    middleware.close()
    assert created["closed"] is True


def test_close_does_not_close_injected_client():
    client = FakeClient()
    middleware = OpenVikingMemoryMiddleware(client=client)
    middleware._ensure_initialized()

    middleware.close()

    assert client.closes == 0


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
    middleware.close()

    assert client.searches == []
    assert client.sessions["session-1"].commits == 0


@pytest.mark.asyncio
async def test_recall_keeps_memories_and_resources_with_context_limit():
    client = FakeClient(
        search_result={
            "memories": [{"abstract": "remembered"}],
            "resources": [{"abstract": "resource"}],
        }
    )
    middleware = OpenVikingMemoryMiddleware(
        client=client, session_id="session-1", max_context_chars=100
    )
    runtime = make_runtime([Message(role="user", content="Recall both")])
    middleware.before_agent({}, runtime)

    async def handler(request):
        assert "remembered" in request.system_prompt
        assert "resource" in request.system_prompt
        return ModelResponse(content="ok")

    await middleware.awrap_model_call(ModelRequest(runtime=runtime), handler)
    assert len(middleware._render_recall(client.search_result)) == 100
    middleware.close()


@pytest.mark.asyncio
async def test_recall_keeps_skills_alongside_other_context():
    client = FakeClient(
        search_result={
            "memories": [{"abstract": "remembered"}],
            "resources": [{"abstract": "resource"}],
            "skills": [{"abstract": "skill"}],
        }
    )
    middleware = OpenVikingMemoryMiddleware(client=client, session_id="session-1")
    runtime = make_runtime([Message(role="user", content="Recall everything")])
    middleware.before_agent({}, runtime)

    async def handler(request):
        assert "remembered" in request.system_prompt
        assert "resource" in request.system_prompt
        assert "skill" in request.system_prompt
        return ModelResponse(content="ok")

    await middleware.awrap_model_call(ModelRequest(runtime=runtime), handler)
    middleware.close()


@pytest.mark.asyncio
async def test_message_trimming_does_not_drop_current_request():
    client = FakeClient()
    middleware = OpenVikingMemoryMiddleware(client=client, session_id="session-1")
    current_request = Message(role="user", content="current request")
    runtime = make_runtime(
        [Message(role="assistant", content=f"old-{index}") for index in range(99)]
        + [current_request]
    )
    middleware.before_agent({}, runtime)

    runtime.messages.pop(0)
    runtime.messages.append(Message(role="assistant", content="middle reply"))
    runtime.messages.pop(0)
    runtime.messages.append(Message(role="assistant", content="final"))
    middleware.after_agent({}, runtime)
    middleware.close()

    assert client.sessions["session-1"].messages == [
        {"role": "user", "content": "current request"},
        {"role": "assistant", "content": "middle reply"},
        {"role": "assistant", "content": "final"},
    ]


@pytest.mark.asyncio
async def test_message_trimming_keeps_run_when_current_request_is_evicted():
    client = FakeClient()
    middleware = OpenVikingMemoryMiddleware(client=client, session_id="session-1")
    current_request = Message(role="user", content="current request")
    runtime = make_runtime([current_request])
    middleware.before_agent({}, runtime)

    runtime.messages[:] = [Message(role="assistant", content="final")]
    middleware.after_agent({}, runtime)
    middleware.close()

    assert client.sessions["session-1"].messages == [
        {"role": "user", "content": "current request"},
        {"role": "assistant", "content": "final"},
    ]


@pytest.mark.asyncio
async def test_capture_respects_openviking_batch_limit():
    client = FakeClient()
    middleware = OpenVikingMemoryMiddleware(client=client, session_id="session-1")
    current_request = Message(role="user", content="current request")
    runtime = make_runtime([current_request])
    middleware.before_agent({}, runtime)

    runtime.messages[:] = [
        Message(role="assistant", content=f"reply-{index}") for index in range(100)
    ]
    middleware.after_agent({}, runtime)
    middleware.close()

    session = client.sessions["session-1"]
    assert len(session.messages) == 101
    assert session.batch_sizes == [100, 1]


@pytest.mark.asyncio
async def test_tool_events_preserve_message_order():
    client = FakeClient()
    middleware = OpenVikingMemoryMiddleware(client=client, session_id="session-1")
    runtime = make_runtime([Message(role="user", content="run tools")])
    middleware.before_agent({}, runtime)

    async def run_tool(request):
        return ToolCallResult(output=f"result-{request.tool_call_id}")

    for index in (1, 2):
        runtime.messages.append(Message(role="assistant", content=f"step-{index}"))
        await middleware.awrap_tool_call(
            ToolCallRequest(
                tool_name="service",
                arguments={"step": index},
                tool_call_id=f"call-{index}",
                runtime=runtime,
            ),
            run_tool,
        )
        runtime.messages.append(
            Message(role="tool", content="raw", tool_call_id=f"call-{index}")
        )
    runtime.messages.append(Message(role="assistant", content="final"))
    middleware.after_agent({}, runtime)
    middleware.close()

    messages = client.sessions["session-1"].messages
    assert [message.get("content") for message in messages] == [
        "run tools",
        "step-1",
        None,
        "step-2",
        None,
        "final",
    ]
    assert messages[2]["parts"][0]["tool_id"] == "call-1"
    assert messages[4]["parts"][0]["tool_id"] == "call-2"


@pytest.mark.asyncio
async def test_slow_provider_does_not_block_event_loop():
    class SlowClient(FakeClient):
        def search(self, **kwargs):
            time.sleep(0.05)
            return super().search(**kwargs)

    middleware = OpenVikingMemoryMiddleware(
        client=SlowClient(),
        session_id="session-1",
        provider_timeout_seconds=0.02,
    )
    runtime = make_runtime([Message(role="user", content="keep loop responsive")])
    middleware.before_agent({}, runtime)
    loop_advanced = False

    async def handler(request):
        return ModelResponse(content="ok")

    async def tick():
        nonlocal loop_advanced
        await asyncio.sleep(0.01)
        loop_advanced = True

    await asyncio.gather(
        middleware.awrap_model_call(ModelRequest(runtime=runtime), handler), tick()
    )
    assert loop_advanced
    middleware.close()


def test_commit_failure_is_fail_open(caplog):
    client = FakeClient(fail_commit=True)
    middleware = OpenVikingMemoryMiddleware(client=client, session_id="session-1")
    runtime = make_runtime([Message(role="user", content="continue")])

    middleware.before_agent({}, runtime)
    runtime.messages.append(Message(role="assistant", content="done"))
    assert middleware.after_agent({}, runtime) is None
    middleware.close()

    assert client.sessions["session-1"].messages
    assert "capture unavailable" in caplog.text
