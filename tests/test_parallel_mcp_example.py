"""Mocked coverage for the opt-in Parallel MCP example.

No network access: the MCP tools are replaced with fakes so the example's
tool discovery, session handling, error paths, and cleanup can be checked.
"""

import asyncio
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.mcp.parallel_search_demo import (
    PARALLEL_MCP_URL,
    create_parallel_tool,
    extract_session_id,
    run_demo,
)


class FakeTool:
    """Stand-in for one MCPTool expanded from the server's tool list."""

    def __init__(self, name, result=None, error=None):
        self.name = name
        self.result = result
        self.error = error
        self.calls = []
        self.cleanup_count = 0

    async def execute(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.result

    async def cleanup(self):
        self.cleanup_count += 1


class FakeServer:
    """Stand-in for the one-per-server MCPTool the example creates."""

    def __init__(self, tools=None, discovery_error=None):
        self.tools = tools or []
        self.discovery_error = discovery_error
        self.cleanup_count = 0

    async def expand_server_tools(self):
        if self.discovery_error is not None:
            raise self.discovery_error
        return list(self.tools)

    async def cleanup(self):
        self.cleanup_count += 1


def install_server(monkeypatch, server):
    monkeypatch.setattr(
        "examples.mcp.parallel_search_demo.create_parallel_tool", lambda: server
    )
    return server


def search_payload(session_id="session_abc123"):
    return json.dumps(
        {
            "search_id": "search_abc123",
            "results": [{"url": "https://example.com", "excerpts": ["example"]}],
            "session_id": session_id,
        }
    )


def test_create_parallel_tool_targets_the_current_endpoint():
    tool = create_parallel_tool()
    config = tool.mcp_config

    assert config["url"] == "https://mcp.parallel.ai/mcp"
    assert PARALLEL_MCP_URL == config["url"]
    assert not config["url"].endswith("/mcp/")
    assert config["transport"] == "http"
    assert "Authorization" not in config["headers"]
    assert "x-parallel-search-config" not in {k.lower() for k in config["headers"]}


@pytest.mark.parametrize(
    "result,expected",
    [
        (search_payload("session_xyz"), "session_xyz"),
        (json.dumps({"results": []}), None),
        (json.dumps({"session_id": ""}), None),
        (json.dumps({"session_id": 7}), None),
        ("not json at all", None),
        ("[1, 2, 3]", None),
        (None, None),
    ],
)
def test_extract_session_id(result, expected):
    assert extract_session_id(result) == expected


async def test_search_omits_session_id_and_fetch_reuses_the_server_value(monkeypatch):
    search = FakeTool("web_search", result=search_payload())
    fetch = FakeTool("web_fetch", result=json.dumps({"results": []}))
    server = install_server(monkeypatch, FakeServer([search, fetch]))

    results = await run_demo("asyncio docs", "https://docs.python.org/3/library/asyncio.html")

    assert set(results) == {"web_search", "web_fetch"}
    assert "session_id" not in search.calls[0]
    assert fetch.calls[0]["session_id"] == "session_abc123"
    assert fetch.calls[0]["urls"] == ["https://docs.python.org/3/library/asyncio.html"]
    assert search.cleanup_count == 1
    assert fetch.cleanup_count == 1
    assert server.cleanup_count == 1


async def test_fetch_omits_session_id_when_the_server_returned_none(monkeypatch):
    search = FakeTool("web_search", result="plain text response")
    fetch = FakeTool("web_fetch", result="fetched")
    install_server(monkeypatch, FakeServer([search, fetch]))

    await run_demo("asyncio docs", "https://docs.python.org/3/library/asyncio.html")

    assert "session_id" not in fetch.calls[0]


async def test_missing_fetch_tool_raises_and_cleans_up(monkeypatch):
    search = FakeTool("web_search", result=search_payload())
    server = install_server(monkeypatch, FakeServer([search]))

    with pytest.raises(RuntimeError) as raised:
        await run_demo("asyncio docs", "https://docs.python.org/3/library/asyncio.html")

    assert "web_fetch" in str(raised.value)
    assert search.calls == []
    assert search.cleanup_count == 1
    assert server.cleanup_count == 1


async def test_empty_tool_list_raises_and_cleans_up(monkeypatch):
    server = install_server(monkeypatch, FakeServer([]))

    with pytest.raises(RuntimeError) as raised:
        await run_demo("asyncio docs")

    assert "web_search" in str(raised.value)
    assert server.cleanup_count == 1


async def test_search_error_propagates_and_cleans_up(monkeypatch):
    search = FakeTool("web_search", error=RuntimeError("MCP tool 'web_search' failed: 500"))
    fetch = FakeTool("web_fetch")
    server = install_server(monkeypatch, FakeServer([search, fetch]))

    with pytest.raises(RuntimeError, match="500"):
        await run_demo("asyncio docs", "https://docs.python.org/3/library/asyncio.html")

    assert fetch.calls == []
    assert search.cleanup_count == 1
    assert fetch.cleanup_count == 1
    assert server.cleanup_count == 1


async def test_search_timeout_propagates_and_cleans_up(monkeypatch):
    search = FakeTool("web_search", error=asyncio.TimeoutError())
    server = install_server(monkeypatch, FakeServer([search]))

    with pytest.raises(asyncio.TimeoutError):
        await run_demo("asyncio docs")

    assert search.cleanup_count == 1
    assert server.cleanup_count == 1


async def test_discovery_failure_cleans_up_the_server(monkeypatch):
    server = install_server(
        monkeypatch, FakeServer(discovery_error=ConnectionError("server unreachable"))
    )

    with pytest.raises(ConnectionError):
        await run_demo("asyncio docs")

    assert server.cleanup_count == 1
