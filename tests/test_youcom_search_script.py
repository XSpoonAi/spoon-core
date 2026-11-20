"""Mocked coverage for the research skill's opt-in You.com search script.

No network access: requests.get and requests.post are replaced with fakes so the
opt-in gate, endpoint choice, headers, result mapping, and error paths can be
checked. The executor tests run the script without the opt-in flag, so it exits
before any request.
"""

import importlib.util
import io
import json
from pathlib import Path

import pytest
import requests

from spoon_ai.skills.executor import ScriptExecutor
from spoon_ai.skills.models import ScriptType, SkillScript

SCRIPT_PATH = (
    Path(__file__).parent.parent
    / "examples" / "skills" / "research" / "scripts" / "youcom_search.py"
)

spec = importlib.util.spec_from_file_location("youcom_search", SCRIPT_PATH)
youcom_search = importlib.util.module_from_spec(spec)
spec.loader.exec_module(youcom_search)

USER_AGENT = "youdotcom-integration/XSpoonAi-spoon-core"


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")

    def json(self):
        return self.payload


@pytest.fixture
def fake_http(monkeypatch):
    calls = []
    state = {"response": FakeResponse({"results": {"web": []}})}

    def record(method):
        def call(url, json=None, params=None, headers=None, timeout=None):
            calls.append(
                {"method": method, "url": url, "json": json, "params": params,
                 "headers": headers, "timeout": timeout}
            )
            if isinstance(state["response"], Exception):
                raise state["response"]
            return state["response"]
        return call

    monkeypatch.setattr(youcom_search.requests, "get", record("GET"))
    monkeypatch.setattr(youcom_search.requests, "post", record("POST"))
    return calls, state


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setenv("YOUCOM_SEARCH_ENABLED", "1")


# Opt-in


@pytest.mark.parametrize("value", [None, "", "0", "false", "no"])
def test_disabled_unless_opted_in(monkeypatch, fake_http, value):
    if value is None:
        monkeypatch.delenv("YOUCOM_SEARCH_ENABLED", raising=False)
    else:
        monkeypatch.setenv("YOUCOM_SEARCH_ENABLED", value)
    monkeypatch.setenv("YDC_API_KEY", "test-key")
    calls, _ = fake_http

    result = youcom_search.search("solana validators")

    assert calls == []
    assert result["status"] == "error"
    assert "YOUCOM_SEARCH_ENABLED=1" in result["message"]


def test_main_disabled_prints_error_without_request(monkeypatch, capsys, fake_http):
    monkeypatch.delenv("YOUCOM_SEARCH_ENABLED", raising=False)
    monkeypatch.setattr("sys.stdin", io.StringIO("bitcoin etf flows"))
    calls, _ = fake_http

    youcom_search.main()

    output = json.loads(capsys.readouterr().out)
    assert calls == []
    assert output["status"] == "error"
    assert "opt-in" in output["message"]


@pytest.mark.parametrize("value", ["1", "true", "YES"])
def test_enabled_makes_request(monkeypatch, fake_http, value):
    monkeypatch.setenv("YOUCOM_SEARCH_ENABLED", value)
    monkeypatch.delenv("YDC_API_KEY", raising=False)
    calls, _ = fake_http

    result = youcom_search.search("solana validators")

    assert result["status"] == "success"
    assert len(calls) == 1


# Key passthrough


def test_keyless_by_default(monkeypatch, enabled, fake_http):
    monkeypatch.delenv("YDC_API_KEY", raising=False)
    calls, _ = fake_http

    result = youcom_search.search("solana validators", max_results=3)

    assert result["status"] == "success"
    assert calls[0]["method"] == "GET"
    assert calls[0]["url"] == "https://api.you.com/v1/agents/search"
    assert calls[0]["params"] == {"query": "solana validators", "count": 3}
    assert calls[0]["headers"] == {"User-Agent": USER_AGENT}


def test_keyless_caps_result_count(monkeypatch, enabled, fake_http):
    monkeypatch.delenv("YDC_API_KEY", raising=False)
    calls, _ = fake_http

    youcom_search.search("neo n3", max_results=80)

    assert calls[0]["params"]["count"] == 50


def test_blank_api_key_stays_keyless(monkeypatch, enabled, fake_http):
    monkeypatch.setenv("YDC_API_KEY", "  ")
    calls, _ = fake_http

    youcom_search.search("neo n3")

    assert calls[0]["url"] == "https://api.you.com/v1/agents/search"
    assert "X-API-Key" not in calls[0]["headers"]


def test_api_key_uses_keyed_endpoint(monkeypatch, enabled, fake_http):
    monkeypatch.setenv("YDC_API_KEY", "test-key")
    calls, _ = fake_http

    youcom_search.search("neo n3", max_results=4)

    assert calls[0]["method"] == "POST"
    assert calls[0]["url"] == "https://ydc-index.io/v1/search"
    assert calls[0]["json"] == {"query": "neo n3", "count": 4}
    assert calls[0]["headers"] == {"User-Agent": USER_AGENT, "X-API-Key": "test-key"}


def test_executor_passes_youcom_env_through(monkeypatch):
    monkeypatch.setenv("YOUCOM_SEARCH_ENABLED", "1")
    monkeypatch.setenv("YDC_API_KEY", "test-key")

    env = ScriptExecutor()._build_environment()

    assert env["YOUCOM_SEARCH_ENABLED"] == "1"
    assert env["YDC_API_KEY"] == "test-key"


def test_executor_omits_unset_youcom_env(monkeypatch):
    monkeypatch.delenv("YOUCOM_SEARCH_ENABLED", raising=False)
    monkeypatch.delenv("YDC_API_KEY", raising=False)

    env = ScriptExecutor()._build_environment()

    assert "YOUCOM_SEARCH_ENABLED" not in env
    assert "YDC_API_KEY" not in env


@pytest.mark.asyncio
async def test_executor_subprocess_sees_youcom_env(monkeypatch):
    monkeypatch.setenv("YOUCOM_SEARCH_ENABLED", "1")
    monkeypatch.setenv("YDC_API_KEY", "test-key")
    script = SkillScript(
        name="print-env",
        type=ScriptType.PYTHON,
        inline=(
            "import os; print(os.environ.get('YOUCOM_SEARCH_ENABLED'), "
            "os.environ.get('YDC_API_KEY'))"
        ),
    )

    result = await ScriptExecutor(enabled=True).execute(script)

    assert result.success is True
    assert result.stdout.strip() == "1 test-key"


@pytest.mark.asyncio
async def test_executor_runs_script_disabled_by_default(monkeypatch):
    monkeypatch.delenv("YOUCOM_SEARCH_ENABLED", raising=False)
    script = SkillScript(
        name="youcom_search",
        type=ScriptType.PYTHON,
        file=str(SCRIPT_PATH),
    )

    result = await ScriptExecutor(enabled=True).execute(script, input_text="neo n3")

    assert result.success is True
    output = json.loads(result.stdout)
    assert output["status"] == "error"
    assert "YOUCOM_SEARCH_ENABLED=1" in output["message"]


# Result mapping and errors


def test_results_map_snippets_to_content(monkeypatch, enabled, fake_http):
    monkeypatch.delenv("YDC_API_KEY", raising=False)
    _, state = fake_http
    state["response"] = FakeResponse({
        "results": {
            "web": [
                {"title": "A", "url": "https://a.example", "description": "d", "snippets": ["x" * 600, "y" * 600]},
                {"title": "B", "url": "https://b.example", "description": "only description", "snippets": []},
                {"title": "C", "url": "https://c.example"},
            ]
        }
    })

    result = youcom_search.search("query")

    assert result["results"][0] == {
        "title": "A",
        "url": "https://a.example",
        "content": ("x" * 600 + " " + "y" * 600)[:youcom_search.MAX_CONTENT_CHARS],
    }
    assert result["results"][1]["content"] == "only description"
    assert result["results"][2]["content"] == ""


def test_empty_results_are_a_success(monkeypatch, enabled, fake_http):
    monkeypatch.delenv("YDC_API_KEY", raising=False)
    _, state = fake_http
    state["response"] = FakeResponse({"results": {}})

    result = youcom_search.search("query")

    assert result == {"status": "success", "query": "query", "results": []}


@pytest.mark.parametrize(
    "error",
    [FakeResponse({}, status_code=429), requests.Timeout("timed out")],
)
def test_errors_return_error_status(monkeypatch, enabled, fake_http, error):
    monkeypatch.delenv("YDC_API_KEY", raising=False)
    _, state = fake_http
    state["response"] = error

    result = youcom_search.search("query")

    assert result["status"] == "error"
    assert result["message"]


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("  bitcoin etf flows \n", "bitcoin etf flows"),
        ('{"query": "eth staking"}', "eth staking"),
        ('{"input": "neo x"}', "neo x"),
        ("{}", ""),
    ],
)
def test_read_query(raw, expected):
    assert youcom_search.read_query(raw) == expected


def test_main_without_query_prints_error(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO(""))

    youcom_search.main()

    assert json.loads(capsys.readouterr().out)["status"] == "error"
