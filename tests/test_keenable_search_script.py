"""Mocked coverage for the research skill's opt-in Keenable search script.

No network access: requests.post is replaced with a fake so the opt-in gate,
endpoint choice, headers, result mapping, and error paths can be checked.
The executor tests run the script without the opt-in flag, so it exits
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
    / "examples" / "skills" / "research" / "scripts" / "keenable_search.py"
)

spec = importlib.util.spec_from_file_location("keenable_search", SCRIPT_PATH)
keenable_search = importlib.util.module_from_spec(spec)
spec.loader.exec_module(keenable_search)


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
def fake_post(monkeypatch):
    calls = []
    state = {"response": FakeResponse({"results": []})}

    def post(url, json=None, headers=None, timeout=None):
        calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        if isinstance(state["response"], Exception):
            raise state["response"]
        return state["response"]

    monkeypatch.setattr(keenable_search.requests, "post", post)
    return calls, state


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setenv("KEENABLE_SEARCH_ENABLED", "1")


# Opt-in


@pytest.mark.parametrize("value", [None, "", "0", "false", "no"])
def test_disabled_unless_opted_in(monkeypatch, fake_post, value):
    if value is None:
        monkeypatch.delenv("KEENABLE_SEARCH_ENABLED", raising=False)
    else:
        monkeypatch.setenv("KEENABLE_SEARCH_ENABLED", value)
    monkeypatch.setenv("KEENABLE_API_KEY", "test-key")
    calls, _ = fake_post

    result = keenable_search.search("solana validators")

    assert calls == []
    assert result["status"] == "error"
    assert "KEENABLE_SEARCH_ENABLED=1" in result["message"]


def test_main_disabled_prints_error_without_request(monkeypatch, capsys, fake_post):
    monkeypatch.delenv("KEENABLE_SEARCH_ENABLED", raising=False)
    monkeypatch.setattr("sys.stdin", io.StringIO("bitcoin etf flows"))
    calls, _ = fake_post

    keenable_search.main()

    output = json.loads(capsys.readouterr().out)
    assert calls == []
    assert output["status"] == "error"
    assert "opt-in" in output["message"]


@pytest.mark.parametrize("value", ["1", "true", "YES"])
def test_enabled_makes_request(monkeypatch, fake_post, value):
    monkeypatch.setenv("KEENABLE_SEARCH_ENABLED", value)
    monkeypatch.delenv("KEENABLE_API_KEY", raising=False)
    calls, _ = fake_post

    result = keenable_search.search("solana validators")

    assert result["status"] == "success"
    assert len(calls) == 1


# Key passthrough


def test_keyless_by_default(monkeypatch, enabled, fake_post):
    monkeypatch.delenv("KEENABLE_API_KEY", raising=False)
    calls, _ = fake_post

    result = keenable_search.search("solana validators", max_results=3)

    assert result["status"] == "success"
    assert calls[0]["url"] == "https://api.keenable.ai/v1/search/public"
    assert calls[0]["json"] == {"query": "solana validators", "max_results": 3}
    assert calls[0]["headers"] == {"X-Keenable-Title": "spoon-core"}


def test_blank_api_key_stays_keyless(monkeypatch, enabled, fake_post):
    monkeypatch.setenv("KEENABLE_API_KEY", "  ")
    calls, _ = fake_post

    keenable_search.search("neo n3")

    assert calls[0]["url"] == "https://api.keenable.ai/v1/search/public"
    assert "X-API-Key" not in calls[0]["headers"]


def test_api_key_uses_keyed_endpoint(monkeypatch, enabled, fake_post):
    monkeypatch.setenv("KEENABLE_API_KEY", "test-key")
    calls, _ = fake_post

    keenable_search.search("neo n3")

    assert calls[0]["url"] == "https://api.keenable.ai/v1/search"
    assert calls[0]["headers"] == {
        "X-Keenable-Title": "spoon-core",
        "X-API-Key": "test-key",
    }


def test_executor_passes_keenable_env_through(monkeypatch):
    monkeypatch.setenv("KEENABLE_SEARCH_ENABLED", "1")
    monkeypatch.setenv("KEENABLE_API_KEY", "test-key")

    env = ScriptExecutor()._build_environment()

    assert env["KEENABLE_SEARCH_ENABLED"] == "1"
    assert env["KEENABLE_API_KEY"] == "test-key"


def test_executor_omits_unset_keenable_env(monkeypatch):
    monkeypatch.delenv("KEENABLE_SEARCH_ENABLED", raising=False)
    monkeypatch.delenv("KEENABLE_API_KEY", raising=False)

    env = ScriptExecutor()._build_environment()

    assert "KEENABLE_SEARCH_ENABLED" not in env
    assert "KEENABLE_API_KEY" not in env


@pytest.mark.asyncio
async def test_executor_subprocess_sees_keenable_env(monkeypatch):
    monkeypatch.setenv("KEENABLE_SEARCH_ENABLED", "1")
    monkeypatch.setenv("KEENABLE_API_KEY", "test-key")
    script = SkillScript(
        name="print-env",
        type=ScriptType.PYTHON,
        inline=(
            "import os; print(os.environ.get('KEENABLE_SEARCH_ENABLED'), "
            "os.environ.get('KEENABLE_API_KEY'))"
        ),
    )

    result = await ScriptExecutor(enabled=True).execute(script)

    assert result.success is True
    assert result.stdout.strip() == "1 test-key"


@pytest.mark.asyncio
async def test_executor_runs_script_disabled_by_default(monkeypatch):
    monkeypatch.delenv("KEENABLE_SEARCH_ENABLED", raising=False)
    script = SkillScript(
        name="keenable_search",
        type=ScriptType.PYTHON,
        file=str(SCRIPT_PATH),
    )

    result = await ScriptExecutor(enabled=True).execute(script, input_text="neo n3")

    assert result.success is True
    output = json.loads(result.stdout)
    assert output["status"] == "error"
    assert "KEENABLE_SEARCH_ENABLED=1" in output["message"]


# Result mapping and errors


def test_results_map_snippet_to_content(monkeypatch, enabled, fake_post):
    monkeypatch.delenv("KEENABLE_API_KEY", raising=False)
    _, state = fake_post
    state["response"] = FakeResponse({
        "results": [
            {"title": "A", "url": "https://a.example", "description": "", "snippet": "x" * 5000},
            {"title": "B", "url": "https://b.example", "snippet": None},
        ]
    })

    result = keenable_search.search("query")

    assert result["results"][0] == {
        "title": "A",
        "url": "https://a.example",
        "content": "x" * keenable_search.MAX_CONTENT_CHARS,
    }
    assert result["results"][1]["content"] == ""


@pytest.mark.parametrize(
    "error",
    [FakeResponse({}, status_code=429), requests.Timeout("timed out")],
)
def test_errors_return_error_status(monkeypatch, enabled, fake_post, error):
    monkeypatch.delenv("KEENABLE_API_KEY", raising=False)
    _, state = fake_post
    state["response"] = error

    result = keenable_search.search("query")

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
    assert keenable_search.read_query(raw) == expected


def test_main_without_query_prints_error(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO(""))

    keenable_search.main()

    assert json.loads(capsys.readouterr().out)["status"] == "error"
