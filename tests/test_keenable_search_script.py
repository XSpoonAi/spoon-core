"""Mocked coverage for the research skill's opt-in Keenable search script.

No network access: requests.post is replaced with a fake so the endpoint
choice, headers, result mapping, and error paths can be checked.
"""

import importlib.util
import io
import json
from pathlib import Path

import pytest
import requests

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


def test_keyless_by_default(monkeypatch, fake_post):
    monkeypatch.delenv("KEENABLE_API_KEY", raising=False)
    calls, _ = fake_post

    result = keenable_search.search("solana validators", max_results=3)

    assert result["status"] == "success"
    assert calls[0]["url"] == "https://api.keenable.ai/v1/search/public"
    assert calls[0]["json"] == {"query": "solana validators", "max_results": 3}
    assert calls[0]["headers"] == {"X-Keenable-Title": "spoon-core"}


def test_api_key_uses_keyed_endpoint(monkeypatch, fake_post):
    monkeypatch.setenv("KEENABLE_API_KEY", "test-key")
    calls, _ = fake_post

    keenable_search.search("neo n3")

    assert calls[0]["url"] == "https://api.keenable.ai/v1/search"
    assert calls[0]["headers"]["X-API-Key"] == "test-key"
    assert calls[0]["headers"]["X-Keenable-Title"] == "spoon-core"


def test_results_map_snippet_to_content(monkeypatch, fake_post):
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
def test_errors_return_error_status(monkeypatch, fake_post, error):
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
