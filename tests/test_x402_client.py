from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # pragma: no cover - defensive path setup
    sys.path.insert(0, str(PROJECT_ROOT))

try:  # pragma: no cover - optional dependency shim for CI environments
    import requests  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback when requests is unavailable
    from types import ModuleType

    class _Response:  # Minimal stub mimicking requests.Response for testing
        def __init__(self) -> None:
            self.status_code = 200
            self.headers: Dict[str, str] = {}
            self._content: bytes = b""
            self.url = ""

        def json(self):
            raise ValueError("JSON not available in stub response")

        @property
        def text(self) -> str:
            return self._content.decode()

    class _Session:  # Minimal stub used only for typing in mocks
        def request(self, *args, **kwargs):  # pragma: no cover - stub method
            raise NotImplementedError

    requests = ModuleType("requests")
    requests.Response = _Response
    requests.Session = _Session
    sys.modules.setdefault("requests", requests)

from spoon_ai.x402.client import (
    StaticX402PaymentProvider,
    X402Authorization,
    X402Challenge,
    X402Client,
)


def _build_response(status_code: int, headers: Dict[str, str] | None = None, body: bytes = b"") -> requests.Response:
    response = requests.Response()
    response.status_code = status_code
    response.headers.update(headers or {})
    response._content = body
    response.url = "https://example.com"
    return response


def test_challenge_parsing_extracts_values() -> None:
    header = 'Basic realm="ignored", X-402 realm="restricted", invoice="bolt11", macaroon="mac", token="abc"'
    challenge = X402Challenge.from_header(header)
    assert challenge is not None
    assert challenge.realm == "restricted"
    assert challenge.invoice == "bolt11"
    assert challenge.macaroon == "mac"
    assert challenge.token == "abc"
    assert challenge.cache_key == "restricted"


def test_challenge_cache_key_falls_back_to_invoice() -> None:
    challenge = X402Challenge.from_header('X-402 invoice="bolt11"')
    assert challenge is not None
    assert challenge.cache_key == "invoice:bolt11"


def test_authorization_header_serialization() -> None:
    auth = X402Authorization(token="token", macaroon="mac", preimage="pre")
    header = auth.as_header()
    assert header.startswith("X-402 ")
    assert 'access_token="token"' in header
    assert 'macaroon="mac"' in header
    assert 'preimage="pre"' in header


class _CountingProvider:
    def __init__(self, authorization: X402Authorization | None = None):
        self.calls = 0
        self.authorization = authorization or X402Authorization(token="token-value")
        self.last_challenge: X402Challenge | None = None

    def obtain_authorization(self, challenge: X402Challenge) -> X402Authorization:
        self.calls += 1
        self.last_challenge = challenge
        return self.authorization


def test_client_retries_request_with_authorization() -> None:
    provider = _CountingProvider()
    session = MagicMock(spec=requests.Session)
    session.request.side_effect = [
        _build_response(
            402,
            headers={"WWW-Authenticate": 'X-402 realm="restricted", invoice="bolt11"'},
        ),
        _build_response(200, body=b"ok"),
    ]

    client = X402Client(provider, session=session)
    response = client.request("GET", "https://example.com")

    assert response.status_code == 200
    assert provider.calls == 1
    assert session.request.call_count == 2
    # The second call should include the Authorization header
    _, second_kwargs = session.request.call_args
    assert "Authorization" in second_kwargs["headers"]


def test_cached_authorization_reused() -> None:
    provider = _CountingProvider()
    session = MagicMock(spec=requests.Session)
    session.request.side_effect = [
        _build_response(
            402,
            headers={"WWW-Authenticate": 'X-402 realm="restricted"'},
        ),
        _build_response(200, body=b"first"),
        _build_response(200, body=b"second"),
    ]

    client = X402Client(provider, session=session)
    assert client.request("GET", "https://example.com").text == "first"
    assert client.request("GET", "https://example.com").text == "second"
    assert provider.calls == 1


def test_authorization_expiry_evicts_cache() -> None:
    provider = StaticX402PaymentProvider(token="abc")
    session = MagicMock(spec=requests.Session)
    expired_auth = X402Authorization(
        token="stale",
        expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
    )
    client = X402Client(
        provider,
        session=session,
        token_cache={"restricted": expired_auth},
    )

    session.request.side_effect = [
        _build_response(402, headers={"WWW-Authenticate": 'X-402 realm="restricted"'}),
        _build_response(200, body=b"ok"),
    ]

    response = client.request("GET", "https://example.com")
    assert response.status_code == 200
    assert session.request.call_count == 2


def test_client_evicts_cache_on_403() -> None:
    provider = _CountingProvider()
    session = MagicMock(spec=requests.Session)
    session.request.side_effect = [
        _build_response(
            402,
            headers={"WWW-Authenticate": 'X-402 realm="restricted"'},
        ),
        _build_response(403),
        _build_response(
            402,
            headers={"WWW-Authenticate": 'X-402 realm="restricted"'},
        ),
        _build_response(200, body=b"ok"),
    ]

    client = X402Client(provider, session=session)
    response = client.request("GET", "https://example.com")

    assert response.status_code == 200
    assert provider.calls == 2
    assert session.request.call_count == 4


def test_missing_challenge_raises() -> None:
    provider = _CountingProvider()
    session = MagicMock(spec=requests.Session)
    session.request.side_effect = [
        _build_response(402, headers={}),
    ]

    client = X402Client(provider, session=session)

    try:
        client.request("GET", "https://example.com")
    except RuntimeError as exc:
        assert "X-402" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected an error when no challenge is present")


def test_provider_exception_wrapped() -> None:
    class _FailingProvider(_CountingProvider):
        def obtain_authorization(self, challenge: X402Challenge) -> X402Authorization:
            raise RuntimeError("boom")

    provider = _FailingProvider()
    session = MagicMock(spec=requests.Session)
    session.request.side_effect = [
        _build_response(
            402,
            headers={"WWW-Authenticate": 'X-402 realm="restricted"'},
        ),
    ]

    client = X402Client(provider, session=session)

    try:
        client.request("GET", "https://example.com")
    except RuntimeError as exc:
        assert "boom" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected provider failure to propagate")
