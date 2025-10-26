"""Client utilities for resolving X402 authentication challenges."""

from __future__ import annotations

import datetime as _dt
import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, Mapping, MutableMapping, Optional, Protocol, Set, runtime_checkable
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)


class X402AuthenticationError(RuntimeError):
    """Raised when an X402 authentication challenge cannot be fulfilled."""


@dataclass(slots=True)
class X402Authorization:
    """Represents credentials that can satisfy an X402 challenge."""

    token: Optional[str] = None
    macaroon: Optional[str] = None
    preimage: Optional[str] = None
    scheme: str = "X-402"
    expires_at: Optional[_dt.datetime] = None
    extra: Dict[str, str] = field(default_factory=dict)

    def as_header(self) -> str:
        """Return the Authorization header value for the credentials."""
        parts: Dict[str, str] = {}
        if self.token:
            parts["access_token"] = self.token
        if self.macaroon:
            parts["macaroon"] = self.macaroon
        if self.preimage:
            parts["preimage"] = self.preimage
        parts.update(self.extra)
        header_params = ", ".join(
            f"{key}={_quote(value)}" for key, value in parts.items()
        )
        return f"{self.scheme} {header_params}" if header_params else self.scheme

    def is_expired(self, *, now: Optional[_dt.datetime] = None) -> bool:
        if self.expires_at is None:
            return False
        now = now or _dt.datetime.now(tz=_dt.timezone.utc)
        return now >= self.expires_at


@dataclass(slots=True)
class X402Challenge:
    """Parsed representation of an X402 challenge."""

    scheme: str
    params: Dict[str, str]
    raw: str

    @property
    def realm(self) -> Optional[str]:
        return self.params.get("realm")

    @property
    def invoice(self) -> Optional[str]:
        return self.params.get("invoice")

    @property
    def macaroon(self) -> Optional[str]:
        return self.params.get("macaroon")

    @property
    def token(self) -> Optional[str]:
        return self.params.get("token") or self.params.get("access_token")

    @property
    def cache_key(self) -> Optional[str]:
        """Return a deterministic cache key for the challenge."""

        if self.realm:
            return self.realm
        if self.token:
            return f"token:{self.token}"
        if self.invoice:
            return f"invoice:{self.invoice}"
        return None

    @classmethod
    def from_header(cls, header_value: Optional[str]) -> Optional["X402Challenge"]:
        """Parse a ``WWW-Authenticate`` header and return the X402 challenge, if any."""
        if not header_value:
            return None

        header_value = header_value.strip()
        if not header_value:
            return None

        fragments = _extract_challenges(header_value)
        index = 0
        while index < len(fragments):
            fragment = fragments[index].strip()
            index += 1
            if not fragment:
                continue
            scheme, _, param_str = fragment.partition(" ")
            if scheme.lower() != "x-402":
                continue

            # Accumulate any trailing parameter fragments that belong to the
            # current challenge. Some servers emit ``scheme key=value, key2=value``.
            assembled = fragment
            while index < len(fragments) and _looks_like_parameter(fragments[index]):
                assembled = f"{assembled}, {fragments[index].strip()}"
                index += 1

            _, _, param_str = assembled.partition(" ")
            params = _parse_params(param_str)
            return cls(scheme=scheme, params=params, raw=assembled)
        return None

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"X402Challenge(scheme={self.scheme!r}, params={self.params!r})"


@runtime_checkable
class X402PaymentProvider(Protocol):
    """Protocol for components capable of resolving X402 challenges."""

    def obtain_authorization(self, challenge: X402Challenge) -> X402Authorization:
        """Return valid credentials for the provided challenge."""


class StaticX402PaymentProvider:
    """Simple provider that always returns the same credentials."""

    def __init__(
        self,
        token: str,
        *,
        macaroon: Optional[str] = None,
        preimage: Optional[str] = None,
        scheme: str = "X-402",
    ):
        if not token and not preimage:
            raise ValueError("token or preimage must be provided")
        self._authorization = X402Authorization(
            token=token,
            macaroon=macaroon,
            preimage=preimage,
            scheme=scheme,
        )

    def obtain_authorization(self, challenge: X402Challenge) -> X402Authorization:  # noqa: D401 - short delegation
        """Return the configured authorization regardless of the challenge."""

        del challenge
        return self._authorization


class X402Client:
    """HTTP client capable of automatically resolving X402 challenges."""

    def __init__(
        self,
        payment_provider: X402PaymentProvider,
        *,
        session: Optional[requests.Session] = None,
        max_challenge_attempts: int = 4,
        token_cache: Optional[MutableMapping[str, X402Authorization]] = None,
        clock: Optional[Callable[[], _dt.datetime]] = None,
    ) -> None:
        if not isinstance(payment_provider, X402PaymentProvider):
            raise TypeError("payment_provider must implement X402PaymentProvider")
        if max_challenge_attempts < 1:
            raise ValueError("max_challenge_attempts must be >= 1")

        self._payment_provider = payment_provider
        self._session = session or requests.Session()
        self._max_attempts = max_challenge_attempts
        self._token_cache: MutableMapping[str, X402Authorization] = token_cache or {}
        self._host_cache: Dict[str, str] = {}
        self._clock = clock or (lambda: _dt.datetime.now(tz=_dt.timezone.utc))

    @property
    def session(self) -> requests.Session:
        return self._session

    def clear_cache(self) -> None:
        """Remove all cached authorizations."""

        self._token_cache.clear()
        self._host_cache.clear()

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Optional[Mapping[str, str]] = None,
        **kwargs,
    ) -> requests.Response:
        """Perform an HTTP request handling X402 challenges transparently."""

        attempts = 0
        host = urlparse(url).netloc
        cached_key = self._host_cache.get(host)
        cached_auth = self._get_cached_authorization(cached_key)
        resolved_keys: Set[str] = set()

        while attempts < self._max_attempts:
            attempts += 1
            now = self._clock()
            request_headers = dict(headers or {})

            if (
                cached_auth
                and not cached_auth.is_expired(now=now)
                and "Authorization" not in request_headers
            ):
                request_headers["Authorization"] = cached_auth.as_header()

            logger.debug(
                "X402 request attempt %s %s (Authorization=%s)",
                attempts,
                url,
                request_headers.get("Authorization"),
            )
            response = self._session.request(
                method,
                url,
                headers=request_headers,
                **kwargs,
            )

            if response.status_code != 402:
                if response.status_code in (401, 403) and cached_key:
                    if cached_auth is not None:
                        self._evict_cached_authorization(cached_key)
                        cached_auth = None
                    resolved_keys.discard(cached_key)
                    if attempts < self._max_attempts:
                        continue
                return response

            challenge = X402Challenge.from_header(response.headers.get("WWW-Authenticate"))
            if challenge is None:
                logger.debug("No X402 challenge present in response headers")
                raise X402AuthenticationError("Received 402 response without X-402 challenge")

            cache_key = challenge.cache_key or host
            self._host_cache[host] = cache_key
            cached_key = cache_key

            cached_auth = self._get_cached_authorization(cache_key)
            if cached_auth and cached_auth.is_expired(now=now):
                self._evict_cached_authorization(cache_key)
                cached_auth = None

            if cached_auth is not None:
                # We already have valid credentials for this challenge; retry with them.
                continue

            if cache_key in resolved_keys:
                raise X402AuthenticationError(
                    "Challenge could not be satisfied after obtaining credentials"
                )
            resolved_keys.add(cache_key)

            cached_auth = self._resolve_challenge(challenge)
            if cached_auth.is_expired(now=now):
                raise X402AuthenticationError(
                    "Received immediately expired X402 authorization from provider"
                )
            self._token_cache[cache_key] = cached_auth

        raise X402AuthenticationError("Exceeded maximum number of X402 challenge attempts")

    def _resolve_challenge(self, challenge: X402Challenge) -> X402Authorization:
        try:
            authorization = self._payment_provider.obtain_authorization(challenge)
        except Exception as exc:  # pragma: no cover - defensive programming
            logger.exception("Failed to resolve X402 challenge: %s", exc)
            raise X402AuthenticationError(str(exc)) from exc

        if not isinstance(authorization, X402Authorization):
            raise X402AuthenticationError(
                "Payment provider did not return an X402Authorization instance"
            )
        return authorization

    def _get_cached_authorization(
        self, cache_key: Optional[str]
    ) -> Optional[X402Authorization]:
        if cache_key is None:
            return None
        authorization = self._token_cache.get(cache_key)
        if authorization is None:
            return None
        return authorization

    def _evict_cached_authorization(self, cache_key: str) -> None:
        self._token_cache.pop(cache_key, None)
        stale_hosts = [host for host, key in self._host_cache.items() if key == cache_key]
        for host in stale_hosts:
            self._host_cache.pop(host, None)


_PARAM_PATTERN = re.compile(r"(?P<key>[^=\s]+)\s*=\s*(?P<value>\"[^\"]*\"|[^,]+)")


def _extract_challenges(header_value: str) -> list[str]:
    """Split a ``WWW-Authenticate`` header into individual challenges."""

    parts: list[str] = []
    current = []
    in_quotes = False
    for char in header_value:
        if char == '"':
            in_quotes = not in_quotes
        if char == ',' and not in_quotes:
            part = ''.join(current).strip()
            if part:
                parts.append(part)
            current = []
            continue
        current.append(char)
    final = ''.join(current).strip()
    if final:
        parts.append(final)
    return parts


def _parse_params(param_str: str) -> Dict[str, str]:
    params: Dict[str, str] = {}
    for match in _PARAM_PATTERN.finditer(param_str):
        key = match.group("key")
        value = match.group("value").strip()
        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        params[key] = value
    return params


def _looks_like_parameter(fragment: str) -> bool:
    fragment = fragment.strip()
    if not fragment or "=" not in fragment:
        return False
    key, _ = fragment.split("=", 1)
    return " " not in key


def _quote(value: str) -> str:
    return '"' + value.replace('"', '\\"') + '"'
