"""Pydantic tool wrapper around :class:`~spoon_ai.x402.client.X402Client`."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Mapping, Optional

import requests
from pydantic import Field

from spoon_ai.tools.base import BaseTool, ToolFailure, ToolResult
from spoon_ai.x402 import X402Client


class X402RequestTool(BaseTool):
    """Agent tool that performs HTTP requests using X402 authentication."""

    name: str = Field(default="x402_request", init=False)
    description: str = Field(
        default=(
            "Perform an HTTP request against an endpoint protected by the X402 "
            "protocol, automatically resolving challenges when possible."
        ),
        init=False,
    )
    parameters: Dict[str, Any] = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The request URL."},
                "method": {
                    "type": "string",
                    "description": "HTTP method to use (defaults to GET).",
                    "default": "GET",
                },
                "headers": {
                    "type": "object",
                    "description": "Optional headers to include in the request.",
                },
                "json": {
                    "description": "JSON payload to send with the request.",
                },
                "params": {
                    "type": "object",
                    "description": "Query parameters to append to the request.",
                },
                "data": {
                    "description": "Arbitrary data payload to include in the request.",
                },
                "timeout": {
                    "type": ["number", "null"],
                    "description": "Optional request timeout in seconds.",
                },
            },
            "required": ["url"],
            "additionalProperties": False,
        },
        init=False,
    )

    client: X402Client = Field(exclude=True)
    default_timeout: Optional[float] = Field(default=None, exclude=True)

    async def execute(
        self,
        *,
        url: str,
        method: str = "GET",
        headers: Optional[Mapping[str, str]] = None,
        params: Optional[Mapping[str, Any]] = None,
        json: Optional[Any] = None,
        data: Optional[Any] = None,
        timeout: Optional[float] = None,
    ) -> ToolResult:
        method = method.upper()
        timeout = timeout if timeout is not None else self.default_timeout

        try:
            response = await asyncio.to_thread(
                self.client.request,
                method,
                url,
                headers=dict(headers or {}),
                params=params,
                json=json,
                data=data,
                timeout=timeout,
            )
        except requests.RequestException as exc:  # pragma: no cover - network errors
            raise ToolFailure(f"HTTP request failed: {exc}") from exc

        content: Any
        try:
            content = response.json()
        except ValueError:
            content = response.text

        return ToolResult(
            output={
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "content": content,
            }
        )
