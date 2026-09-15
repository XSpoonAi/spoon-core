"""HOL Guard middleware for command-bearing tool calls."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Iterable
from typing import Any

from .base import AgentMiddleware, ToolCallRequest, ToolCallResult


class HolGuardMiddleware(AgentMiddleware):
    """Gate selected command-bearing tools through the HOL Guard CLI.

    Only an explicit benign ``allow`` decision reaches the wrapped tool handler.
    Review, deny, malformed output, provider errors, timeouts, and launch failures
    all stop execution before the tool runs.
    """

    def __init__(
        self,
        *,
        executable: str = "hol-guard",
        timeout_seconds: float = 10.0,
        guarded_tools: Iterable[str] = ("execute",),
    ) -> None:
        super().__init__()
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.executable = executable
        self.timeout_seconds = timeout_seconds
        self.guarded_tools = frozenset(guarded_tools)

    def _evaluate_command(self, command: str) -> tuple[bool, str]:
        try:
            result = subprocess.run(
                [self.executable, "command", "test", command, "--json"],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except FileNotFoundError:
            return False, "guard_unavailable"
        except subprocess.TimeoutExpired:
            return False, "guard_timeout"
        except (OSError, UnicodeError):
            return False, "guard_error"

        if result.returncode != 0:
            return False, "guard_error"

        try:
            payload: Any = json.loads(result.stdout)
        except (json.JSONDecodeError, TypeError):
            return False, "guard_invalid_output"

        if not isinstance(payload, dict):
            return False, "guard_invalid_output"
        classification = payload.get("classification")
        if not isinstance(classification, dict):
            return False, "guard_invalid_output"

        if (
            classification.get("explicitly_benign") is True
            and payload.get("minimum_action") == "allow"
        ):
            return True, "guard_allow"
        return False, "guard_block"

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolCallResult],
    ) -> ToolCallResult:
        if request.tool_name not in self.guarded_tools:
            return handler(request)

        command = request.arguments.get("command")
        if not isinstance(command, str) or not command.strip():
            return ToolCallResult.from_error(
                "HOL Guard blocked tool execution: missing command"
            )

        allowed, reason = self._evaluate_command(command)
        if not allowed:
            return ToolCallResult.from_error(
                f"HOL Guard blocked tool execution: {reason}"
            )

        return handler(request)


def create_hol_guard_middleware(**kwargs: Any) -> HolGuardMiddleware:
    """Create a HOL Guard middleware instance."""

    return HolGuardMiddleware(**kwargs)
