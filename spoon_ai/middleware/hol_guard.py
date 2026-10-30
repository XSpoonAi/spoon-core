"""HOL Guard middleware for command-bearing tool calls."""

from __future__ import annotations

import asyncio
import copy
import inspect
import json
import math
import os
import shutil
import subprocess
from dataclasses import replace
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from .base import AgentMiddleware, ToolCallRequest, ToolCallResult


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject contradictory JSON members instead of taking the last verdict."""
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError("duplicate JSON key in Guard response")
        payload[key] = value
    return payload


def _worktree_root(cwd: str) -> str:
    """Locate the outermost Git checkout marker, including linked worktrees."""
    current = os.path.realpath(cwd)
    root = current
    while True:
        if os.path.lexists(os.path.join(current, ".git")):
            root = current
        parent = os.path.dirname(current)
        if parent == current:
            return root
        current = parent


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
        working_directory: str | None = None,
    ) -> None:
        super().__init__()
        # A non-finite timeout could let a stalled security check hang forever.
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be a finite positive number")
        self.executable = executable
        self.timeout_seconds = timeout_seconds
        # A bare string is iterable character-by-character. Without this guard,
        # guarded_tools="execute" silently matches only individual letters and
        # lets the actual execute tool run without its security check.
        if isinstance(guarded_tools, (str, bytes)):
            raise ValueError("guarded_tools must be an iterable of tool names")
        tool_names = tuple(guarded_tools)
        # An empty filter would silently bypass every tool, even with Guard enabled.
        if not tool_names:
            raise ValueError("guarded_tools must contain at least one tool name")
        if any(
            not isinstance(name, str)
            or not name.strip()
            or name != name.strip()
            for name in tool_names
        ):
            # Whitespace-padded tool names never match the real tool name.
            # Reject them rather than silently skipping the Guard check.
            raise ValueError("guarded_tools must contain nonempty, unpadded tool names")
        self.guarded_tools = frozenset(tool_names)
        if working_directory is not None and (
            not isinstance(working_directory, str)
            or not working_directory.strip()
            or not os.path.isabs(working_directory)
        ):
            raise ValueError("working_directory must be an absolute path or None")
        # Match the effective local sandbox root, not an unrelated launcher cwd.
        # This cannot emulate a remote/container filesystem.
        self.working_directory = working_directory

    def _evaluate_command(self, command: str) -> tuple[bool, str]:
        try:
            # Bind the Guard binary before changing into the tool's directory.
            # A relative PATH entry must not select a project-local executable.
            executable = self.executable
            if self.working_directory is None:
                # Without an explicit cwd, subprocess inherits the project's
                # launcher directory. Relative or repo-owned PATH entries can
                # select a forged Guard that returns an allow verdict.
                launcher_root = _worktree_root(os.getcwd())
                path_entries = os.environ.get("PATH", os.defpath).split(os.pathsep)
                try:
                    if any(
                        not os.path.isabs(entry)
                        or os.path.commonpath(
                            (os.path.realpath(entry), launcher_root)
                        ) == launcher_root
                        for entry in path_entries
                    ):
                        return False, "guard_untrusted_executable"
                    if (
                        os.path.isabs(executable)
                        and os.path.commonpath(
                            (os.path.realpath(executable), launcher_root)
                        ) == launcher_root
                    ) or (
                        not os.path.isabs(executable)
                        and (
                            os.path.sep in executable
                            or (os.path.altsep and os.path.altsep in executable)
                        )
                    ):
                        return False, "guard_untrusted_executable"
                except ValueError:
                    return False, "guard_untrusted_executable"
            if self.working_directory is not None:
                resolved = shutil.which(executable)
                if not resolved:
                    return False, "guard_unavailable"
                selected_executable = os.path.abspath(resolved)
                executable = os.path.realpath(selected_executable)
                # A project-controlled PATH entry can impersonate Guard and
                # return forged benign JSON. Check the symlink and its target.
                roots = {
                    _worktree_root(self.working_directory),
                    _worktree_root(os.getcwd()),
                }
                try:
                    if any(
                        os.path.commonpath((candidate, root)) == root
                        for root in roots
                        for candidate in (selected_executable, executable)
                    ):
                        return False, "guard_untrusted_executable"
                except ValueError:
                    return False, "guard_untrusted_executable"
            # Keep arbitrary command text positional, including a leading `-`.
            run_kwargs = (
                {"cwd": self.working_directory}
                if self.working_directory is not None else {}
            )
            result = subprocess.run(
                [executable, "command", "test", "--json", "--", command],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
                **run_kwargs,
            )
        except FileNotFoundError:
            return False, "guard_unavailable"
        except subprocess.TimeoutExpired:
            return False, "guard_timeout"
        except (OSError, UnicodeError, ValueError, TypeError):
            return False, "guard_error"

        if result.returncode != 0:
            return False, "guard_error"

        try:
            payload: Any = json.loads(result.stdout, object_pairs_hook=_reject_duplicate_keys)
        except (json.JSONDecodeError, TypeError, ValueError, RecursionError):
            return False, "guard_invalid_output"

        if not isinstance(payload, dict):
            return False, "guard_invalid_output"
        classification = payload.get("classification")
        if not isinstance(classification, dict):
            return False, "guard_invalid_output"

        # The classifier cannot override an explicit Guard stop signal.
        decision = payload.get("decision")
        if ("continue" in payload and payload["continue"] is not True) or (
            "decision" in payload
            and (not isinstance(decision, str) or decision.strip().lower() != "allow")
        ):
            return False, "guard_block"

        # Optional hook-level decisions must not contradict a benign classifier.
        policy_action = payload.get("policy_action")
        if "policy_action" in payload and (
            not isinstance(policy_action, str) or policy_action not in {"allow", "not_run"}
        ):
            return False, "guard_block"
        hook_output = payload.get("hookSpecificOutput")
        if "hookSpecificOutput" in payload and (
            not isinstance(hook_output, dict)
            or hook_output.get("permissionDecision") != "allow"
            or hook_output.get("hookEventName") != "PreToolUse"
        ):
            return False, "guard_block"

        if (
            classification.get("explicitly_benign") is True
            and payload.get("minimum_action") == "allow"
        ):
            return True, "guard_allow"
        return False, "guard_block"

    @staticmethod
    def _guarded_request_unchanged(
        request: ToolCallRequest, guarded_request: ToolCallRequest
    ) -> bool:
        """Refuse a tool call if its command or other arguments changed during inspection."""
        try:
            return (
                request.tool_name == guarded_request.tool_name
                and isinstance(request.arguments, Mapping)
                and dict(request.arguments) == guarded_request.arguments
            )
        except Exception:
            return False

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolCallResult],
    ) -> ToolCallResult:
        if request.tool_name not in self.guarded_tools:
            return handler(request)

        if not isinstance(request.arguments, Mapping):
            return ToolCallResult.from_error(
                "HOL Guard blocked tool execution: invalid tool arguments"
            )

        try:
            # Inspect a deep copy; concurrent callers may mutate nested arguments
            # while Guard is evaluating the original request.
            guarded_request = replace(
                request, arguments=copy.deepcopy(dict(request.arguments))
            )
        except Exception:
            return ToolCallResult.from_error(
                "HOL Guard blocked tool execution: invalid tool arguments"
            )
        command = guarded_request.arguments.get("command")
        if not isinstance(command, str) or not command.strip():
            return ToolCallResult.from_error(
                "HOL Guard blocked tool execution: missing command"
            )

        try:
            allowed, reason = self._evaluate_command(command)
        except Exception:
            # An unexpected failure in the security adapter must not pass
            # the command on to the execution handler.
            return ToolCallResult.from_error(
                "HOL Guard blocked tool execution: guard_error"
            )
        if not allowed:
            return ToolCallResult.from_error(
                f"HOL Guard blocked tool execution: {reason}"
            )
        if not self._guarded_request_unchanged(request, guarded_request):
            return ToolCallResult.from_error(
                "HOL Guard blocked tool execution: guard_input_changed"
            )

        return handler(guarded_request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolCallResult],
    ) -> ToolCallResult:
        """Inspect Guard off the event loop, then dispatch the handler on-loop.

        The base middleware async adapter calls the synchronous wrapper inline.
        A slow Guard CLI would therefore otherwise block every other agent task
        for up to the configured security-check timeout.
        """
        if request.tool_name not in self.guarded_tools:
            result = handler(request)
            return await result if inspect.isawaitable(result) else result

        if not isinstance(request.arguments, Mapping):
            return ToolCallResult.from_error(
                "HOL Guard blocked tool execution: invalid tool arguments"
            )

        try:
            # Inspect a deep copy; concurrent callers may mutate nested arguments
            # while Guard is evaluating the original request.
            guarded_request = replace(
                request, arguments=copy.deepcopy(dict(request.arguments))
            )
        except Exception:
            return ToolCallResult.from_error(
                "HOL Guard blocked tool execution: invalid tool arguments"
            )
        command = guarded_request.arguments.get("command")
        if not isinstance(command, str) or not command.strip():
            return ToolCallResult.from_error(
                "HOL Guard blocked tool execution: missing command"
            )

        try:
            allowed, reason = await asyncio.to_thread(self._evaluate_command, command)
        except Exception:
            return ToolCallResult.from_error(
                "HOL Guard blocked tool execution: guard_error"
            )
        if not allowed:
            return ToolCallResult.from_error(
                f"HOL Guard blocked tool execution: {reason}"
            )
        if not self._guarded_request_unchanged(request, guarded_request):
            return ToolCallResult.from_error(
                "HOL Guard blocked tool execution: guard_input_changed"
            )

        # Keep downstream tool callbacks on the event-loop thread; only the
        # subprocess-bound inspection is dispatched to a worker.
        result = handler(guarded_request)
        return await result if inspect.isawaitable(result) else result


def create_hol_guard_middleware(**kwargs: Any) -> HolGuardMiddleware:
    """Create a HOL Guard middleware instance."""

    return HolGuardMiddleware(**kwargs)
