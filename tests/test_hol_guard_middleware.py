import subprocess
from types import SimpleNamespace

from spoon_ai.middleware.base import ToolCallRequest, ToolCallResult
from spoon_ai.middleware.hol_guard import HolGuardMiddleware


def _request(tool_name: str = "execute", **arguments) -> ToolCallRequest:
    return ToolCallRequest(
        tool_name=tool_name,
        arguments=arguments or {"command": "git status"},
        tool_call_id="call-1",
    )


def _guard_result(*, benign: bool, action: str, returncode: int = 0):
    import json

    return SimpleNamespace(
        returncode=returncode,
        stdout=json.dumps(
            {
                "classification": {"explicitly_benign": benign},
                "minimum_action": action,
            }
        ),
        stderr="",
    )


def test_explicit_benign_allow_reaches_handler(monkeypatch):
    middleware = HolGuardMiddleware()
    calls = []
    monkeypatch.setattr(
        "spoon_ai.middleware.hol_guard.subprocess.run",
        lambda *args, **kwargs: _guard_result(benign=True, action="allow"),
    )

    def handler(request):
        calls.append(request)
        return ToolCallResult.from_string("executed")

    result = middleware.wrap_tool_call(_request(command="git status"), handler)

    assert result.output == "executed"
    assert result.error is None
    assert len(calls) == 1


def test_inspection_only_benign_classification_preserves_safe_command(monkeypatch):
    """The command-test CLI may report not_run without running runtime policy."""
    import json

    middleware = HolGuardMiddleware()
    payload = {
        "policy_action": "not_run",
        "minimum_action": "allow",
        "classification": {"explicitly_benign": True},
    }
    monkeypatch.setattr(
        "spoon_ai.middleware.hol_guard.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout=json.dumps(payload), stderr=""
        ),
    )
    calls = []
    result = middleware.wrap_tool_call(
        _request(command="git status"),
        lambda request: calls.append(request) or ToolCallResult.from_string("executed"),
    )
    assert result.error is None
    assert result.output == "executed"
    assert len(calls) == 1

    for contradictory in (
        {"minimum_action": "review"},
        {"classification": {"explicitly_benign": False}},
        {"policy_action": "warn"},
        {"continue": False},
        {"hookSpecificOutput": {"permissionDecision": "allow", "hookEventName": "PostToolUse"}},
    ):
        calls.clear()
        unsafe = {**payload, **contradictory}
        monkeypatch.setattr(
            "spoon_ai.middleware.hol_guard.subprocess.run",
            lambda *args, **kwargs: SimpleNamespace(
                returncode=0, stdout=json.dumps(unsafe), stderr=""
            ),
        )
        stopped = middleware.wrap_tool_call(
            _request(command="git status"),
            lambda request: calls.append(request) or ToolCallResult.from_string("executed"),
        )
        assert stopped.error == "HOL Guard blocked tool execution: guard_block"
        assert calls == []


def test_review_blocks_before_handler(monkeypatch):
    middleware = HolGuardMiddleware()
    calls = []
    monkeypatch.setattr(
        "spoon_ai.middleware.hol_guard.subprocess.run",
        lambda *args, **kwargs: _guard_result(benign=False, action="review"),
    )

    result = middleware.wrap_tool_call(
        _request(command="rm -rf build"),
        lambda request: calls.append(request) or ToolCallResult.from_string("executed"),
    )

    assert result.error == "HOL Guard blocked tool execution: guard_block"
    assert calls == []


def test_implicit_allow_blocks_before_handler(monkeypatch):
    middleware = HolGuardMiddleware()
    calls = []
    monkeypatch.setattr(
        "spoon_ai.middleware.hol_guard.subprocess.run",
        lambda *args, **kwargs: _guard_result(benign=False, action="allow"),
    )

    result = middleware.wrap_tool_call(
        _request(command="echo hello"),
        lambda request: calls.append(request) or ToolCallResult.from_string("executed"),
    )

    assert result.error == "HOL Guard blocked tool execution: guard_block"
    assert calls == []


def test_explicit_guard_stop_overrides_benign_classification(monkeypatch):
    import json

    middleware = HolGuardMiddleware()
    for stop_signal in ({"continue": False}, {"continue": "false"}, {"continue": None}, {"decision": "deny"}, {"decision": "review"}, {"decision": None}):
        calls = []
        payload = {
            "minimum_action": "allow",
            "classification": {"explicitly_benign": True},
            **stop_signal,
        }
        monkeypatch.setattr(
            "spoon_ai.middleware.hol_guard.subprocess.run",
            lambda *args, **kwargs: SimpleNamespace(
                returncode=0, stdout=json.dumps(payload), stderr=""
            ),
        )
        result = middleware.wrap_tool_call(
            _request(command="git status"),
            lambda request: calls.append(request) or ToolCallResult.from_string("executed"),
        )
        assert result.error == "HOL Guard blocked tool execution: guard_block"
        assert calls == []


def test_conflicting_hook_verdict_blocks_even_when_classifier_says_allow(monkeypatch):
    import json

    middleware = HolGuardMiddleware()
    for contradictory in (
        {"policy_action": "block"},
        {"policy_action": None},
        {"policy_action": []},
        {"policy_action": {}},
        {"hookSpecificOutput": None},
        {"hookSpecificOutput": {"permissionDecision": "deny"}},
        {"hookSpecificOutput": "malformed"},
    ):
        calls = []
        payload = {
            "classification": {"explicitly_benign": True},
            "minimum_action": "allow",
            **contradictory,
        }
        monkeypatch.setattr(
            "spoon_ai.middleware.hol_guard.subprocess.run",
            lambda *args, **kwargs: SimpleNamespace(
                returncode=0, stdout=json.dumps(payload), stderr=""
            ),
        )
        result = middleware.wrap_tool_call(
            _request(command="git status"),
            lambda request: calls.append(request) or ToolCallResult.from_string("executed"),
        )
        assert result.error == "HOL Guard blocked tool execution: guard_block"
        assert calls == []


def test_nonzero_guard_exit_blocks(monkeypatch):
    middleware = HolGuardMiddleware()
    calls = []
    monkeypatch.setattr(
        "spoon_ai.middleware.hol_guard.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="", stderr="error"),
    )

    result = middleware.wrap_tool_call(
        _request(),
        lambda request: calls.append(request) or ToolCallResult.from_string("executed"),
    )

    assert result.error == "HOL Guard blocked tool execution: guard_error"
    assert calls == []


def test_guard_timeout_stops_handler(monkeypatch):
    """A stalled Guard subprocess must never reach the protected tool."""
    middleware = HolGuardMiddleware(timeout_seconds=0.1)
    calls = []

    def stalled(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr("spoon_ai.middleware.hol_guard.subprocess.run", stalled)
    result = middleware.wrap_tool_call(
        _request(command="rm -rf build"),
        lambda request: calls.append(request) or ToolCallResult.from_string("executed"),
    )

    assert result.error == "HOL Guard blocked tool execution: guard_timeout"
    assert calls == []


def test_malformed_guard_output_blocks(monkeypatch):
    middleware = HolGuardMiddleware()
    calls = []
    monkeypatch.setattr(
        "spoon_ai.middleware.hol_guard.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="{", stderr=""),
    )

    result = middleware.wrap_tool_call(
        _request(),
        lambda request: calls.append(request) or ToolCallResult.from_string("executed"),
    )

    assert result.error == "HOL Guard blocked tool execution: guard_invalid_output"
    assert calls == []


def test_guard_launch_failure_blocks(monkeypatch):
    middleware = HolGuardMiddleware()
    calls = []

    def fail(*args, **kwargs):
        raise PermissionError("not executable")

    monkeypatch.setattr("spoon_ai.middleware.hol_guard.subprocess.run", fail)
    result = middleware.wrap_tool_call(
        _request(),
        lambda request: calls.append(request) or ToolCallResult.from_string("executed"),
    )

    assert result.error == "HOL Guard blocked tool execution: guard_error"
    assert calls == []


def test_missing_command_blocks_without_guard_or_handler(monkeypatch):
    middleware = HolGuardMiddleware()
    calls = []
    guard_calls = []
    monkeypatch.setattr(
        "spoon_ai.middleware.hol_guard.subprocess.run",
        lambda *args, **kwargs: guard_calls.append((args, kwargs)),
    )

    result = middleware.wrap_tool_call(
        _request(command=""),
        lambda request: calls.append(request) or ToolCallResult.from_string("executed"),
    )

    assert result.error == "HOL Guard blocked tool execution: missing command"
    assert guard_calls == []
    assert calls == []


def test_unguarded_tool_passes_through_without_guard(monkeypatch):
    middleware = HolGuardMiddleware()
    guard_calls = []
    monkeypatch.setattr(
        "spoon_ai.middleware.hol_guard.subprocess.run",
        lambda *args, **kwargs: guard_calls.append((args, kwargs)),
    )

    result = middleware.wrap_tool_call(
        _request(tool_name="read_file", file_path="/tmp/a"),
        lambda request: ToolCallResult.from_string("read"),
    )

    assert result.output == "read"
    assert guard_calls == []


def test_command_arguments_preserve_leading_dashes_and_exact_command_text(monkeypatch):
    """Regression: user command text must not be parsed as Guard CLI switches."""
    for command in ("--help", "-n", "echo 'quoted value'", "printf first\nprintf second"):
        actual_calls = []
        tool_calls = []

        def capture_subprocess(argv, **kwargs):
            actual_calls.append((argv, kwargs))
            return _guard_result(benign=True, action="allow")

        monkeypatch.setattr("spoon_ai.middleware.hol_guard.subprocess.run", capture_subprocess)
        result = HolGuardMiddleware().wrap_tool_call(
            _request(command=command),
            lambda request: tool_calls.append(request) or ToolCallResult.from_string("executed"),
        )
        assert result.error is None
        assert len(tool_calls) == 1
        assert actual_calls == [(
            ["hol-guard", "command", "test", "--json", "--", command],
            {"capture_output": True, "text": True, "timeout": 10.0, "check": False},
        )]

def test_malformed_tool_arguments_fail_closed(monkeypatch):
    """A malformed guarded request must never reach Guard or the tool handler."""
    guard_calls = []
    handler_calls = []
    monkeypatch.setattr(
        "spoon_ai.middleware.hol_guard.subprocess.run",
        lambda *args, **kwargs: guard_calls.append((args, kwargs)),
    )
    middleware = HolGuardMiddleware()
    for arguments in (None, "git status", ["git status"]):
        request = ToolCallRequest(
            tool_name="execute", arguments=arguments, tool_call_id="malformed-1",
        )
        result = middleware.wrap_tool_call(
            request,
            lambda current: handler_calls.append(current) or ToolCallResult.from_string("executed"),
        )
        assert result.error == "HOL Guard blocked tool execution: invalid tool arguments"
    assert guard_calls == []
    assert handler_calls == []


def test_guard_stdout_rejects_extra_verdicts_and_diagnostics(monkeypatch):
    """A single valid allow must not survive additional stdout content."""
    import json

    allow = json.dumps({
        "classification": {"explicitly_benign": True},
        "minimum_action": "allow",
    })
    middleware = HolGuardMiddleware()
    handler_calls = []
    for raw_stdout in (allow + "\n" + allow, "status: ok\n" + allow, allow + "\nstatus: ok"):
        monkeypatch.setattr(
            "spoon_ai.middleware.hol_guard.subprocess.run",
            lambda *args, **kwargs: SimpleNamespace(
                returncode=0, stdout=raw_stdout, stderr=""
            ),
        )
        result = middleware.wrap_tool_call(
            _request(command="git status"),
            lambda request: handler_calls.append(request) or ToolCallResult.from_string("executed"),
        )
        assert result.error == "HOL Guard blocked tool execution: guard_invalid_output"
    assert handler_calls == []

def test_subprocess_value_error_blocks_before_handler(monkeypatch):
    """Invalid subprocess argv must not accidentally bypass Guard."""
    middleware = HolGuardMiddleware()
    calls = []

    def invalid_argv(*args, **kwargs):
        raise ValueError("embedded null byte")

    monkeypatch.setattr("spoon_ai.middleware.hol_guard.subprocess.run", invalid_argv)
    result = middleware.wrap_tool_call(
        _request(command="git status"),
        lambda request: calls.append(request) or ToolCallResult.from_string("executed"),
    )

    assert result.error == "HOL Guard blocked tool execution: guard_error"
    assert calls == []


def test_duplicate_guard_json_keys_block(monkeypatch):
    """Even a final allow cannot override a conflicting earlier JSON member."""
    middleware = HolGuardMiddleware()
    handler_calls = []
    for stdout in (
        '{"classification":{"explicitly_benign":true},"minimum_action":"deny","minimum_action":"allow"}',
        '{"classification":{"explicitly_benign":false,"explicitly_benign":true},"minimum_action":"allow"}',
        '{"classification":{"explicitly_benign":true},"continue":false,"continue":true,"minimum_action":"allow"}',
    ):
        monkeypatch.setattr(
            "spoon_ai.middleware.hol_guard.subprocess.run",
            lambda *args, **kwargs: SimpleNamespace(
                returncode=0, stdout=stdout, stderr=""
            ),
        )
        result = middleware.wrap_tool_call(
            _request(command="git status"),
            lambda request: handler_calls.append(request)
            or ToolCallResult.from_string("executed"),
        )
        assert result.error == "HOL Guard blocked tool execution: guard_invalid_output"
    assert handler_calls == []


def test_overdeep_guard_verdict_blocks_before_handler(monkeypatch):
    """Malformed deeply nested Guard output must not escape the tool gate."""
    middleware = HolGuardMiddleware()
    calls = []
    stdout = "[" * 20000 + "]" * 20000
    monkeypatch.setattr(
        "spoon_ai.middleware.hol_guard.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout=stdout, stderr=""
        ),
    )

    result = middleware.wrap_tool_call(
        _request(command="git status"),
        lambda request: calls.append(request) or ToolCallResult.from_string("executed"),
    )

    assert result.error == "HOL Guard blocked tool execution: guard_invalid_output"
    assert calls == []


def test_guard_uses_configured_local_sandbox_working_directory(monkeypatch, tmp_path):
    """The classifier must inspect local paths relative to the command sandbox."""
    seen = {}
    handler_calls = []

    def fake_guard(argv, **kwargs):
        seen["cwd"] = kwargs.get("cwd")
        return _guard_result(benign=True, action="allow")

    monkeypatch.setattr("spoon_ai.middleware.hol_guard.subprocess.run", fake_guard)
    middleware = HolGuardMiddleware(working_directory=str(tmp_path))
    result = middleware.wrap_tool_call(
        _request(command="git status"),
        lambda request: handler_calls.append(request)
        or ToolCallResult.from_string("executed"),
    )

    assert result.error is None
    assert result.output == "executed"
    assert seen["cwd"] == str(tmp_path)
    assert len(handler_calls) == 1


def test_invalid_sandbox_directory_blocks_before_handler(monkeypatch, tmp_path):
    """A failed Guard launch in the configured directory cannot run the tool."""
    missing = str(tmp_path / "missing-sandbox")
    handler_calls = []

    def fail_in_directory(argv, **kwargs):
        assert kwargs.get("cwd") == missing
        raise FileNotFoundError("sandbox directory not found")

    monkeypatch.setattr(
        "spoon_ai.middleware.hol_guard.subprocess.run", fail_in_directory
    )
    result = HolGuardMiddleware(working_directory=missing).wrap_tool_call(
        _request(command="git status"),
        lambda request: handler_calls.append(request)
        or ToolCallResult.from_string("executed"),
    )
    assert result.error == "HOL Guard blocked tool execution: guard_unavailable"
    assert handler_calls == []


def test_invalid_working_directory_configuration_is_rejected():
    import pytest

    for bad_directory in ("", " ", 123, []):
        with pytest.raises(ValueError, match="working_directory"):
            HolGuardMiddleware(working_directory=bad_directory)
