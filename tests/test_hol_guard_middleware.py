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
