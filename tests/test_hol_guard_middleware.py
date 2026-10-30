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
    import os

    seen = {}
    handler_calls = []
    expected_executable = os.path.realpath("./trusted/hol-guard")
    monkeypatch.setattr(
        "spoon_ai.middleware.hol_guard.shutil.which",
        lambda executable: "./trusted/hol-guard",
    )

    def fake_guard(argv, **kwargs):
        seen["executable"] = argv[0]
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
    assert seen["executable"] == expected_executable
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
    monkeypatch.setattr(
        "spoon_ai.middleware.hol_guard.shutil.which",
        lambda executable: "/usr/bin/hol-guard",
    )
    result = HolGuardMiddleware(working_directory=missing).wrap_tool_call(
        _request(command="git status"),
        lambda request: handler_calls.append(request)
        or ToolCallResult.from_string("executed"),
    )
    assert result.error == "HOL Guard blocked tool execution: guard_unavailable"
    assert handler_calls == []



def test_worktree_guard_shim_cannot_impersonate_cli(monkeypatch, tmp_path):
    """A nested fake .git cannot hide an executable inside the real checkout."""
    project = tmp_path / "project"
    sandbox = project / "src" / "nested"
    binary = project / ".venv" / "bin" / "hol-guard"
    sandbox.mkdir(parents=True)
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\\nexit 0\\n")
    (project / ".git").mkdir()
    (sandbox / ".git").write_text("fake nested marker")

    monkeypatch.setattr(
        "spoon_ai.middleware.hol_guard.shutil.which",
        lambda _name: str(binary),
    )

    def must_not_launch(*_args, **_kwargs):
        raise AssertionError("Never execute a project-controlled Guard binary")

    monkeypatch.setattr(
        "spoon_ai.middleware.hol_guard.subprocess.run", must_not_launch
    )
    handler_calls = []
    result = HolGuardMiddleware(working_directory=str(sandbox)).wrap_tool_call(
        _request(command="git status"),
        lambda request: handler_calls.append(request)
        or ToolCallResult.from_string("executed"),
    )

    assert result.error == (
        "HOL Guard blocked tool execution: guard_untrusted_executable"
    )
    assert not handler_calls


def test_worktree_guard_symlink_cannot_launder_external_target(monkeypatch, tmp_path):
    """A project-owned PATH symlink remains untrusted when its target is outside."""
    import pytest

    project = tmp_path / "project"
    project.mkdir()
    (project / ".git").mkdir()
    outside = tmp_path / "trusted" / "hol-guard"
    outside.parent.mkdir()
    outside.write_text("#!/bin/sh\\nexit 0\\n")
    project_bin = project / ".bin"
    project_bin.mkdir()
    project_link = project_bin / "hol-guard"
    try:
        project_link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation unavailable")

    monkeypatch.setattr(
        "spoon_ai.middleware.hol_guard.shutil.which",
        lambda _name: str(project_link),
    )

    def must_not_launch(*_args, **_kwargs):
        raise AssertionError("Never execute a project-owned Guard symlink")

    monkeypatch.setattr(
        "spoon_ai.middleware.hol_guard.subprocess.run", must_not_launch
    )
    calls = []
    result = HolGuardMiddleware(working_directory=str(project)).wrap_tool_call(
        _request(command="git status"),
        lambda request: calls.append(request)
        or ToolCallResult.from_string("executed"),
    )

    assert result.error == (
        "HOL Guard blocked tool execution: guard_untrusted_executable"
    )
    assert not calls


def test_missing_guard_binary_with_sandbox_blocks_without_running_handler(monkeypatch, tmp_path):
    """A failed executable lookup must not fall through to the protected handler."""
    calls = []
    monkeypatch.setattr(
        "spoon_ai.middleware.hol_guard.shutil.which",
        lambda executable: None,
    )

    def must_not_launch(*args, **kwargs):
        raise AssertionError("Guard must not launch without a resolved executable")

    monkeypatch.setattr(
        "spoon_ai.middleware.hol_guard.subprocess.run", must_not_launch
    )
    result = HolGuardMiddleware(working_directory=str(tmp_path)).wrap_tool_call(
        _request(command="git status"),
        lambda request: calls.append(request) or ToolCallResult.from_string("executed"),
    )
    assert result.error == "HOL Guard blocked tool execution: guard_unavailable"
    assert calls == []


def test_invalid_working_directory_configuration_is_rejected():
    import pytest

    for bad_directory in ("", " ", 123, [], ".", "./different-project", "relative/worktree"):
        with pytest.raises(ValueError, match="working_directory"):
            HolGuardMiddleware(working_directory=bad_directory)


def test_string_guarded_tools_rejected_before_unprotected_execution():
    """Misconfigured string filters must not turn 'execute' into character names."""
    import pytest

    for invalid in ("execute", b"execute"):
        with pytest.raises(ValueError, match="guarded_tools"):
            HolGuardMiddleware(guarded_tools=invalid)

    # A tuple of bytes/other values previously built a silent nonmatching
    # filter, letting the actual execute tool run without a Guard decision.
    for invalid in (
        (b"execute",),
        (None,),
        (" ",),
        (123,),
        ("execute", None),
        (" execute",),
        ("execute ",),
        ("\texecute",),
        ("execute\n",),
    ):
        with pytest.raises(ValueError, match="guarded_tools"):
            HolGuardMiddleware(guarded_tools=invalid)

    valid = HolGuardMiddleware(guarded_tools=("execute",))
    assert valid.guarded_tools == frozenset({"execute"})


def test_empty_guarded_tool_filters_fail_closed_at_configuration_time():
    """An enabled Guard must not silently become a no-op for an empty filter."""
    import pytest

    for tool_names in ((), [], iter(())):
        with pytest.raises(ValueError, match="at least one tool name"):
            HolGuardMiddleware(guarded_tools=tool_names)


def test_invalid_timeout_configuration_is_rejected():
    """The security gate must never accept an unlimited or invalid deadline."""
    import pytest

    for timeout in (0, -1, float("inf"), float("-inf"), float("nan"), True, "10", None):
        with pytest.raises(ValueError, match="finite positive number"):
            HolGuardMiddleware(timeout_seconds=timeout)

def test_unexpected_guard_adapter_failure_blocks_execution(monkeypatch):
    """Unexpected Guard integration failures must never invoke the tool handler."""
    middleware = HolGuardMiddleware()
    handler_calls = []

    def fail_unexpectedly(command):
        raise RuntimeError("unexpected Guard adapter failure")

    monkeypatch.setattr(middleware, "_evaluate_command", fail_unexpectedly)
    result = middleware.wrap_tool_call(
        _request(command="git status"),
        lambda request: handler_calls.append(request)
        or ToolCallResult.from_string("executed"),
    )

    assert result.error == "HOL Guard blocked tool execution: guard_error"
    assert handler_calls == []


def test_async_pipeline_enforces_guard_before_execute_handler(monkeypatch):
    """The real async agent pipeline must not bypass the sync Guard middleware."""
    import asyncio

    from spoon_ai.middleware.base import MiddlewarePipeline

    middleware = HolGuardMiddleware()
    pipeline = MiddlewarePipeline([middleware])
    calls = []

    async def execute(request):
        calls.append(request)
        return ToolCallResult.from_string("executed")

    monkeypatch.setattr(
        "spoon_ai.middleware.hol_guard.subprocess.run",
        lambda *args, **kwargs: _guard_result(benign=False, action="review"),
    )
    blocked = asyncio.run(
        pipeline.awrap_tool_call(_request(command="rm -rf build"), execute)
    )
    assert blocked.error is not None
    assert "HOL Guard blocked" in blocked.error
    assert not calls

    monkeypatch.setattr(
        "spoon_ai.middleware.hol_guard.subprocess.run",
        lambda *args, **kwargs: _guard_result(benign=True, action="allow"),
    )
    allowed = asyncio.run(
        pipeline.awrap_tool_call(_request(command="git status"), execute)
    )
    assert allowed.error is None
    assert allowed.output == "executed"
    assert len(calls) == 1


def test_async_guard_check_keeps_loop_responsive_and_handler_on_loop(monkeypatch):
    """A slow Guard CLI must not stall the agent's async event loop."""
    import asyncio
    import threading

    middleware = HolGuardMiddleware()
    loop_thread = threading.current_thread()
    started = threading.Event()
    release = threading.Event()
    handler_threads = []

    def guarded_command(_command):
        started.set()
        if not release.wait(timeout=4):
            raise RuntimeError("test did not release Guard")
        return True, "guard_allow"

    monkeypatch.setattr(middleware, "_evaluate_command", guarded_command)

    async def handler(_request):
        handler_threads.append(threading.current_thread())
        return ToolCallResult.from_string("executed")

    async def check():
        pending = asyncio.create_task(
            middleware.awrap_tool_call(_request(command="git status"), handler)
        )
        try:
            observed = await asyncio.wait_for(
                asyncio.to_thread(started.wait, 2), timeout=3
            )
            assert observed
            assert not pending.done()
            await asyncio.sleep(0)
            assert not pending.done()
        finally:
            release.set()
        result = await asyncio.wait_for(pending, timeout=3)
        assert result.output == "executed"
        assert result.error is None

    asyncio.run(check())
    assert handler_threads == [loop_thread]


def test_async_guard_denial_and_evaluation_failure_never_dispatch(monkeypatch):
    """Async inspection has the same fail-closed behavior as the sync path."""
    import asyncio

    middleware = HolGuardMiddleware()
    handler_calls = []

    async def handler(request):
        handler_calls.append(request)
        return ToolCallResult.from_string("executed")

    async def check():
        monkeypatch.setattr(middleware, "_evaluate_command", lambda _: (False, "guard_block"))
        denied = await middleware.awrap_tool_call(
            _request(command="rm -rf build"), handler
        )
        assert denied.error == "HOL Guard blocked tool execution: guard_block"

        def crash(_command):
            raise RuntimeError("unexpected Guard adapter failure")

        monkeypatch.setattr(middleware, "_evaluate_command", crash)
        failed = await middleware.awrap_tool_call(
            _request(command="git status"), handler
        )
        assert failed.error == "HOL Guard blocked tool execution: guard_error"

    asyncio.run(check())
    assert not handler_calls


def test_guard_input_mutation_during_inspection_is_blocked(monkeypatch):
    """The handler must not run a command substituted after Guard inspected it."""
    middleware = HolGuardMiddleware()
    request = _request(command="git status")
    handler_calls = []

    def mutate_original_command(_command):
        request.arguments["command"] = "rm -rf ./workspace"
        return True, "guard_allow"

    monkeypatch.setattr(middleware, "_evaluate_command", mutate_original_command)
    result = middleware.wrap_tool_call(
        request,
        lambda current: handler_calls.append(current) or ToolCallResult.from_string("executed"),
    )
    assert result.error == "HOL Guard blocked tool execution: guard_input_changed"
    assert handler_calls == []


def test_guard_passes_snapshot_to_handler_after_success(monkeypatch):
    """The downstream command must be the same one Guard classified."""
    middleware = HolGuardMiddleware()
    request = _request(command="git status")
    received = []
    monkeypatch.setattr(middleware, "_evaluate_command", lambda command: (True, "guard_allow"))
    result = middleware.wrap_tool_call(
        request,
        lambda current: received.append(current) or ToolCallResult.from_string("executed"),
    )
    assert result.error is None
    assert received[0].arguments == {"command": "git status"}
    assert received[0].arguments is not request.arguments
    request.arguments["command"] = "rm -rf ./workspace"
    assert received[0].arguments["command"] == "git status"


def test_async_guard_input_mutation_during_inspection_is_blocked(monkeypatch):
    """An input substitution during asyncio.to_thread inspection must fail closed."""
    import asyncio

    middleware = HolGuardMiddleware()
    request = _request(command="git status")
    handler_calls = []

    def mutate_original_command(_command):
        request.arguments["command"] = "rm -rf ./workspace"
        return True, "guard_allow"

    async def handler(current):
        handler_calls.append(current)
        return ToolCallResult.from_string("executed")

    monkeypatch.setattr(middleware, "_evaluate_command", mutate_original_command)
    result = asyncio.run(middleware.awrap_tool_call(request, handler))
    assert result.error == "HOL Guard blocked tool execution: guard_input_changed"
    assert handler_calls == []


def test_nested_argument_mutation_during_inspection_is_blocked(monkeypatch):
    """A shallow snapshot could share mutable nested tool args with the caller."""
    middleware = HolGuardMiddleware()
    request = _request(command="git status", options={"mode": "safe"})
    handler_calls = []

    def mutate_nested_argument(_command):
        request.arguments["options"]["mode"] = "unsafe"
        return True, "guard_allow"

    monkeypatch.setattr(middleware, "_evaluate_command", mutate_nested_argument)
    result = middleware.wrap_tool_call(
        request,
        lambda current: handler_calls.append(current) or ToolCallResult.from_string("executed"),
    )
    assert result.error == "HOL Guard blocked tool execution: guard_input_changed"
    assert handler_calls == []


def test_async_nested_argument_mutation_during_inspection_is_blocked(monkeypatch):
    """Async pre-tool evaluation must also freeze nested execution arguments."""
    import asyncio

    middleware = HolGuardMiddleware()
    request = _request(command="git status", options={"mode": "safe"})
    handler_calls = []

    def mutate_nested_argument(_command):
        request.arguments["options"]["mode"] = "unsafe"
        return True, "guard_allow"

    async def handler(current):
        handler_calls.append(current)
        return ToolCallResult.from_string("executed")

    monkeypatch.setattr(middleware, "_evaluate_command", mutate_nested_argument)
    result = asyncio.run(middleware.awrap_tool_call(request, handler))
    assert result.error == "HOL Guard blocked tool execution: guard_input_changed"
    assert handler_calls == []
