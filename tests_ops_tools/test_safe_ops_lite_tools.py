import asyncio
from pathlib import Path

from spoon_ai.tools.ops_tools import build_ops_tools, LocalExec
from spoon_ai.tools.tool_manager import ToolManager


def test_fs_write_and_read_confined(tmp_path: Path):
    tools = build_ops_tools(root_dir=str(tmp_path), allow_exec=["echo "])
    tm = ToolManager(tools)

    r = asyncio.run(tm.execute(name="fs_write", tool_input={"path": "a/b.txt", "content": "hello"}))
    assert not r.error

    r2 = asyncio.run(tm.execute(name="fs_read", tool_input={"path": "a/b.txt"}))
    assert r2.output == "hello"


def test_fs_blocks_escape(tmp_path: Path):
    tools = build_ops_tools(root_dir=str(tmp_path), allow_exec=["echo "])
    tm = ToolManager(tools)

    r = asyncio.run(tm.execute(name="fs_read", tool_input={"path": "../nope.txt"}))
    assert r.error
    assert "escapes root_dir" in r.error


def test_fs_edit_replaces_once(tmp_path: Path):
    tools = build_ops_tools(root_dir=str(tmp_path), allow_exec=["echo "])
    tm = ToolManager(tools)

    asyncio.run(tm.execute(name="fs_write", tool_input={"path": "x.txt", "content": "a a"}))
    r = asyncio.run(tm.execute(name="fs_edit", tool_input={"path": "x.txt", "old": "a", "new": "b"}))
    assert not r.error

    r2 = asyncio.run(tm.execute(name="fs_read", tool_input={"path": "x.txt"}))
    assert r2.output == "b a"


def test_exec_allowlist_denies(tmp_path: Path):
    tm = ToolManager([LocalExec(root_dir=str(tmp_path), allowlist=["echo "])])

    r = asyncio.run(tm.execute(name="local_exec", tool_input={"command": "ls"}))
    assert r.error
    assert "not allowed" in r.error


def test_exec_runs_allowed(tmp_path: Path):
    tm = ToolManager([LocalExec(root_dir=str(tmp_path), allowlist=["echo "])])

    r = asyncio.run(tm.execute(name="local_exec", tool_input={"command": "echo hi"}))
    assert not r.error
    assert r.output["exit_code"] == 0
    assert "hi" in r.output["stdout"]


def test_exec_blocks_cwd_escape(tmp_path: Path):
    tm = ToolManager([LocalExec(root_dir=str(tmp_path), allowlist=["echo "])])

    r = asyncio.run(tm.execute(name="local_exec", tool_input={"command": "echo hi", "cwd": "../"}))
    assert r.error
    assert "escapes root_dir" in r.error
