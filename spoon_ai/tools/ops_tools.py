"""ops tools: local exec + filesystem ops confined to a root dir.

Goal: provide a minimal, reusable toolset that can be injected into any agent
(ReAct, graph, etc.) via ToolManager.

Security model:
- All file operations are confined to a configured root_dir.
- local_exec is limited by an allowlist of commands/prefixes, a root_dir cwd,
  and a timeout.

These tools return ToolResult for compatibility with existing ToolManager.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from spoon_ai.tools.base import BaseTool, ToolResult


class SecurityError(Exception):
    pass


def _resolve_under_root(root_dir: Path, user_path: str) -> Path:
    root = root_dir.resolve()
    p = (root / user_path).resolve()
    try:
        # Python 3.9+: raises ValueError if not relative
        p.relative_to(root)
    except ValueError as e:
        raise SecurityError(f"Path escapes root_dir: {user_path}") from e
    return p


def _is_allowed_command(command: str, allowlist: List[str]) -> bool:
    """Allow exact match, prefix match, or simple token-based match.

    - If allowlist item ends with a space, treat it as a prefix rule (e.g. "python ").
    - Otherwise, allow either exact match or matching the first token (e.g. "ls").
    """
    cmd = (command or "").strip()
    if not cmd:
        return False

    first = cmd.split()[0] if cmd.split() else cmd

    for a in allowlist:
        a = (a or "").strip()
        if not a:
            continue
        if a.endswith(" "):
            if cmd.startswith(a):
                return True
        else:
            if cmd == a or first == a or cmd.startswith(a + " "):
                return True

    return False


class LocalExec(BaseTool):
    name: str = "local_exec"
    description: str = (
        "Execute a local shell command within a confined working directory, "
        "restricted by an allowlist and a timeout."
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Command to execute"},
            "timeout_s": {"type": "number", "description": "Timeout seconds", "default": 30},
            "cwd": {"type": "string", "description": "Working directory relative to root", "default": "."},
            "env": {
                "type": "object",
                "description": "Optional environment overrides (key/value).",
                "additionalProperties": {"type": "string"},
            },
        },
        "required": ["command"],
    }

    root_dir: str
    allowlist: List[str]

    async def execute(
        self,
        command: str,
        timeout_s: float = 30,
        cwd: str = ".",
        env: Optional[Dict[str, str]] = None,
    ) -> ToolResult:
        try:
            root = Path(self.root_dir)
            run_cwd = _resolve_under_root(root, cwd)

            if not _is_allowed_command(command, self.allowlist):
                return ToolResult(error=f"Command not allowed: {command}")

            merged_env = os.environ.copy()
            if env:
                merged_env.update({str(k): str(v) for k, v in env.items()})

            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=str(run_cwd),
                env=merged_env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
            except asyncio.TimeoutError:
                proc.kill()
                try:
                    await proc.wait()
                except Exception:
                    pass
                return ToolResult(error=f"Command timed out after {timeout_s}s")

            stdout = (stdout_b or b"").decode("utf-8", errors="replace")
            stderr = (stderr_b or b"").decode("utf-8", errors="replace")

            return ToolResult(
                output={
                    "exit_code": proc.returncode,
                    "stdout": stdout,
                    "stderr": stderr,
                }
            )
        except SecurityError as e:
            return ToolResult(error=str(e))
        except Exception as e:
            return ToolResult(error=f"local_exec failed: {e}")


class FSRead(BaseTool):
    name: str = "fs_read"
    description: str = "Read a text file from within a confined root directory."
    parameters: dict = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path relative to root"},
            "max_bytes": {"type": "integer", "default": 200000},
        },
        "required": ["path"],
    }

    root_dir: str

    async def execute(self, path: str, max_bytes: int = 200_000) -> ToolResult:
        try:
            p = _resolve_under_root(Path(self.root_dir), path)
            data = p.read_bytes()
            if len(data) > max_bytes:
                data = data[:max_bytes]
            return ToolResult(output=data.decode("utf-8", errors="replace"))
        except SecurityError as e:
            return ToolResult(error=str(e))
        except FileNotFoundError:
            return ToolResult(error=f"File not found: {path}")
        except Exception as e:
            return ToolResult(error=f"fs_read failed: {e}")


class FSWrite(BaseTool):
    name: str = "fs_write"
    description: str = "Write (overwrite) a text file within a confined root directory."
    parameters: dict = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path relative to root"},
            "content": {"type": "string", "description": "Content to write"},
            "mkdirs": {"type": "boolean", "default": True},
        },
        "required": ["path", "content"],
    }

    root_dir: str

    async def execute(self, path: str, content: str, mkdirs: bool = True) -> ToolResult:
        try:
            p = _resolve_under_root(Path(self.root_dir), path)
            if mkdirs:
                p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return ToolResult(output={"written": str(p)})
        except SecurityError as e:
            return ToolResult(error=str(e))
        except Exception as e:
            return ToolResult(error=f"fs_write failed: {e}")


class FSEdit(BaseTool):
    name: str = "fs_edit"
    description: str = "Replace exact text in a file within a confined root directory."
    parameters: dict = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "old": {"type": "string"},
            "new": {"type": "string"},
        },
        "required": ["path", "old", "new"],
    }

    root_dir: str

    async def execute(self, path: str, old: str, new: str) -> ToolResult:
        try:
            p = _resolve_under_root(Path(self.root_dir), path)
            text = p.read_text(encoding="utf-8")
            if old not in text:
                return ToolResult(error="Old text not found")
            p.write_text(text.replace(old, new, 1), encoding="utf-8")
            return ToolResult(output={"edited": str(p)})
        except SecurityError as e:
            return ToolResult(error=str(e))
        except Exception as e:
            return ToolResult(error=f"fs_edit failed: {e}")


def build_ops_tools(
    *,
    root_dir: str,
    allow_exec: Optional[List[str]] = None,
) -> List[BaseTool]:
    """Build the ops-lite toolset for injection into ToolManager."""
    allow_exec = allow_exec or ["python", "pytest", "echo", "ls", "cat"]
    return [
        LocalExec(root_dir=root_dir, allowlist=allow_exec),
        FSRead(root_dir=root_dir),
        FSWrite(root_dir=root_dir),
        FSEdit(root_dir=root_dir),
        FSList(root_dir=root_dir),
        FSMkdir(root_dir=root_dir),
        FSExists(root_dir=root_dir),
    ]


class FSList(BaseTool):
    name: str = "fs_list"
    description: str = "List files/directories under a path within the confined root directory."
    parameters: dict = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory path relative to root", "default": "."},
            "recursive": {"type": "boolean", "default": False},
            "max_entries": {"type": "integer", "default": 200},
        },
        "required": [],
    }

    root_dir: str

    async def execute(self, path: str = ".", recursive: bool = False, max_entries: int = 200) -> ToolResult:
        try:
            base = _resolve_under_root(Path(self.root_dir), path)
            if not base.exists():
                return ToolResult(error=f"Path not found: {path}")
            if not base.is_dir():
                return ToolResult(error=f"Not a directory: {path}")

            items = []
            it = base.rglob("*") if recursive else base.iterdir()
            for p in it:
                rel = str(p.relative_to(Path(self.root_dir).resolve()))
                items.append(
                    {
                        "path": rel,
                        "type": "dir" if p.is_dir() else "file",
                    }
                )
                if len(items) >= max_entries:
                    break

            return ToolResult(output={"entries": items, "truncated": len(items) >= max_entries})
        except SecurityError as e:
            return ToolResult(error=str(e))
        except Exception as e:
            return ToolResult(error=f"fs_list failed: {e}")


class FSMkdir(BaseTool):
    name: str = "fs_mkdir"
    description: str = "Create a directory (and parents) within the confined root directory."
    parameters: dict = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory path relative to root"},
            "parents": {"type": "boolean", "default": True},
            "exist_ok": {"type": "boolean", "default": True},
        },
        "required": ["path"],
    }

    root_dir: str

    async def execute(self, path: str, parents: bool = True, exist_ok: bool = True) -> ToolResult:
        try:
            p = _resolve_under_root(Path(self.root_dir), path)
            p.mkdir(parents=parents, exist_ok=exist_ok)
            return ToolResult(output={"created": str(p)})
        except SecurityError as e:
            return ToolResult(error=str(e))
        except Exception as e:
            return ToolResult(error=f"fs_mkdir failed: {e}")


class FSExists(BaseTool):
    name: str = "fs_exists"
    description: str = "Check whether a path exists within the confined root directory."
    parameters: dict = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path relative to root"},
        },
        "required": ["path"],
    }

    root_dir: str

    async def execute(self, path: str) -> ToolResult:
        try:
            p = _resolve_under_root(Path(self.root_dir), path)
            return ToolResult(output={"exists": p.exists(), "is_dir": p.is_dir(), "is_file": p.is_file()})
        except SecurityError as e:
            return ToolResult(error=str(e))
        except Exception as e:
            return ToolResult(error=f"fs_exists failed: {e}")
