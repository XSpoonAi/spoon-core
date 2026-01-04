"""Filesystem Middleware - 7 Built-in Tools for File Operations.

Provides filesystem tools to agents:
1. ls - List files in directory
2. read_file - Read file content
3. write_file - Write new file
4. edit_file - Edit existing file (string replacement)
5. glob - Find files by pattern
6. grep - Search content in files
7. execute - Run shell commands (if backend supports)

Compatible with LangChain DeepAgents filesystem middleware interface.

Usage:
    from spoon_ai.middleware.filesystem import FilesystemMiddleware
    from spoon_ai.backends import create_state_backend

    backend, runtime = create_state_backend()
    middleware = FilesystemMiddleware(backend=backend)

    agent = ToolCallAgent(
        middleware=[middleware],
        ...
    )
"""

import os
import re
import asyncio
import subprocess
from typing import Any, Dict, List, Optional, Callable, Literal
from dataclasses import dataclass

from spoon_ai.middleware.base import (
    AgentMiddleware,
    AgentRuntime,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
    ToolCallResult,
)
from spoon_ai.backends.protocol import (
    BackendProtocol,
    BackendRuntime,
    SandboxBackendProtocol,
    WriteResult,
    EditResult,
    ExecuteResponse,
)
from spoon_ai.backends.state import StateBackend
from spoon_ai.backends.utils import (
    format_content_with_line_numbers,
    format_grep_results,
    truncate_if_too_long,
    GrepMatch,
)
from spoon_ai.tools.base import BaseTool


# ============================================================================
# Constants
# ============================================================================

DEFAULT_READ_OFFSET = 0
DEFAULT_READ_LIMIT = 500
MAX_LINE_LENGTH = 2000
TOOL_TOKEN_LIMIT = 20000


# ============================================================================
# Tool Descriptions
# ============================================================================

LIST_FILES_DESCRIPTION = """Lists all files in the filesystem, filtering by directory.

Usage:
- The path parameter must be an absolute path, not a relative path
- Returns a list of all files in the specified directory
- Use this tool to explore the file system before reading or editing files
- You should almost ALWAYS use this tool before using read_file or edit_file"""

READ_FILE_DESCRIPTION = """Reads a file from the filesystem.

Usage:
- The file_path parameter must be an absolute path
- By default, reads up to 500 lines starting from the beginning
- For large files, use pagination with offset and limit parameters:
  - First scan: read_file(path, limit=100) to see file structure
  - Read more: read_file(path, offset=100, limit=200) for next section
- Results are returned with line numbers (cat -n format)
- ALWAYS read a file before editing it"""

WRITE_FILE_DESCRIPTION = """Writes to a new file in the filesystem.

Usage:
- The file_path parameter must be an absolute path
- Creates a new file with the provided content
- Will error if file already exists - use edit_file for existing files
- Prefer editing existing files over creating new ones"""

EDIT_FILE_DESCRIPTION = """Performs exact string replacements in files.

Usage:
- You must read the file before editing it
- Preserve exact indentation (tabs/spaces) from the read output
- The edit will FAIL if old_string is not unique in the file
- Use replace_all=True to replace all occurrences
- Provide more context to make old_string unique if needed"""

GLOB_DESCRIPTION = """Find files matching a glob pattern.

Usage:
- Supports standard glob patterns: * (any chars), ** (any dirs), ? (single char)
- Patterns can be absolute (starting with /) or relative

Examples:
- **/*.py - Find all Python files
- *.txt - Find all text files in root
- /subdir/**/*.md - Find markdown files under /subdir"""

GREP_DESCRIPTION = """Search for a pattern in files.

Usage:
- pattern: Text to search for (literal string)
- path: Directory to search in (default: /)
- glob: Filter files by pattern (e.g., *.py)
- output_mode: files_with_matches (default), content, or count

Examples:
- grep(pattern="TODO")
- grep(pattern="import", glob="*.py")
- grep(pattern="error", output_mode="content")"""

EXECUTE_DESCRIPTION = """Executes a shell command in the sandbox environment.

Usage:
- The command parameter is required
- Commands run in an isolated sandbox
- Returns combined stdout/stderr with exit code
- Use ; or && to chain commands (no newlines)
- Use absolute paths, avoid cd
- Do NOT use grep/find/cat - use the filesystem tools instead

Examples:
- execute(command="pytest /tests")
- execute(command="npm install && npm test")"""


# ============================================================================
# System Prompts
# ============================================================================

FILESYSTEM_SYSTEM_PROMPT = """## Filesystem Tools

You have access to a filesystem with these tools:
- ls: list files in a directory (requires absolute path)
- read_file: read a file from the filesystem
- write_file: write to a new file in the filesystem
- edit_file: edit a file using string replacement
- glob: find files matching a pattern (e.g., "**/*.py")
- grep: search for text within files

All file paths must start with /."""

EXECUTION_SYSTEM_PROMPT = """## Execute Tool

You have access to an execute tool for running shell commands.
Use this for running commands, scripts, tests, and builds.

- execute: run a shell command (returns output and exit code)"""


# ============================================================================
# Path Validation
# ============================================================================

def validate_path(path: str, allowed_prefixes: Optional[List[str]] = None) -> str:
    """Validate and normalize file path for security.

    Args:
        path: The path to validate
        allowed_prefixes: Optional list of allowed path prefixes

    Returns:
        Normalized canonical path starting with /

    Raises:
        ValueError: If path contains traversal sequences or invalid format
    """
    if ".." in path or path.startswith("~"):
        raise ValueError(f"Path traversal not allowed: {path}")

    # Reject Windows absolute paths
    if re.match(r"^[a-zA-Z]:", path):
        raise ValueError(
            f"Windows absolute paths not supported: {path}. "
            f"Use virtual paths starting with / (e.g., /workspace/file.txt)"
        )

    normalized = os.path.normpath(path)
    normalized = normalized.replace("\\", "/")

    if not normalized.startswith("/"):
        normalized = f"/{normalized}"

    if allowed_prefixes is not None:
        if not any(normalized.startswith(prefix) for prefix in allowed_prefixes):
            raise ValueError(f"Path must start with one of {allowed_prefixes}: {path}")

    return normalized


# ============================================================================
# Filesystem Tools
# ============================================================================

class LsTool(BaseTool):
    """List files in a directory."""

    name: str = "ls"
    description: str = LIST_FILES_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute path to the directory to list"
            }
        },
        "required": ["path"]
    }
    backend: Any = None

    def __init__(self, backend: BackendProtocol, **kwargs):
        super().__init__(backend=backend, **kwargs)

    async def execute(self, path: str, **kwargs) -> str:
        validated_path = validate_path(path)
        infos = await self.backend.als_info(validated_path)
        paths = [fi.get("path", "") for fi in infos]
        result = truncate_if_too_long(paths)
        return str(result) if result else "No files found"


class ReadFileTool(BaseTool):
    """Read file content."""

    name: str = "read_file"
    description: str = READ_FILE_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Absolute path to the file to read"
            },
            "offset": {
                "type": "integer",
                "description": "Line offset to start reading from (0-indexed)",
                "default": 0
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of lines to read",
                "default": 500
            }
        },
        "required": ["file_path"]
    }
    backend: Any = None

    def __init__(self, backend: BackendProtocol, **kwargs):
        super().__init__(backend=backend, **kwargs)

    async def execute(
        self,
        file_path: str,
        offset: int = DEFAULT_READ_OFFSET,
        limit: int = DEFAULT_READ_LIMIT,
        **kwargs
    ) -> str:
        validated_path = validate_path(file_path)
        return await self.backend.aread(validated_path, offset=offset, limit=limit)


class WriteFileTool(BaseTool):
    """Write to a new file."""

    name: str = "write_file"
    description: str = WRITE_FILE_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Absolute path for the new file"
            },
            "content": {
                "type": "string",
                "description": "Content to write to the file"
            }
        },
        "required": ["file_path", "content"]
    }
    backend: Any = None

    def __init__(self, backend: BackendProtocol, **kwargs):
        super().__init__(backend=backend, **kwargs)

    async def execute(self, file_path: str, content: str, **kwargs) -> str:
        validated_path = validate_path(file_path)
        result: WriteResult = await self.backend.awrite(validated_path, content)
        if result.error:
            return result.error
        return f"Successfully wrote to {result.path}"


class EditFileTool(BaseTool):
    """Edit existing file with string replacement."""

    name: str = "edit_file"
    description: str = EDIT_FILE_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Absolute path to the file to edit"
            },
            "old_string": {
                "type": "string",
                "description": "Exact string to find and replace"
            },
            "new_string": {
                "type": "string",
                "description": "String to replace old_string with"
            },
            "replace_all": {
                "type": "boolean",
                "description": "Replace all occurrences (default: false)",
                "default": False
            }
        },
        "required": ["file_path", "old_string", "new_string"]
    }
    backend: Any = None

    def __init__(self, backend: BackendProtocol, **kwargs):
        super().__init__(backend=backend, **kwargs)

    async def execute(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
        **kwargs
    ) -> str:
        validated_path = validate_path(file_path)
        result: EditResult = await self.backend.aedit(
            validated_path, old_string, new_string, replace_all=replace_all
        )
        if result.error:
            return result.error
        return f"Successfully replaced {result.occurrences} occurrence(s) in '{result.path}'"


class GlobTool(BaseTool):
    """Find files by glob pattern."""

    name: str = "glob"
    description: str = GLOB_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern (e.g., **/*.py)"
            },
            "path": {
                "type": "string",
                "description": "Base directory to search from",
                "default": "/"
            }
        },
        "required": ["pattern"]
    }
    backend: Any = None

    def __init__(self, backend: BackendProtocol, **kwargs):
        super().__init__(backend=backend, **kwargs)

    async def execute(self, pattern: str, path: str = "/", **kwargs) -> str:
        infos = await self.backend.aglob_info(pattern, path=path)
        paths = [fi.get("path", "") for fi in infos]
        result = truncate_if_too_long(paths)
        return str(result) if result else "No files found"


class GrepTool(BaseTool):
    """Search for pattern in files."""

    name: str = "grep"
    description: str = GREP_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Text pattern to search for"
            },
            "path": {
                "type": "string",
                "description": "Directory to search in",
                "default": "/"
            },
            "glob": {
                "type": "string",
                "description": "Filter files by glob pattern (e.g., *.py)"
            },
            "output_mode": {
                "type": "string",
                "enum": ["files_with_matches", "content", "count"],
                "description": "Output format",
                "default": "files_with_matches"
            }
        },
        "required": ["pattern"]
    }
    backend: Any = None

    def __init__(self, backend: BackendProtocol, **kwargs):
        super().__init__(backend=backend, **kwargs)

    async def execute(
        self,
        pattern: str,
        path: Optional[str] = None,
        glob: Optional[str] = None,
        output_mode: Literal["files_with_matches", "content", "count"] = "files_with_matches",
        **kwargs
    ) -> str:
        raw = await self.backend.agrep_raw(pattern, path=path, glob=glob)
        if isinstance(raw, str):
            return raw

        # Convert to dict format for formatting
        results_dict: Dict[str, List[tuple]] = {}
        for match in raw:
            path_key = match["path"]
            if path_key not in results_dict:
                results_dict[path_key] = []
            results_dict[path_key].append((match["line"], match["text"]))

        if not results_dict:
            return "No matches found"

        formatted = format_grep_results(results_dict, output_mode)
        return truncate_if_too_long(formatted)


class ExecuteTool(BaseTool):
    """Execute shell command in sandbox."""

    name: str = "execute"
    description: str = EXECUTE_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Shell command to execute"
            }
        },
        "required": ["command"]
    }
    backend: Any = None

    def __init__(self, backend: BackendProtocol, **kwargs):
        super().__init__(backend=backend, **kwargs)

    async def execute(self, command: str, **kwargs) -> str:
        # Check if backend supports execution
        if not isinstance(self.backend, SandboxBackendProtocol):
            return (
                "Error: Execution not available. This backend does not support "
                "command execution (SandboxBackendProtocol)."
            )

        try:
            result: ExecuteResponse = await self.backend.aexecute(command)
        except NotImplementedError as e:
            return f"Error: Execution not available. {e}"

        # Format output
        parts = [result.output]

        if result.exit_code is not None:
            status = "succeeded" if result.exit_code == 0 else "failed"
            parts.append(f"\n[Command {status} with exit code {result.exit_code}]")

        if result.truncated:
            parts.append("\n[Output was truncated due to size limits]")

        return "".join(parts)


# ============================================================================
# Tool Generators
# ============================================================================

TOOL_GENERATORS = {
    "ls": LsTool,
    "read_file": ReadFileTool,
    "write_file": WriteFileTool,
    "edit_file": EditFileTool,
    "glob": GlobTool,
    "grep": GrepTool,
    "execute": ExecuteTool,
}


def get_filesystem_tools(backend: BackendProtocol) -> List[BaseTool]:
    """Get all filesystem tools for a backend.

    Args:
        backend: Backend to use for file operations

    Returns:
        List of 7 filesystem tools
    """
    return [tool_class(backend) for tool_class in TOOL_GENERATORS.values()]


# ============================================================================
# Filesystem Middleware
# ============================================================================

class FilesystemMiddleware(AgentMiddleware):
    """Middleware for providing filesystem and execution tools to an agent.

    Adds 7 filesystem tools to the agent:
    - ls: list files in directory
    - read_file: read file content
    - write_file: write new file
    - edit_file: edit existing file
    - glob: find files by pattern
    - grep: search content in files
    - execute: run shell commands (if backend supports)

    Example:
        ```python
        from spoon_ai.middleware.filesystem import FilesystemMiddleware
        from spoon_ai.backends import create_state_backend, create_composite_backend

        # With ephemeral storage (default)
        middleware = FilesystemMiddleware()

        # With custom backend
        backend, runtime = create_state_backend()
        middleware = FilesystemMiddleware(backend=backend)

        # With composite backend (mixed storage)
        composite = create_composite_backend(
            default=state_backend,
            routes={"/persistent/": store_backend}
        )
        middleware = FilesystemMiddleware(backend=composite)

        agent = ToolCallAgent(
            middleware=[middleware],
            ...
        )
        ```
    """

    def __init__(
        self,
        backend: Optional[BackendProtocol] = None,
        system_prompt: Optional[str] = None,
        include_execute: bool = True,
        tool_token_limit: int = TOOL_TOKEN_LIMIT,
    ):
        """Initialize filesystem middleware.

        Args:
            backend: Backend for file operations. Defaults to StateBackend.
            system_prompt: Optional custom system prompt override.
            include_execute: Whether to include execute tool (default: True)
            tool_token_limit: Token limit before truncating tool results
        """
        super().__init__()

        # Create default backend if not provided
        if backend is None:
            self._runtime = BackendRuntime(state={"files": {}})
            self._backend = StateBackend(self._runtime)
        else:
            self._backend = backend
            self._runtime = None

        self._custom_system_prompt = system_prompt
        self._include_execute = include_execute
        self._tool_token_limit = tool_token_limit

        # Generate tools
        self._tools = self._create_tools()

    def _create_tools(self) -> List[BaseTool]:
        """Create filesystem tools."""
        tools = []
        for name, tool_class in TOOL_GENERATORS.items():
            # Skip execute if not supported or not requested
            if name == "execute":
                if not self._include_execute:
                    continue
                if not isinstance(self._backend, SandboxBackendProtocol):
                    continue
            tools.append(tool_class(self._backend))
        return tools

    @property
    def tools(self) -> List[BaseTool]:
        """Get filesystem tools."""
        return self._tools

    @property
    def system_prompt(self) -> str:
        """Get system prompt for filesystem tools."""
        if self._custom_system_prompt:
            return self._custom_system_prompt

        # Build dynamic prompt based on available tools
        prompts = [FILESYSTEM_SYSTEM_PROMPT]

        # Add execution prompt if execute tool is available
        if any(t.name == "execute" for t in self._tools):
            prompts.append(EXECUTION_SYSTEM_PROMPT)

        return "\n\n".join(prompts)

    @property
    def backend(self) -> BackendProtocol:
        """Get the backend."""
        return self._backend

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable
    ) -> ModelResponse:
        """Inject system prompt for filesystem tools."""
        # Append filesystem system prompt
        if request.system_prompt:
            new_prompt = f"{request.system_prompt}\n\n{self.system_prompt}"
        else:
            new_prompt = self.system_prompt

        # Create new request with updated system prompt
        request = ModelRequest(
            messages=request.messages,
            system_prompt=new_prompt,
            tools=request.tools,
            phase=request.phase,
            runtime=request.runtime,
        )

        return await handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable
    ) -> ToolCallResult:
        """Handle large tool results by truncating."""
        result = await handler(request)

        # Truncate large results
        if result.success and isinstance(result.result, str):
            if len(result.result) > self._tool_token_limit * 4:
                # Truncate and add notice
                truncated = result.result[:self._tool_token_limit * 4]
                truncated += "\n\n[Output truncated due to size limits]"
                return ToolCallResult(
                    tool_name=result.tool_name,
                    result=truncated,
                    success=True
                )

        return result


# ============================================================================
# Convenience Functions
# ============================================================================

def create_filesystem_middleware(
    backend: Optional[BackendProtocol] = None,
    include_execute: bool = True,
) -> FilesystemMiddleware:
    """Create a filesystem middleware.

    Args:
        backend: Backend for file operations
        include_execute: Whether to include execute tool

    Returns:
        FilesystemMiddleware instance

    Example:
        ```python
        middleware = create_filesystem_middleware()
        agent = ToolCallAgent(middleware=[middleware], ...)
        ```
    """
    return FilesystemMiddleware(
        backend=backend,
        include_execute=include_execute
    )


def create_sandbox_backend(
    root_dir: Optional[str] = None,
    timeout: int = 30,
) -> "LocalSandboxBackend":
    """Create a local sandbox backend that supports command execution.

    Args:
        root_dir: Root directory for sandbox
        timeout: Command execution timeout in seconds

    Returns:
        LocalSandboxBackend instance
    """
    return LocalSandboxBackend(root_dir=root_dir, timeout=timeout)


# ============================================================================
# Local Sandbox Backend (supports execute)
# ============================================================================

class LocalSandboxBackend(SandboxBackendProtocol):
    """Local sandbox backend with command execution support.

    Wraps FilesystemBackend and adds execute capability.
    """

    def __init__(
        self,
        root_dir: Optional[str] = None,
        timeout: int = 30,
        virtual_mode: bool = True,
    ):
        from spoon_ai.backends.filesystem import FilesystemBackend

        self._fs_backend = FilesystemBackend(
            root_dir=root_dir,
            virtual_mode=virtual_mode
        )
        self._timeout = timeout
        self._root_dir = root_dir or os.getcwd()
        self._id = f"local-sandbox-{id(self)}"

    @property
    def id(self) -> str:
        return self._id

    # Delegate all file operations to filesystem backend
    def ls_info(self, path: str):
        return self._fs_backend.ls_info(path)

    async def als_info(self, path: str):
        return await self._fs_backend.als_info(path)

    def read(self, file_path: str, offset: int = 0, limit: int = 2000):
        return self._fs_backend.read(file_path, offset, limit)

    async def aread(self, file_path: str, offset: int = 0, limit: int = 2000):
        return await self._fs_backend.aread(file_path, offset, limit)

    def write(self, file_path: str, content: str):
        return self._fs_backend.write(file_path, content)

    async def awrite(self, file_path: str, content: str):
        return await self._fs_backend.awrite(file_path, content)

    def edit(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False):
        return self._fs_backend.edit(file_path, old_string, new_string, replace_all)

    async def aedit(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False):
        return await self._fs_backend.aedit(file_path, old_string, new_string, replace_all)

    def grep_raw(self, pattern: str, path: Optional[str] = None, glob: Optional[str] = None):
        return self._fs_backend.grep_raw(pattern, path, glob)

    async def agrep_raw(self, pattern: str, path: Optional[str] = None, glob: Optional[str] = None):
        return await self._fs_backend.agrep_raw(pattern, path, glob)

    def glob_info(self, pattern: str, path: str = "/"):
        return self._fs_backend.glob_info(pattern, path)

    async def aglob_info(self, pattern: str, path: str = "/"):
        return await self._fs_backend.aglob_info(pattern, path)

    def upload_files(self, files):
        return self._fs_backend.upload_files(files)

    async def aupload_files(self, files):
        return await self._fs_backend.aupload_files(files)

    def download_files(self, paths):
        return self._fs_backend.download_files(paths)

    async def adownload_files(self, paths):
        return await self._fs_backend.adownload_files(paths)

    # Execute implementation
    def execute(self, command: str) -> ExecuteResponse:
        """Execute a shell command."""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                cwd=self._root_dir,
            )

            output = result.stdout + result.stderr
            truncated = False

            # Truncate if too large
            if len(output) > 100000:
                output = output[:100000]
                truncated = True

            return ExecuteResponse(
                output=output,
                exit_code=result.returncode,
                truncated=truncated
            )

        except subprocess.TimeoutExpired:
            return ExecuteResponse(
                output=f"Command timed out after {self._timeout} seconds",
                exit_code=-1,
                truncated=False
            )
        except Exception as e:
            return ExecuteResponse(
                output=f"Error executing command: {e}",
                exit_code=-1,
                truncated=False
            )

    async def aexecute(self, command: str) -> ExecuteResponse:
        """Async execute a shell command."""
        return await asyncio.to_thread(self.execute, command)
