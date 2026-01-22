"""Protocol definition for pluggable memory backends.

This module defines the BackendProtocol that all backend implementations
must follow. Backends can store files in different locations (state, filesystem,
database, etc.) and provide a uniform interface for file operations.

Compatible with LangChain DeepAgents backend interface.

Backend Types:
    - StateBackend: Ephemeral in-memory storage (per-thread)
    - FilesystemBackend: Real filesystem access (optionally sandboxed)
    - StoreBackend: Persistent key-value storage (cross-thread)
    - CompositeBackend: Route operations to multiple backends by path prefix
    - BaseSandbox: Abstract base for remote sandboxes (Docker, Modal, etc.)

Example:
    ```python
    from spoon_ai.backends import (
        StateBackend,
        FilesystemBackend,
        create_state_backend,
    )

    # Create a state backend
    backend, runtime = create_state_backend()

    # Write a file
    result = backend.write("/notes.txt", "Hello, World!")
    if result.error:
        print(f"Error: {result.error}")
    else:
        print(f"Created: {result.path}")

    # Read it back
    content = backend.read("/notes.txt")
    print(content)
    ```
"""

import abc
import asyncio
from dataclasses import dataclass, field
from typing import Any, Literal, Callable, TypeAlias, Optional, List, Tuple
from typing_extensions import TypedDict, NotRequired


# ============================================================================
# Error Types
# ============================================================================

FileOperationError = Literal[
    "file_not_found",      # Download: file doesn't exist
    "permission_denied",   # Both: access denied
    "is_directory",        # Download: tried to download directory as file
    "invalid_path",        # Both: path syntax malformed (parent dir missing, invalid chars)
]
"""Standardized error codes for file upload/download operations.

These represent common, recoverable errors that an LLM can understand and potentially fix:

- ``file_not_found``: The requested file doesn't exist (download)
- ``permission_denied``: Access denied for the operation
- ``is_directory``: Attempted to download a directory as a file
- ``invalid_path``: Path syntax is malformed or contains invalid characters

Example:
    ```python
    response = backend.download_files(["/nonexistent.txt"])
    if response[0].error == "file_not_found":
        print("File does not exist")
    ```
"""


# ============================================================================
# Response Data Classes
# ============================================================================

@dataclass
class FileDownloadResponse:
    """Result of a single file download operation.

    The response is designed to allow partial success in batch operations.
    The errors are standardized using FileOperationError literals
    for certain recoverable conditions for use cases that involve
    LLMs performing file operations.

    Attributes:
        path: The file path that was requested. Included for easy correlation
            when processing batch results, especially useful for error messages.
        content: File contents as bytes on success, None on failure.
        error: Standardized error code on failure, None on success.
            Uses FileOperationError literal for structured, LLM-actionable error reporting.

    Examples:
        >>> # Success
        >>> FileDownloadResponse(path="/app/config.json", content=b"{...}", error=None)
        >>> # Failure
        >>> FileDownloadResponse(path="/wrong/path.txt", content=None, error="file_not_found")
    """
    path: str
    content: Optional[bytes] = None
    error: Optional[FileOperationError] = None


@dataclass
class FileUploadResponse:
    """Result of a single file upload operation.

    The response is designed to allow partial success in batch operations.
    The errors are standardized using FileOperationError literals
    for certain recoverable conditions for use cases that involve
    LLMs performing file operations.

    Attributes:
        path: The file path that was requested. Included for easy correlation
            when processing batch results and for clear error messages.
        error: Standardized error code on failure, None on success.
            Uses FileOperationError literal for structured, LLM-actionable error reporting.

    Examples:
        >>> # Success
        >>> FileUploadResponse(path="/app/data.txt", error=None)
        >>> # Failure
        >>> FileUploadResponse(path="/readonly/file.txt", error="permission_denied")
    """
    path: str
    error: Optional[FileOperationError] = None


class FileInfo(TypedDict):
    """Structured file listing info.

    Minimal contract used across backends. Only "path" is required.
    Other fields are best-effort and may be absent depending on backend.

    Attributes:
        path: Absolute file path (required)
        is_dir: True if directory (optional)
        size: File size in bytes (optional, approximate)
        modified_at: ISO 8601 timestamp if known (optional)

    Example:
        ```python
        files = backend.ls_info("/workspace")
        for f in files:
            print(f"Path: {f['path']}, Is Dir: {f.get('is_dir', False)}")
        ```
    """
    path: str
    is_dir: NotRequired[bool]
    size: NotRequired[int]
    modified_at: NotRequired[str]


class GrepMatch(TypedDict):
    """Structured grep match entry.

    Represents a single match from a grep search operation.

    Attributes:
        path: Absolute file path where match was found
        line: Line number (1-indexed)
        text: Full line content containing the match

    Example:
        ```python
        matches = backend.grep_raw("TODO", path="/src")
        for m in matches:
            print(f"{m['path']}:{m['line']}: {m['text']}")
        ```
    """
    path: str
    line: int
    text: str


@dataclass
class WriteResult:
    """Result from backend write operations.

    Attributes:
        error: Error message on failure, None on success.
        path: Absolute path of written file, None on failure.
        files_update: State update dict for checkpoint backends, None for external storage.
            Checkpoint backends populate this with {file_path: file_data} for state sync.
            External backends set None (already persisted to disk/S3/database/etc).

    Examples:
        >>> # Checkpoint storage (state backend)
        >>> WriteResult(path="/f.txt", files_update={"/f.txt": {...}})
        >>> # External storage (filesystem backend)
        >>> WriteResult(path="/f.txt", files_update=None)
        >>> # Error
        >>> WriteResult(error="File '/f.txt' already exists")
    """
    error: Optional[str] = None
    path: Optional[str] = None
    files_update: Optional[dict[str, Any]] = None


@dataclass
class EditResult:
    """Result from backend edit operations.

    Attributes:
        error: Error message on failure, None on success.
        path: Absolute path of edited file, None on failure.
        files_update: State update dict for checkpoint backends, None for external storage.
            Checkpoint backends populate this with {file_path: file_data} for state sync.
            External backends set None (already persisted to disk/S3/database/etc).
        occurrences: Number of replacements made, None on failure.

    Examples:
        >>> # Checkpoint storage with single replacement
        >>> EditResult(path="/f.txt", files_update={"/f.txt": {...}}, occurrences=1)
        >>> # External storage with replace_all
        >>> EditResult(path="/f.txt", files_update=None, occurrences=5)
        >>> # Error - string not found
        >>> EditResult(error="String not found in file")
        >>> # Error - multiple occurrences without replace_all
        >>> EditResult(error="String appears 3 times. Use replace_all=True")
    """
    error: Optional[str] = None
    path: Optional[str] = None
    files_update: Optional[dict[str, Any]] = None
    occurrences: Optional[int] = None


@dataclass
class ExecuteResponse:
    """Result of code execution.

    Simplified schema optimized for LLM consumption.

    Attributes:
        output: Combined stdout and stderr output of the executed command.
        exit_code: The process exit code. 0 indicates success, non-zero indicates failure.
        truncated: Whether the output was truncated due to backend limitations.

    Example:
        ```python
        result = sandbox.execute("ls -la /workspace")
        if result.exit_code == 0:
            print(result.output)
        else:
            print(f"Command failed with exit code {result.exit_code}")
        ```
    """
    output: str
    """Combined stdout and stderr output of the executed command."""

    exit_code: Optional[int] = None
    """The process exit code. 0 indicates success, non-zero indicates failure."""

    truncated: bool = False
    """Whether the output was truncated due to backend limitations."""


# ============================================================================
# Runtime Context
# ============================================================================

@dataclass
class BackendRuntime:
    """Runtime context for backend operations.

    Provides access to agent state and configuration without
    exposing the entire agent instance. This is an independent
    implementation that doesn't depend on LangChain's ToolRuntime.

    Attributes:
        state: Mutable state dictionary for the current execution.
        config: Immutable configuration dictionary.
        store: Optional persistent store for cross-session data.

    Example:
        ```python
        runtime = BackendRuntime(
            state={"files": {}},
            config={"max_file_size": 1024 * 1024},
        )

        # Access state
        files = runtime.get_state("files", {})

        # Update state
        runtime.set_state("last_read", "/notes.txt")
        ```
    """
    state: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    store: Optional[Any] = None  # Optional persistent store

    def get_state(self, key: str, default: Any = None) -> Any:
        """Get state value by key.

        Args:
            key: State key to retrieve.
            default: Default value if key not found.

        Returns:
            State value or default.
        """
        return self.state.get(key, default)

    def set_state(self, key: str, value: Any) -> None:
        """Set state value.

        Args:
            key: State key to set.
            value: Value to store.
        """
        self.state[key] = value


# ============================================================================
# Backend Protocol
# ============================================================================

class BackendProtocol(abc.ABC):
    """Protocol for pluggable memory backends (single, unified).

    Backends can store files in different locations (state, filesystem, database, etc.)
    and provide a uniform interface for file operations.

    All file data is represented as dicts with the following structure::

        {
            "content": list[str],      # Lines of text content
            "created_at": str,         # ISO format timestamp
            "modified_at": str,        # ISO format timestamp
        }

    Implementing a Backend:
        To create a custom backend, subclass BackendProtocol and implement
        all abstract methods::

            class MyBackend(BackendProtocol):
                def ls_info(self, path: str) -> list[FileInfo]:
                    # List directory contents
                    ...

                def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
                    # Read file with line numbers
                    ...

                # ... implement other abstract methods
    """

    @abc.abstractmethod
    def ls_info(self, path: str) -> List[FileInfo]:
        """List all files in a directory with metadata.

        Args:
            path: Absolute path to the directory to list. Must start with '/'.

        Returns:
            List of FileInfo dicts containing file metadata:

            - ``path`` (required): Absolute file path
            - ``is_dir`` (optional): True if directory
            - ``size`` (optional): File size in bytes
            - ``modified_at`` (optional): ISO 8601 timestamp

        Example:
            ```python
            files = backend.ls_info("/workspace/src")
            for f in files:
                if f.get("is_dir"):
                    print(f"📁 {f['path']}")
                else:
                    print(f"📄 {f['path']} ({f.get('size', '?')} bytes)")
            ```
        """
        ...

    async def als_info(self, path: str) -> List[FileInfo]:
        """Async version of ls_info."""
        return await asyncio.to_thread(self.ls_info, path)

    @abc.abstractmethod
    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        """Read file content with line numbers.

        Args:
            file_path: Absolute path to the file to read. Must start with '/'.
            offset: Line number to start reading from (0-indexed). Default: 0.
            limit: Maximum number of lines to read. Default: 2000.

        Returns:
            String containing file content formatted with line numbers (cat -n format),
            starting at line 1. Lines longer than 2000 characters are truncated.

            Returns an error string if the file doesn't exist or can't be read.

        Note:
            - Use pagination (offset/limit) for large files to avoid context overflow
            - First scan: ``read(path, limit=100)`` to see file structure
            - Read more: ``read(path, offset=100, limit=200)`` for next section
            - ALWAYS read a file before editing it
            - If file exists but is empty, you'll receive a system reminder warning

        Example:
            ```python
            # Read first 100 lines
            content = backend.read("/workspace/main.py", limit=100)
            print(content)
            # Output:
            #      1	import os
            #      2	import sys
            #      3
            #      4	def main():
            #      ...
            ```
        """
        ...

    async def aread(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        """Async version of read."""
        return await asyncio.to_thread(self.read, file_path, offset, limit)

    @abc.abstractmethod
    def write(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        """Write content to a new file in the filesystem, error if file exists.

        Args:
            file_path: Absolute path where the file should be created.
                Must start with '/'.
            content: String content to write to the file.

        Returns:
            WriteResult with:
            - ``path``: Absolute path of created file (on success)
            - ``error``: Error message (on failure)
            - ``files_update``: State dict for checkpoint backends

        Note:
            This method creates NEW files only. To modify existing files,
            use the ``edit()`` method instead.

        Example:
            ```python
            result = backend.write("/workspace/hello.txt", "Hello, World!")
            if result.error:
                print(f"Failed: {result.error}")
            else:
                print(f"Created: {result.path}")
            ```
        """
        ...

    async def awrite(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        """Async version of write."""
        return await asyncio.to_thread(self.write, file_path, content)

    @abc.abstractmethod
    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Perform exact string replacements in an existing file.

        Args:
            file_path: Absolute path to the file to edit. Must start with '/'.
            old_string: Exact string to search for and replace.
                Must match exactly including whitespace and indentation.
            new_string: String to replace old_string with.
                Must be different from old_string.
            replace_all: If True, replace all occurrences. If False (default),
                old_string must be unique in the file or the edit fails.

        Returns:
            EditResult with:
            - ``path``: Absolute path of edited file (on success)
            - ``occurrences``: Number of replacements made (on success)
            - ``error``: Error message (on failure)
            - ``files_update``: State dict for checkpoint backends

        Note:
            - ALWAYS read the file first to understand its current content
            - The old_string must match EXACTLY, including whitespace
            - If old_string appears multiple times and replace_all=False, the edit fails

        Example:
            ```python
            # Replace a single occurrence
            result = backend.edit(
                "/workspace/config.py",
                old_string="DEBUG = False",
                new_string="DEBUG = True"
            )

            # Replace all occurrences
            result = backend.edit(
                "/workspace/code.py",
                old_string="old_function",
                new_string="new_function",
                replace_all=True
            )
            print(f"Replaced {result.occurrences} occurrences")
            ```
        """
        ...

    async def aedit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Async version of edit."""
        return await asyncio.to_thread(
            self.edit, file_path, old_string, new_string, replace_all
        )

    @abc.abstractmethod
    def grep_raw(
        self,
        pattern: str,
        path: Optional[str] = None,
        glob: Optional[str] = None,
    ) -> List[GrepMatch] | str:
        """Search for a literal text pattern in files.

        Args:
            pattern: Literal string to search for (NOT regex).
                Performs exact substring matching within file content.
                Example: "TODO" matches any line containing "TODO"

            path: Optional directory path to search in.
                If None, searches in current working directory.
                Example: "/workspace/src"

            glob: Optional glob pattern to filter which FILES to search.
                Filters by filename/path, not content.
                Supports standard glob wildcards:

                - ``*`` matches any characters in filename
                - ``**`` matches any directories recursively
                - ``?`` matches single character
                - ``[abc]`` matches one character from set

                Examples:
                    - ``"*.py"`` - only search Python files
                    - ``"**/*.txt"`` - search all .txt files recursively
                    - ``"src/**/*.js"`` - search JS files under src/
                    - ``"test[0-9].txt"`` - search test0.txt, test1.txt, etc.

        Returns:
            On success: list[GrepMatch] with structured results containing:
                - path: Absolute file path
                - line: Line number (1-indexed)
                - text: Full line content containing the match

            On error: str with error message (e.g., invalid path, permission denied)

        Example:
            ```python
            # Search for TODOs in Python files
            matches = backend.grep_raw("TODO", path="/workspace", glob="*.py")
            for m in matches:
                print(f"{m['path']}:{m['line']}: {m['text']}")
            ```
        """
        ...

    async def agrep_raw(
        self,
        pattern: str,
        path: Optional[str] = None,
        glob: Optional[str] = None,
    ) -> List[GrepMatch] | str:
        """Async version of grep_raw."""
        return await asyncio.to_thread(self.grep_raw, pattern, path, glob)

    @abc.abstractmethod
    def glob_info(self, pattern: str, path: str = "/") -> List[FileInfo]:
        """Find files matching a glob pattern.

        Args:
            pattern: Glob pattern with wildcards to match file paths.
                Supports standard glob syntax:

                - ``*`` matches any characters within a filename/directory
                - ``**`` matches any directories recursively
                - ``?`` matches a single character
                - ``[abc]`` matches one character from set

            path: Base directory to search from. Default: "/" (root).
                The pattern is applied relative to this path.

        Returns:
            List of FileInfo dicts for matching files.

        Example:
            ```python
            # Find all Python files
            py_files = backend.glob_info("**/*.py", path="/workspace")

            # Find config files in root
            configs = backend.glob_info("*.json", path="/workspace")

            # Find test files
            tests = backend.glob_info("**/test_*.py", path="/workspace")
            ```
        """
        ...

    async def aglob_info(self, pattern: str, path: str = "/") -> List[FileInfo]:
        """Async version of glob_info."""
        return await asyncio.to_thread(self.glob_info, pattern, path)

    def upload_files(
        self, files: List[Tuple[str, bytes]]
    ) -> List[FileUploadResponse]:
        """Upload multiple files to the backend.

        This API is designed to allow developers to use it either directly or
        by exposing it to LLMs via custom tools. Default implementation uses
        write() for each file.

        Args:
            files: List of (path, content) tuples to upload.

        Returns:
            List of FileUploadResponse objects, one per input file.
            Response order matches input order (response[i] for files[i]).
            Check the error field to determine success/failure per file.

        Example:
            ```python
            responses = backend.upload_files([
                ("/app/config.json", b'{"key": "value"}'),
                ("/app/data.txt", b"content here"),
            ])
            for resp in responses:
                if resp.error:
                    print(f"Failed {resp.path}: {resp.error}")
                else:
                    print(f"Uploaded {resp.path}")
            ```
        """
        responses = []
        for path, content in files:
            try:
                content_str = content.decode("utf-8")
                result = self.write(path, content_str)
                if result.error:
                    responses.append(FileUploadResponse(path=path, error="invalid_path"))
                else:
                    responses.append(FileUploadResponse(path=path, error=None))
            except Exception:
                responses.append(FileUploadResponse(path=path, error="invalid_path"))
        return responses

    async def aupload_files(
        self, files: List[Tuple[str, bytes]]
    ) -> List[FileUploadResponse]:
        """Async version of upload_files."""
        return await asyncio.to_thread(self.upload_files, files)

    def download_files(self, paths: List[str]) -> List[FileDownloadResponse]:
        """Download multiple files from the backend.

        This API is designed to allow developers to use it either directly or
        by exposing it to LLMs via custom tools. Default implementation uses
        read() for each file.

        Args:
            paths: List of file paths to download.

        Returns:
            List of FileDownloadResponse objects, one per input path.
            Response order matches input order (response[i] for paths[i]).
            Check the error field to determine success/failure per file.

        Example:
            ```python
            responses = backend.download_files([
                "/app/config.json",
                "/app/data.txt",
            ])
            for resp in responses:
                if resp.error:
                    print(f"Failed {resp.path}: {resp.error}")
                else:
                    print(f"Downloaded {resp.path}: {len(resp.content)} bytes")
            ```
        """
        responses = []
        for path in paths:
            result = self.read(path)
            if result.startswith("Error:"):
                responses.append(
                    FileDownloadResponse(path=path, content=None, error="file_not_found")
                )
            else:
                # Strip line numbers and reconstruct content
                lines = []
                for line in result.split("\n"):
                    if "\t" in line:
                        lines.append(line.split("\t", 1)[1])
                content = "\n".join(lines).encode("utf-8")
                responses.append(
                    FileDownloadResponse(path=path, content=content, error=None)
                )
        return responses

    async def adownload_files(self, paths: List[str]) -> List[FileDownloadResponse]:
        """Async version of download_files."""
        return await asyncio.to_thread(self.download_files, paths)


# ============================================================================
# Sandbox Backend Protocol (extends BackendProtocol)
# ============================================================================

class SandboxBackendProtocol(BackendProtocol):
    """Protocol for sandboxed backends with isolated runtime.

    Sandboxed backends run in isolated environments (e.g., separate processes,
    containers) and communicate via defined interfaces. This extends
    BackendProtocol with command execution capabilities.

    Use Cases:
        - Docker containers for isolated code execution
        - Modal/Daytona for cloud-based sandboxes
        - Local subprocess sandboxes
        - Remote SSH execution

    Example:
        ```python
        class DockerSandbox(SandboxBackendProtocol):
            def __init__(self, container_id: str):
                self._container_id = container_id

            @property
            def id(self) -> str:
                return f"docker-{self._container_id}"

            def execute(self, command: str) -> ExecuteResponse:
                result = docker.exec_run(self._container_id, command)
                return ExecuteResponse(
                    output=result.output.decode(),
                    exit_code=result.exit_code
                )

            # ... implement other methods using execute()
        ```
    """

    @abc.abstractmethod
    def execute(self, command: str) -> ExecuteResponse:
        """Execute a command in the sandbox.

        Simplified interface optimized for LLM consumption.

        Args:
            command: Full shell command string to execute.

        Returns:
            ExecuteResponse with:
            - ``output``: Combined stdout and stderr
            - ``exit_code``: Process exit code (0 = success)
            - ``truncated``: Whether output was truncated

        Example:
            ```python
            result = sandbox.execute("python3 -c 'print(1+1)'")
            if result.exit_code == 0:
                print(f"Output: {result.output}")  # "2\n"
            else:
                print(f"Failed: {result.output}")
            ```
        """
        ...

    async def aexecute(self, command: str) -> ExecuteResponse:
        """Async version of execute."""
        return await asyncio.to_thread(self.execute, command)

    @property
    @abc.abstractmethod
    def id(self) -> str:
        """Unique identifier for this sandbox instance.

        Used for tracking, logging, and distinguishing between
        multiple sandbox instances.

        Returns:
            Unique string identifier (e.g., "docker-abc123", "modal-xyz789")
        """
        ...


# ============================================================================
# Type Aliases
# ============================================================================

BackendFactory: TypeAlias = Callable[[BackendRuntime], BackendProtocol]
"""Factory function that creates a backend from a runtime context.

Example:
    ```python
    def my_backend_factory(runtime: BackendRuntime) -> BackendProtocol:
        return StateBackend(runtime)

    # Use with agent
    agent = ToolCallAgent(backend=my_backend_factory, ...)
    ```
"""

BACKEND_TYPES = BackendProtocol | BackendFactory
"""Union type for backend specification.

Can be either:
- A BackendProtocol instance (pre-created backend)
- A BackendFactory callable (creates backend from runtime)
"""
