"""Base sandbox implementation with execute() as the only required abstract method.

This module provides a base class that implements all SandboxBackendProtocol
methods using shell commands executed via execute(). Concrete implementations
only need to implement the execute() method.

This design allows for remote sandboxes (Docker, Modal, Daytona, etc.) where
you just implement execute() to run commands remotely.

Compatible with LangChain DeepAgents BaseSandbox interface.

Usage:
    # For local execution
    class LocalSandbox(BaseSandbox):
        def execute(self, command: str) -> ExecuteResponse:
            # Run locally
            result = subprocess.run(command, shell=True, ...)
            return ExecuteResponse(output=result.stdout, exit_code=result.returncode)

    # For remote execution (Docker, Modal, etc.)
    class DockerSandbox(BaseSandbox):
        def execute(self, command: str) -> ExecuteResponse:
            # Run in Docker container
            result = docker_client.containers.run(self.image, command, ...)
            return ExecuteResponse(output=result, exit_code=0)
"""

from __future__ import annotations

import asyncio
import base64
import json
import shlex
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple

from .protocol import (
    BackendProtocol,
    EditResult,
    ExecuteResponse,
    FileDownloadResponse,
    FileInfo,
    FileUploadResponse,
    GrepMatch,
    SandboxBackendProtocol,
    WriteResult,
)


# ============================================================================
# Shell Command Templates
# ============================================================================

_GLOB_COMMAND_TEMPLATE = '''python3 -c "
import glob
import os
import json
import base64

# Decode base64-encoded parameters
path = base64.b64decode('{path_b64}').decode('utf-8')
pattern = base64.b64decode('{pattern_b64}').decode('utf-8')

os.chdir(path)
matches = sorted(glob.glob(pattern, recursive=True))
for m in matches:
    stat = os.stat(m)
    result = {{
        'path': m,
        'size': stat.st_size,
        'mtime': stat.st_mtime,
        'is_dir': os.path.isdir(m)
    }}
    print(json.dumps(result))
" 2>/dev/null'''

_WRITE_COMMAND_TEMPLATE = '''python3 -c "
import os
import sys
import base64

file_path = '{file_path}'

# Check if file already exists (atomic with write)
if os.path.exists(file_path):
    print(f'Error: File \\'{file_path}\\' already exists', file=sys.stderr)
    sys.exit(1)

# Create parent directory if needed
parent_dir = os.path.dirname(file_path) or '.'
os.makedirs(parent_dir, exist_ok=True)

# Decode and write content
content = base64.b64decode('{content_b64}').decode('utf-8')
with open(file_path, 'w') as f:
    f.write(content)
" 2>&1'''

_EDIT_COMMAND_TEMPLATE = '''python3 -c "
import sys
import base64

# Read file content
with open('{file_path}', 'r') as f:
    text = f.read()

# Decode base64-encoded strings
old = base64.b64decode('{old_b64}').decode('utf-8')
new = base64.b64decode('{new_b64}').decode('utf-8')

# Count occurrences
count = text.count(old)

# Exit with error codes if issues found
if count == 0:
    sys.exit(1)  # String not found
elif count > 1 and not {replace_all}:
    sys.exit(2)  # Multiple occurrences without replace_all

# Perform replacement
if {replace_all}:
    result = text.replace(old, new)
else:
    result = text.replace(old, new, 1)

# Write back to file
with open('{file_path}', 'w') as f:
    f.write(result)

print(count)
" 2>&1'''

_READ_COMMAND_TEMPLATE = '''python3 -c "
import os
import sys

file_path = '{file_path}'
offset = {offset}
limit = {limit}

# Check if file exists
if not os.path.isfile(file_path):
    print('Error: File not found')
    sys.exit(1)

# Check if file is empty
if os.path.getsize(file_path) == 0:
    print('System reminder: File exists but has empty contents')
    sys.exit(0)

# Read file with offset and limit
with open(file_path, 'r') as f:
    lines = f.readlines()

# Apply offset and limit
start_idx = offset
end_idx = offset + limit
selected_lines = lines[start_idx:end_idx]

# Format with line numbers (1-indexed, starting from offset + 1)
for i, line in enumerate(selected_lines):
    line_num = offset + i + 1
    # Remove trailing newline for formatting, then add it back
    line_content = line.rstrip('\\n')
    print(f'{{line_num:6d}}\\t{{line_content}}')
" 2>&1'''

_LS_COMMAND_TEMPLATE = '''python3 -c "
import os
import json

path = '{path}'

try:
    with os.scandir(path) as it:
        for entry in it:
            result = {{
                'path': entry.name,
                'is_dir': entry.is_dir(follow_symlinks=False)
            }}
            print(json.dumps(result))
except FileNotFoundError:
    pass
except PermissionError:
    pass
" 2>/dev/null'''


# ============================================================================
# Base Sandbox Abstract Class
# ============================================================================

class BaseSandbox(SandboxBackendProtocol, ABC):
    """Base sandbox implementation with execute() as abstract method.

    This class provides default implementations for all protocol methods
    using shell commands. Subclasses only need to implement:
    - execute(): Run a shell command and return output
    - id: Unique identifier property
    - upload_files(): Upload files to sandbox (optional, has default)
    - download_files(): Download files from sandbox (optional, has default)

    The default implementations use Python commands executed via execute()
    to perform file operations, making this suitable for remote sandboxes
    where you only have shell access.

    Example:
        ```python
        class DockerSandbox(BaseSandbox):
            def __init__(self, container_id: str):
                self._container_id = container_id

            @property
            def id(self) -> str:
                return f"docker-{self._container_id}"

            def execute(self, command: str) -> ExecuteResponse:
                result = docker_exec(self._container_id, command)
                return ExecuteResponse(
                    output=result.output,
                    exit_code=result.exit_code
                )
        ```
    """

    @abstractmethod
    def execute(self, command: str) -> ExecuteResponse:
        """Execute a command in the sandbox and return ExecuteResponse.

        This is the core method that subclasses must implement.
        All other file operations are built on top of this.

        Args:
            command: Full shell command string to execute.

        Returns:
            ExecuteResponse with combined output, exit code, and truncation flag.
        """
        ...

    async def aexecute(self, command: str) -> ExecuteResponse:
        """Async version of execute."""
        return await asyncio.to_thread(self.execute, command)

    @property
    @abstractmethod
    def id(self) -> str:
        """Unique identifier for the sandbox backend."""
        ...

    # ========================================================================
    # Default Implementations Using Shell Commands
    # ========================================================================

    def ls_info(self, path: str) -> List[FileInfo]:
        """List directory contents using shell command."""
        cmd = _LS_COMMAND_TEMPLATE.format(path=path)
        result = self.execute(cmd)

        file_infos: List[FileInfo] = []
        for line in result.output.strip().split("\n"):
            if not line:
                continue
            try:
                data = json.loads(line)
                file_infos.append({"path": data["path"], "is_dir": data["is_dir"]})
            except json.JSONDecodeError:
                continue

        return file_infos

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        """Read file content with line numbers using shell command."""
        cmd = _READ_COMMAND_TEMPLATE.format(
            file_path=file_path,
            offset=offset,
            limit=limit
        )
        result = self.execute(cmd)

        output = result.output.rstrip()
        exit_code = result.exit_code

        if exit_code != 0 or "Error: File not found" in output:
            return f"Error: File '{file_path}' not found"

        return output

    def write(self, file_path: str, content: str) -> WriteResult:
        """Create a new file using shell command."""
        # Encode content as base64 to avoid escaping issues
        content_b64 = base64.b64encode(content.encode("utf-8")).decode("ascii")

        cmd = _WRITE_COMMAND_TEMPLATE.format(
            file_path=file_path,
            content_b64=content_b64
        )
        result = self.execute(cmd)

        if result.exit_code != 0 or "Error:" in result.output:
            error_msg = result.output.strip() or f"Failed to write file '{file_path}'"
            return WriteResult(error=error_msg)

        return WriteResult(path=file_path, files_update=None)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Edit a file by replacing string occurrences using shell command."""
        # Encode strings as base64 to avoid escaping issues
        old_b64 = base64.b64encode(old_string.encode("utf-8")).decode("ascii")
        new_b64 = base64.b64encode(new_string.encode("utf-8")).decode("ascii")

        cmd = _EDIT_COMMAND_TEMPLATE.format(
            file_path=file_path,
            old_b64=old_b64,
            new_b64=new_b64,
            replace_all=replace_all
        )
        result = self.execute(cmd)

        exit_code = result.exit_code
        output = result.output.strip()

        if exit_code == 1:
            return EditResult(error=f"Error: String not found in file: '{old_string}'")
        if exit_code == 2:
            return EditResult(
                error=f"Error: String '{old_string}' appears multiple times. "
                      "Use replace_all=True to replace all occurrences."
            )
        if exit_code != 0:
            return EditResult(error=f"Error: File '{file_path}' not found")

        count = int(output)
        return EditResult(path=file_path, files_update=None, occurrences=count)

    def grep_raw(
        self,
        pattern: str,
        path: Optional[str] = None,
        glob: Optional[str] = None,
    ) -> List[GrepMatch]:
        """Search for pattern in files using grep command."""
        search_path = shlex.quote(path or ".")

        # Build grep command
        grep_opts = "-rHnF"  # recursive, with filename, with line number, fixed-strings

        glob_pattern = ""
        if glob:
            glob_pattern = f"--include='{glob}'"

        pattern_escaped = shlex.quote(pattern)

        cmd = f"grep {grep_opts} {glob_pattern} -e {pattern_escaped} {search_path} 2>/dev/null || true"
        result = self.execute(cmd)

        output = result.output.rstrip()
        if not output:
            return []

        matches: List[GrepMatch] = []
        for line in output.split("\n"):
            parts = line.split(":", 2)
            if len(parts) >= 3:
                matches.append({
                    "path": parts[0],
                    "line": int(parts[1]),
                    "text": parts[2],
                })

        return matches

    def glob_info(self, pattern: str, path: str = "/") -> List[FileInfo]:
        """Find files matching pattern using glob command."""
        pattern_b64 = base64.b64encode(pattern.encode("utf-8")).decode("ascii")
        path_b64 = base64.b64encode(path.encode("utf-8")).decode("ascii")

        cmd = _GLOB_COMMAND_TEMPLATE.format(
            path_b64=path_b64,
            pattern_b64=pattern_b64
        )
        result = self.execute(cmd)

        output = result.output.strip()
        if not output:
            return []

        file_infos: List[FileInfo] = []
        for line in output.split("\n"):
            try:
                data = json.loads(line)
                file_infos.append({
                    "path": data["path"],
                    "is_dir": data["is_dir"],
                })
            except json.JSONDecodeError:
                continue

        return file_infos

    # ========================================================================
    # File Upload/Download - Default implementations
    # ========================================================================

    def upload_files(
        self,
        files: List[Tuple[str, bytes]]
    ) -> List[FileUploadResponse]:
        """Upload multiple files to the sandbox.

        Default implementation uses base64 encoding via execute().
        Override for more efficient implementations.

        Args:
            files: List of (path, content) tuples

        Returns:
            List of FileUploadResponse for each file
        """
        responses = []
        for path, content in files:
            try:
                content_b64 = base64.b64encode(content).decode("ascii")
                cmd = f'''python3 -c "
import os
import base64
path = '{path}'
os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
content = base64.b64decode('{content_b64}')
with open(path, 'wb') as f:
    f.write(content)
" 2>&1'''
                result = self.execute(cmd)
                if result.exit_code != 0:
                    responses.append(FileUploadResponse(
                        path=path,
                        error="permission_denied"
                    ))
                else:
                    responses.append(FileUploadResponse(path=path))
            except Exception:
                responses.append(FileUploadResponse(
                    path=path,
                    error="permission_denied"
                ))
        return responses

    def download_files(self, paths: List[str]) -> List[FileDownloadResponse]:
        """Download multiple files from the sandbox.

        Default implementation uses base64 encoding via execute().
        Override for more efficient implementations.

        Args:
            paths: List of file paths to download

        Returns:
            List of FileDownloadResponse for each file
        """
        responses = []
        for path in paths:
            try:
                cmd = f'''python3 -c "
import os
import base64
import sys
path = '{path}'
if not os.path.isfile(path):
    sys.exit(1)
with open(path, 'rb') as f:
    print(base64.b64encode(f.read()).decode('ascii'))
" 2>&1'''
                result = self.execute(cmd)
                if result.exit_code != 0:
                    responses.append(FileDownloadResponse(
                        path=path,
                        error="file_not_found"
                    ))
                else:
                    content = base64.b64decode(result.output.strip())
                    responses.append(FileDownloadResponse(
                        path=path,
                        content=content
                    ))
            except Exception:
                responses.append(FileDownloadResponse(
                    path=path,
                    error="file_not_found"
                ))
        return responses

    # ========================================================================
    # Async Versions
    # ========================================================================

    async def als_info(self, path: str) -> List[FileInfo]:
        """Async version of ls_info."""
        return await asyncio.to_thread(self.ls_info, path)

    async def aread(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        """Async version of read."""
        return await asyncio.to_thread(self.read, file_path, offset, limit)

    async def awrite(self, file_path: str, content: str) -> WriteResult:
        """Async version of write."""
        return await asyncio.to_thread(self.write, file_path, content)

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

    async def agrep_raw(
        self,
        pattern: str,
        path: Optional[str] = None,
        glob: Optional[str] = None,
    ) -> List[GrepMatch]:
        """Async version of grep_raw."""
        return await asyncio.to_thread(self.grep_raw, pattern, path, glob)

    async def aglob_info(self, pattern: str, path: str = "/") -> List[FileInfo]:
        """Async version of glob_info."""
        return await asyncio.to_thread(self.glob_info, pattern, path)

    async def aupload_files(
        self,
        files: List[Tuple[str, bytes]]
    ) -> List[FileUploadResponse]:
        """Async version of upload_files."""
        return await asyncio.to_thread(self.upload_files, files)

    async def adownload_files(self, paths: List[str]) -> List[FileDownloadResponse]:
        """Async version of download_files."""
        return await asyncio.to_thread(self.download_files, paths)
