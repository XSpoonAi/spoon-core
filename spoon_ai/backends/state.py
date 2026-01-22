"""StateBackend: Store files in agent state (ephemeral).

Files persist within a conversation thread but not across threads.
State is automatically checkpointed after each agent step.
"""

from typing import Any, Optional

from spoon_ai.backends.protocol import (
    BackendProtocol,
    BackendRuntime,
    EditResult,
    FileInfo,
    GrepMatch,
    WriteResult,
)
from spoon_ai.backends.utils import (
    glob_search_files,
    create_file_data,
    file_data_to_string,
    format_read_response,
    grep_matches_from_files,
    perform_string_replacement,
    update_file_data,
)


class StateBackend(BackendProtocol):
    """Backend that stores files in agent state (ephemeral).

    Uses agent's state management and checkpointing. Files persist within
    a conversation thread but not across threads. State is automatically
    checkpointed after each agent step.

    Example:
        ```python
        runtime = BackendRuntime(state={"files": {}})
        backend = StateBackend(runtime)

        # Write a file
        result = backend.write("/hello.txt", "Hello, World!")

        # Read the file
        content = backend.read("/hello.txt")
        ```
    """

    def __init__(self, runtime: BackendRuntime):
        """Initialize StateBackend with runtime.

        Args:
            runtime: BackendRuntime instance providing state access.
        """
        self.runtime = runtime

    def _get_files(self) -> dict[str, Any]:
        """Get files dictionary from state."""
        return self.runtime.state.get("files", {})

    def _set_files(self, files: dict[str, Any]) -> None:
        """Set files dictionary in state."""
        self.runtime.state["files"] = files

    def ls_info(self, path: str) -> list[FileInfo]:
        """List files and directories in the specified directory (non-recursive).

        Args:
            path: Absolute path to directory.

        Returns:
            List of FileInfo dicts for files and directories in the directory.
        """
        files = self._get_files()
        infos: list[FileInfo] = []
        subdirs: set[str] = set()

        # Normalize path to have trailing slash
        normalized_path = path if path.endswith("/") else path + "/"

        for k, fd in files.items():
            # Check if file is in the specified directory or subdirectory
            if not k.startswith(normalized_path):
                continue

            # Get the relative path after the directory
            relative = k[len(normalized_path):]

            # If relative path contains '/', it's in a subdirectory
            if "/" in relative:
                subdir_name = relative.split("/")[0]
                subdirs.add(normalized_path + subdir_name + "/")
                continue

            # This is a file directly in the current directory
            size = len("\n".join(fd.get("content", [])))
            infos.append({
                "path": k,
                "is_dir": False,
                "size": int(size),
                "modified_at": fd.get("modified_at", ""),
            })

        # Add directories to the results
        for subdir in sorted(subdirs):
            infos.append({
                "path": subdir,
                "is_dir": True,
                "size": 0,
                "modified_at": "",
            })

        infos.sort(key=lambda x: x.get("path", ""))
        return infos

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        """Read file content with line numbers.

        Args:
            file_path: Absolute file path.
            offset: Line offset to start reading from (0-indexed).
            limit: Maximum number of lines to read.

        Returns:
            Formatted file content with line numbers, or error message.
        """
        files = self._get_files()
        file_data = files.get(file_path)

        if file_data is None:
            return f"Error: File '{file_path}' not found"

        return format_read_response(file_data, offset, limit)

    def write(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        """Create a new file with content.

        Returns WriteResult with files_update to update state.
        """
        files = self._get_files()

        if file_path in files:
            return WriteResult(
                error=f"Cannot write to {file_path} because it already exists. "
                      f"Read and then make an edit, or write to a new path."
            )

        new_file_data = create_file_data(content)

        # Update state
        files[file_path] = new_file_data
        self._set_files(files)

        return WriteResult(
            path=file_path,
            files_update={file_path: new_file_data}
        )

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Edit a file by replacing string occurrences.

        Returns EditResult with files_update and occurrences.
        """
        files = self._get_files()
        file_data = files.get(file_path)

        if file_data is None:
            return EditResult(error=f"Error: File '{file_path}' not found")

        content = file_data_to_string(file_data)
        result = perform_string_replacement(content, old_string, new_string, replace_all)

        if isinstance(result, str):
            return EditResult(error=result)

        new_content, occurrences = result
        new_file_data = update_file_data(file_data, new_content)

        # Update state
        files[file_path] = new_file_data
        self._set_files(files)

        return EditResult(
            path=file_path,
            files_update={file_path: new_file_data},
            occurrences=int(occurrences)
        )

    def grep_raw(
        self,
        pattern: str,
        path: Optional[str] = None,
        glob: Optional[str] = None,
    ) -> list[GrepMatch] | str:
        """Search for pattern in files."""
        files = self._get_files()
        return grep_matches_from_files(files, pattern, path or "/", glob)

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        """Get FileInfo for files matching glob pattern."""
        files = self._get_files()
        result = glob_search_files(files, pattern, path)

        if result == "No files found":
            return []

        paths = result.split("\n")
        infos: list[FileInfo] = []

        for p in paths:
            fd = files.get(p)
            size = len("\n".join(fd.get("content", []))) if fd else 0
            infos.append({
                "path": p,
                "is_dir": False,
                "size": int(size),
                "modified_at": fd.get("modified_at", "") if fd else "",
            })

        return infos


def create_state_backend(
    initial_files: Optional[dict[str, Any]] = None
) -> tuple[StateBackend, BackendRuntime]:
    """Create a StateBackend with optional initial files.

    Args:
        initial_files: Optional dict of file paths to FileData.

    Returns:
        Tuple of (StateBackend, BackendRuntime).

    Example:
        ```python
        backend, runtime = create_state_backend()
        backend.write("/hello.txt", "Hello!")
        ```
    """
    runtime = BackendRuntime(state={"files": initial_files or {}})
    backend = StateBackend(runtime)
    return backend, runtime
