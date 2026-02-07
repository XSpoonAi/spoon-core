"""FilesystemBackend: Read and write files directly from the filesystem.

Security features:
- Secure path resolution with root containment when in virtual_mode
- Prevent symlink-following on file I/O using O_NOFOLLOW when available
- Max file size enforcement
"""

import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

from spoon_ai.backends.protocol import (
    BackendProtocol,
    EditResult,
    FileDownloadResponse,
    FileInfo,
    FileUploadResponse,
    GrepMatch,
    WriteResult,
)
from spoon_ai.backends.utils import (
    check_empty_content,
    format_content_with_line_numbers,
    perform_string_replacement,
    glob_match,
)


class FilesystemBackend(BackendProtocol):
    """Backend that reads and writes files directly from the filesystem.

    Files are accessed using their actual filesystem paths. Relative paths are
    resolved relative to the current working directory or root_dir.

    Example:
        ```python
        # Real filesystem access
        backend = FilesystemBackend()
        content = backend.read("/path/to/file.txt")

        # Sandboxed to a directory
        backend = FilesystemBackend(
            root_dir="/workspace",
            virtual_mode=True
        )
        # "/file.txt" maps to "/workspace/file.txt"
        ```
    """

    def __init__(
        self,
        root_dir: Optional[str | Path] = None,
        virtual_mode: bool = False,
        max_file_size_mb: int = 10,
    ) -> None:
        """Initialize filesystem backend.

        Args:
            root_dir: Optional root directory. If provided, all paths are
                     resolved relative to this directory.
            virtual_mode: If True, treat paths as virtual absolute paths under
                         root_dir. Disallows path traversal.
            max_file_size_mb: Maximum file size in MB for operations.
        """
        self.cwd = Path(root_dir).resolve() if root_dir else Path.cwd()
        self.virtual_mode = virtual_mode
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024

    def _resolve_path(self, key: str) -> Path:
        """Resolve a file path with security checks.

        When virtual_mode=True:
        - Treat paths as virtual absolute paths under self.cwd
        - Disallow traversal (.., ~)
        - Ensure resolved path stays within root

        When virtual_mode=False:
        - Absolute paths allowed as-is
        - Relative paths resolve under cwd

        Args:
            key: File path

        Returns:
            Resolved absolute Path

        Raises:
            ValueError: If path traversal detected
        """
        if self.virtual_mode:
            vpath = key if key.startswith("/") else "/" + key
            if ".." in vpath or vpath.startswith("~"):
                raise ValueError("Path traversal not allowed")

            full = (self.cwd / vpath.lstrip("/")).resolve()
            try:
                full.relative_to(self.cwd)
            except ValueError:
                raise ValueError(
                    f"Path:{full} outside root directory: {self.cwd}"
                ) from None
            return full

        path = Path(key)
        if path.is_absolute():
            return path
        return (self.cwd / path).resolve()

    def _to_virtual_path(self, abs_path: str | Path) -> str:
        """Convert absolute path to virtual path.

        Args:
            abs_path: Absolute path

        Returns:
            Virtual path (starts with /)
        """
        abs_path_str = str(abs_path)
        cwd_str = str(self.cwd)

        if not cwd_str.endswith("/"):
            cwd_str += "/"

        if abs_path_str.startswith(cwd_str):
            relative = abs_path_str[len(cwd_str):]
            return "/" + relative
        elif abs_path_str.startswith(str(self.cwd)):
            relative = abs_path_str[len(str(self.cwd)):].lstrip("/")
            return "/" + relative
        else:
            return abs_path_str

    def ls_info(self, path: str) -> list[FileInfo]:
        """List files and directories in the specified directory (non-recursive).

        Args:
            path: Absolute directory path.

        Returns:
            List of FileInfo dicts for files and directories.
        """
        try:
            dir_path = self._resolve_path(path)
        except ValueError:
            return []

        if not dir_path.exists() or not dir_path.is_dir():
            return []

        results: list[FileInfo] = []

        try:
            for child_path in dir_path.iterdir():
                try:
                    is_file = child_path.is_file()
                    is_dir = child_path.is_dir()
                except OSError:
                    continue

                abs_path = str(child_path)

                if self.virtual_mode:
                    virt_path = self._to_virtual_path(abs_path)
                else:
                    virt_path = abs_path

                if is_file:
                    try:
                        st = child_path.stat()
                        results.append({
                            "path": virt_path,
                            "is_dir": False,
                            "size": int(st.st_size),
                            "modified_at": datetime.fromtimestamp(st.st_mtime).isoformat(),
                        })
                    except OSError:
                        results.append({"path": virt_path, "is_dir": False})

                elif is_dir:
                    try:
                        st = child_path.stat()
                        results.append({
                            "path": virt_path + "/",
                            "is_dir": True,
                            "size": 0,
                            "modified_at": datetime.fromtimestamp(st.st_mtime).isoformat(),
                        })
                    except OSError:
                        results.append({"path": virt_path + "/", "is_dir": True})

        except (OSError, PermissionError):
            pass

        results.sort(key=lambda x: x.get("path", ""))
        return results

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        """Read file content with line numbers.

        Args:
            file_path: Absolute or relative file path.
            offset: Line offset to start reading from (0-indexed).
            limit: Maximum number of lines to read.

        Returns:
            Formatted file content with line numbers, or error message.
        """
        try:
            resolved_path = self._resolve_path(file_path)
        except ValueError as e:
            return f"Error: {e}"

        if not resolved_path.exists() or not resolved_path.is_file():
            return f"Error: File '{file_path}' not found"

        try:
            # Open with O_NOFOLLOW where available
            fd = os.open(
                resolved_path,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            )
            with os.fdopen(fd, "r", encoding="utf-8") as f:
                content = f.read()

            empty_msg = check_empty_content(content)
            if empty_msg:
                return empty_msg

            lines = content.splitlines()
            start_idx = offset
            end_idx = min(start_idx + limit, len(lines))

            if start_idx >= len(lines):
                return f"Error: Line offset {offset} exceeds file length ({len(lines)} lines)"

            selected_lines = lines[start_idx:end_idx]
            return format_content_with_line_numbers(selected_lines, start_line=start_idx + 1)

        except (OSError, UnicodeDecodeError) as e:
            return f"Error reading file '{file_path}': {e}"

    def write(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        """Create a new file with content.

        Returns WriteResult. External storage sets files_update=None.
        """
        try:
            resolved_path = self._resolve_path(file_path)
        except ValueError as e:
            return WriteResult(error=str(e))

        if resolved_path.exists():
            return WriteResult(
                error=f"Cannot write to {file_path} because it already exists. "
                      f"Read and then make an edit, or write to a new path."
            )

        try:
            # Create parent directories if needed
            resolved_path.parent.mkdir(parents=True, exist_ok=True)

            # Prefer O_NOFOLLOW to avoid writing through symlinks
            flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW

            fd = os.open(resolved_path, flags, 0o644)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)

            return WriteResult(path=file_path, files_update=None)

        except (OSError, UnicodeEncodeError) as e:
            return WriteResult(error=f"Error writing file '{file_path}': {e}")

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Edit a file by replacing string occurrences.

        Returns EditResult. External storage sets files_update=None.
        """
        try:
            resolved_path = self._resolve_path(file_path)
        except ValueError as e:
            return EditResult(error=str(e))

        if not resolved_path.exists() or not resolved_path.is_file():
            return EditResult(error=f"Error: File '{file_path}' not found")

        try:
            # Read securely
            fd = os.open(
                resolved_path,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            )
            with os.fdopen(fd, "r", encoding="utf-8") as f:
                content = f.read()

            result = perform_string_replacement(
                content, old_string, new_string, replace_all
            )

            if isinstance(result, str):
                return EditResult(error=result)

            new_content, occurrences = result

            # Write securely
            flags = os.O_WRONLY | os.O_TRUNC
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW

            fd = os.open(resolved_path, flags)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(new_content)

            return EditResult(
                path=file_path,
                files_update=None,
                occurrences=int(occurrences)
            )

        except (OSError, UnicodeDecodeError, UnicodeEncodeError) as e:
            return EditResult(error=f"Error editing file '{file_path}': {e}")

    def grep_raw(
        self,
        pattern: str,
        path: Optional[str] = None,
        glob: Optional[str] = None,
    ) -> list[GrepMatch] | str:
        """Search for pattern in files.

        Uses ripgrep if available, falls back to Python search.
        """
        # Validate regex
        try:
            re.compile(pattern)
        except re.error as e:
            return f"Invalid regex pattern: {e}"

        # Resolve base path
        try:
            base_full = self._resolve_path(path or ".")
        except ValueError:
            return []

        if not base_full.exists():
            return []

        # Try ripgrep first
        results = self._ripgrep_search(pattern, base_full, glob)
        if results is None:
            results = self._python_search(pattern, base_full, glob)

        matches: list[GrepMatch] = []
        for fpath, items in results.items():
            for line_num, line_text in items:
                matches.append({
                    "path": fpath,
                    "line": int(line_num),
                    "text": line_text
                })

        return matches

    def _ripgrep_search(
        self,
        pattern: str,
        base_full: Path,
        include_glob: Optional[str]
    ) -> Optional[dict[str, list[tuple[int, str]]]]:
        """Search using ripgrep."""
        cmd = ["rg", "--json"]
        if include_glob:
            cmd.extend(["--glob", include_glob])
        cmd.extend(["--", pattern, str(base_full)])

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return None

        results: dict[str, list[tuple[int, str]]] = {}

        for line in proc.stdout.splitlines():
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            if data.get("type") != "match":
                continue

            pdata = data.get("data", {})
            ftext = pdata.get("path", {}).get("text")
            if not ftext:
                continue

            p = Path(ftext)
            if self.virtual_mode:
                try:
                    virt = "/" + str(p.resolve().relative_to(self.cwd))
                except Exception:
                    continue
            else:
                virt = str(p)

            ln = pdata.get("line_number")
            lt = pdata.get("lines", {}).get("text", "").rstrip("\n")
            if ln is None:
                continue

            results.setdefault(virt, []).append((int(ln), lt))

        return results

    def _python_search(
        self,
        pattern: str,
        base_full: Path,
        include_glob: Optional[str]
    ) -> dict[str, list[tuple[int, str]]]:
        """Search using Python (fallback)."""
        try:
            regex = re.compile(pattern)
        except re.error:
            return {}

        results: dict[str, list[tuple[int, str]]] = {}
        root = base_full if base_full.is_dir() else base_full.parent

        for fp in root.rglob("*"):
            if not fp.is_file():
                continue

            if include_glob and not glob_match(fp.name, include_glob):
                continue

            try:
                if fp.stat().st_size > self.max_file_size_bytes:
                    continue
            except OSError:
                continue

            try:
                content = fp.read_text()
            except (UnicodeDecodeError, PermissionError, OSError):
                continue

            for line_num, line in enumerate(content.splitlines(), 1):
                if regex.search(line):
                    if self.virtual_mode:
                        try:
                            virt_path = "/" + str(fp.resolve().relative_to(self.cwd))
                        except Exception:
                            continue
                    else:
                        virt_path = str(fp)

                    results.setdefault(virt_path, []).append((line_num, line))

        return results

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        """Find files matching glob pattern."""
        if pattern.startswith("/"):
            pattern = pattern.lstrip("/")

        try:
            search_path = self.cwd if path == "/" else self._resolve_path(path)
        except ValueError:
            return []

        if not search_path.exists() or not search_path.is_dir():
            return []

        results: list[FileInfo] = []

        try:
            for matched_path in search_path.rglob(pattern):
                try:
                    is_file = matched_path.is_file()
                except OSError:
                    continue

                if not is_file:
                    continue

                abs_path = str(matched_path)

                if self.virtual_mode:
                    virt = self._to_virtual_path(abs_path)
                else:
                    virt = abs_path

                try:
                    st = matched_path.stat()
                    results.append({
                        "path": virt,
                        "is_dir": False,
                        "size": int(st.st_size),
                        "modified_at": datetime.fromtimestamp(st.st_mtime).isoformat(),
                    })
                except OSError:
                    results.append({"path": virt, "is_dir": False})

        except (OSError, ValueError):
            pass

        results.sort(key=lambda x: x.get("path", ""))
        return results

    def upload_files(
        self, files: list[tuple[str, bytes]]
    ) -> list[FileUploadResponse]:
        """Upload multiple files to the filesystem."""
        responses: list[FileUploadResponse] = []

        for path, content in files:
            try:
                resolved_path = self._resolve_path(path)

                # Create parent directories if needed
                resolved_path.parent.mkdir(parents=True, exist_ok=True)

                flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
                if hasattr(os, "O_NOFOLLOW"):
                    flags |= os.O_NOFOLLOW

                fd = os.open(resolved_path, flags, 0o644)
                with os.fdopen(fd, "wb") as f:
                    f.write(content)

                responses.append(FileUploadResponse(path=path, error=None))

            except FileNotFoundError:
                responses.append(FileUploadResponse(path=path, error="file_not_found"))
            except PermissionError:
                responses.append(FileUploadResponse(path=path, error="permission_denied"))
            except (ValueError, OSError):
                responses.append(FileUploadResponse(path=path, error="invalid_path"))

        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download multiple files from the filesystem."""
        responses: list[FileDownloadResponse] = []

        for path in paths:
            try:
                resolved_path = self._resolve_path(path)
                fd = os.open(
                    resolved_path,
                    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                )
                with os.fdopen(fd, "rb") as f:
                    content = f.read()
                responses.append(FileDownloadResponse(path=path, content=content, error=None))

            except FileNotFoundError:
                responses.append(FileDownloadResponse(path=path, content=None, error="file_not_found"))
            except PermissionError:
                responses.append(FileDownloadResponse(path=path, content=None, error="permission_denied"))
            except IsADirectoryError:
                responses.append(FileDownloadResponse(path=path, content=None, error="is_directory"))
            except ValueError:
                responses.append(FileDownloadResponse(path=path, content=None, error="invalid_path"))

        return responses


def create_filesystem_backend(
    root_dir: Optional[str | Path] = None,
    virtual_mode: bool = False,
    max_file_size_mb: int = 10,
) -> FilesystemBackend:
    """Create a FilesystemBackend.

    Args:
        root_dir: Root directory for file operations.
        virtual_mode: If True, sandbox paths to root_dir.
        max_file_size_mb: Maximum file size for operations.

    Returns:
        FilesystemBackend instance.

    Example:
        ```python
        # Access real filesystem
        backend = create_filesystem_backend()

        # Sandboxed to workspace
        backend = create_filesystem_backend(
            root_dir="/workspace",
            virtual_mode=True
        )
        ```
    """
    return FilesystemBackend(
        root_dir=root_dir,
        virtual_mode=virtual_mode,
        max_file_size_mb=max_file_size_mb
    )
