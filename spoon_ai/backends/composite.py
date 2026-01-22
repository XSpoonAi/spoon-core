"""CompositeBackend: Route operations to different backends based on path prefix.

Enables mixing multiple storage backends, e.g.:
- /ephemeral/* -> StateBackend (in-memory)
- /persistent/* -> StoreBackend (database)
- /local/* -> FilesystemBackend (filesystem)
"""

from collections import defaultdict
from typing import Optional

from spoon_ai.backends.protocol import (
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


class CompositeBackend:
    """Route file operations to different backends based on path prefix.

    The CompositeBackend dispatches operations to specialized backends based
    on path prefixes. This allows mixing ephemeral, persistent, and filesystem
    storage in a single agent.

    Example:
        ```python
        from spoon_ai.backends import (
            CompositeBackend,
            StateBackend,
            StoreBackend,
            FilesystemBackend,
            BackendRuntime,
        )

        # Create backends
        runtime = BackendRuntime(state={"files": {}})
        state_backend = StateBackend(runtime)
        store_backend = StoreBackend(SQLiteStore("agent.db"))
        fs_backend = FilesystemBackend(root_dir="/workspace", virtual_mode=True)

        # Create composite with routes
        backend = CompositeBackend(
            default=state_backend,
            routes={
                "/persistent/": store_backend,
                "/local/": fs_backend,
            }
        )

        # Operations route automatically
        backend.write("/temp.txt", "Ephemeral")      # -> state_backend
        backend.write("/persistent/note.txt", "DB")  # -> store_backend
        backend.write("/local/code.py", "File")      # -> fs_backend
        ```
    """

    def __init__(
        self,
        default: BackendProtocol,
        routes: dict[str, BackendProtocol],
    ) -> None:
        """Initialize CompositeBackend.

        Args:
            default: Default backend for unmatched paths.
            routes: Dict mapping path prefixes to backends.
                   Prefixes should end with '/' (e.g., "/persistent/").
        """
        self.default = default
        self.routes = routes

        # Sort routes by length (longest first) for correct prefix matching
        self.sorted_routes = sorted(
            routes.items(),
            key=lambda x: len(x[0]),
            reverse=True
        )

    def _get_backend_and_key(
        self, key: str
    ) -> tuple[BackendProtocol, str]:
        """Determine which backend handles this key and strip prefix.

        Args:
            key: Original file path

        Returns:
            Tuple of (backend, stripped_key) where stripped_key has the route
            prefix removed (but keeps leading slash).
        """
        for prefix, backend in self.sorted_routes:
            if key.startswith(prefix):
                # Strip full prefix, ensure leading slash remains
                suffix = key[len(prefix):]
                stripped_key = f"/{suffix}" if suffix else "/"
                return backend, stripped_key

        return self.default, key

    # ========================================================================
    # List Operations
    # ========================================================================

    def ls_info(self, path: str) -> list[FileInfo]:
        """List files and directories in the specified directory."""
        # Check if path matches a specific route
        for route_prefix, backend in self.sorted_routes:
            if path.startswith(route_prefix.rstrip("/")):
                suffix = path[len(route_prefix):]
                search_path = f"/{suffix}" if suffix else "/"
                infos = backend.ls_info(search_path)

                # Add route prefix back to paths
                prefixed: list[FileInfo] = []
                for fi in infos:
                    fi_copy = dict(fi)
                    fi_copy["path"] = f"{route_prefix[:-1]}{fi['path']}"
                    prefixed.append(fi_copy)
                return prefixed

        # At root, aggregate default and all routed backends
        if path == "/":
            results: list[FileInfo] = []
            results.extend(self.default.ls_info(path))

            # Add route directories
            for route_prefix, backend in self.sorted_routes:
                results.append({
                    "path": route_prefix,
                    "is_dir": True,
                    "size": 0,
                    "modified_at": "",
                })

            results.sort(key=lambda x: x.get("path", ""))
            return results

        # Path doesn't match a route: query only default backend
        return self.default.ls_info(path)

    async def als_info(self, path: str) -> list[FileInfo]:
        """Async version of ls_info."""
        for route_prefix, backend in self.sorted_routes:
            if path.startswith(route_prefix.rstrip("/")):
                suffix = path[len(route_prefix):]
                search_path = f"/{suffix}" if suffix else "/"
                infos = await backend.als_info(search_path)

                prefixed: list[FileInfo] = []
                for fi in infos:
                    fi_copy = dict(fi)
                    fi_copy["path"] = f"{route_prefix[:-1]}{fi['path']}"
                    prefixed.append(fi_copy)
                return prefixed

        if path == "/":
            results: list[FileInfo] = []
            results.extend(await self.default.als_info(path))

            for route_prefix, backend in self.sorted_routes:
                results.append({
                    "path": route_prefix,
                    "is_dir": True,
                    "size": 0,
                    "modified_at": "",
                })

            results.sort(key=lambda x: x.get("path", ""))
            return results

        return await self.default.als_info(path)

    # ========================================================================
    # Read Operations
    # ========================================================================

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        """Read file content, routing to appropriate backend."""
        backend, stripped_key = self._get_backend_and_key(file_path)
        return backend.read(stripped_key, offset=offset, limit=limit)

    async def aread(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        """Async version of read."""
        backend, stripped_key = self._get_backend_and_key(file_path)
        return await backend.aread(stripped_key, offset=offset, limit=limit)

    # ========================================================================
    # Write Operations
    # ========================================================================

    def write(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        """Create a new file, routing to appropriate backend."""
        backend, stripped_key = self._get_backend_and_key(file_path)
        result = backend.write(stripped_key, content)

        # Merge state updates if needed
        if result.files_update:
            self._merge_state_update(result.files_update)

        return result

    async def awrite(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        """Async version of write."""
        backend, stripped_key = self._get_backend_and_key(file_path)
        result = await backend.awrite(stripped_key, content)

        if result.files_update:
            self._merge_state_update(result.files_update)

        return result

    def _merge_state_update(self, files_update: dict) -> None:
        """Merge state update into default backend if it uses state."""
        try:
            runtime = getattr(self.default, "runtime", None)
            if runtime is not None:
                state = runtime.state
                files = state.get("files", {})
                files.update(files_update)
                state["files"] = files
        except Exception:
            pass

    # ========================================================================
    # Edit Operations
    # ========================================================================

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Edit a file, routing to appropriate backend."""
        backend, stripped_key = self._get_backend_and_key(file_path)
        result = backend.edit(
            stripped_key, old_string, new_string, replace_all=replace_all
        )

        if result.files_update:
            self._merge_state_update(result.files_update)

        return result

    async def aedit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Async version of edit."""
        backend, stripped_key = self._get_backend_and_key(file_path)
        result = await backend.aedit(
            stripped_key, old_string, new_string, replace_all=replace_all
        )

        if result.files_update:
            self._merge_state_update(result.files_update)

        return result

    # ========================================================================
    # Search Operations
    # ========================================================================

    def grep_raw(
        self,
        pattern: str,
        path: Optional[str] = None,
        glob: Optional[str] = None,
    ) -> list[GrepMatch] | str:
        """Search for pattern in files."""
        # If path targets a specific route
        for route_prefix, backend in self.sorted_routes:
            if path is not None and path.startswith(route_prefix.rstrip("/")):
                search_path = path[len(route_prefix) - 1:]
                raw = backend.grep_raw(
                    pattern, search_path if search_path else "/", glob
                )
                if isinstance(raw, str):
                    return raw
                # Add prefix back to paths
                return [
                    {**m, "path": f"{route_prefix[:-1]}{m['path']}"}
                    for m in raw
                ]

        # Search default and all routed backends
        all_matches: list[GrepMatch] = []

        raw_default = self.default.grep_raw(pattern, path, glob)
        if isinstance(raw_default, str):
            return raw_default
        all_matches.extend(raw_default)

        for route_prefix, backend in self.routes.items():
            raw = backend.grep_raw(pattern, "/", glob)
            if isinstance(raw, str):
                return raw
            all_matches.extend(
                {**m, "path": f"{route_prefix[:-1]}{m['path']}"}
                for m in raw
            )

        return all_matches

    async def agrep_raw(
        self,
        pattern: str,
        path: Optional[str] = None,
        glob: Optional[str] = None,
    ) -> list[GrepMatch] | str:
        """Async version of grep_raw."""
        for route_prefix, backend in self.sorted_routes:
            if path is not None and path.startswith(route_prefix.rstrip("/")):
                search_path = path[len(route_prefix) - 1:]
                raw = await backend.agrep_raw(
                    pattern, search_path if search_path else "/", glob
                )
                if isinstance(raw, str):
                    return raw
                return [
                    {**m, "path": f"{route_prefix[:-1]}{m['path']}"}
                    for m in raw
                ]

        all_matches: list[GrepMatch] = []

        raw_default = await self.default.agrep_raw(pattern, path, glob)
        if isinstance(raw_default, str):
            return raw_default
        all_matches.extend(raw_default)

        for route_prefix, backend in self.routes.items():
            raw = await backend.agrep_raw(pattern, "/", glob)
            if isinstance(raw, str):
                return raw
            all_matches.extend(
                {**m, "path": f"{route_prefix[:-1]}{m['path']}"}
                for m in raw
            )

        return all_matches

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        """Find files matching glob pattern."""
        results: list[FileInfo] = []

        # Route based on path
        for route_prefix, backend in self.sorted_routes:
            if path.startswith(route_prefix.rstrip("/")):
                search_path = path[len(route_prefix) - 1:]
                infos = backend.glob_info(
                    pattern, search_path if search_path else "/"
                )
                return [
                    {**fi, "path": f"{route_prefix[:-1]}{fi['path']}"}
                    for fi in infos
                ]

        # Search default and all routed backends
        results.extend(self.default.glob_info(pattern, path))

        for route_prefix, backend in self.routes.items():
            infos = backend.glob_info(pattern, "/")
            results.extend(
                {**fi, "path": f"{route_prefix[:-1]}{fi['path']}"}
                for fi in infos
            )

        results.sort(key=lambda x: x.get("path", ""))
        return results

    async def aglob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        """Async version of glob_info."""
        results: list[FileInfo] = []

        for route_prefix, backend in self.sorted_routes:
            if path.startswith(route_prefix.rstrip("/")):
                search_path = path[len(route_prefix) - 1:]
                infos = await backend.aglob_info(
                    pattern, search_path if search_path else "/"
                )
                return [
                    {**fi, "path": f"{route_prefix[:-1]}{fi['path']}"}
                    for fi in infos
                ]

        results.extend(await self.default.aglob_info(pattern, path))

        for route_prefix, backend in self.routes.items():
            infos = await backend.aglob_info(pattern, "/")
            results.extend(
                {**fi, "path": f"{route_prefix[:-1]}{fi['path']}"}
                for fi in infos
            )

        results.sort(key=lambda x: x.get("path", ""))
        return results

    # ========================================================================
    # Execution (delegates to default if sandbox)
    # ========================================================================

    def execute(self, command: str) -> ExecuteResponse:
        """Execute a command via the default backend.

        Only works if default backend implements SandboxBackendProtocol.
        """
        if isinstance(self.default, SandboxBackendProtocol):
            return self.default.execute(command)

        raise NotImplementedError(
            "Default backend doesn't support command execution. "
            "Provide a backend that implements SandboxBackendProtocol."
        )

    async def aexecute(self, command: str) -> ExecuteResponse:
        """Async version of execute."""
        if isinstance(self.default, SandboxBackendProtocol):
            return await self.default.aexecute(command)

        raise NotImplementedError(
            "Default backend doesn't support command execution. "
            "Provide a backend that implements SandboxBackendProtocol."
        )

    # ========================================================================
    # Batch File Operations
    # ========================================================================

    def upload_files(
        self, files: list[tuple[str, bytes]]
    ) -> list[FileUploadResponse]:
        """Upload multiple files, batching by backend."""
        results: list[Optional[FileUploadResponse]] = [None] * len(files)

        # Group files by backend
        backend_batches: dict[BackendProtocol, list[tuple[int, str, bytes]]] = (
            defaultdict(list)
        )

        for idx, (path, content) in enumerate(files):
            backend, stripped_path = self._get_backend_and_key(path)
            backend_batches[backend].append((idx, stripped_path, content))

        # Process each backend's batch
        for backend, batch in backend_batches.items():
            indices, stripped_paths, contents = zip(*batch, strict=False)
            batch_files = list(zip(stripped_paths, contents, strict=False))

            batch_responses = backend.upload_files(batch_files)

            for i, orig_idx in enumerate(indices):
                results[orig_idx] = FileUploadResponse(
                    path=files[orig_idx][0],
                    error=batch_responses[i].error if i < len(batch_responses) else None,
                )

        return results  # type: ignore

    async def aupload_files(
        self, files: list[tuple[str, bytes]]
    ) -> list[FileUploadResponse]:
        """Async version of upload_files."""
        results: list[Optional[FileUploadResponse]] = [None] * len(files)

        backend_batches: dict[BackendProtocol, list[tuple[int, str, bytes]]] = (
            defaultdict(list)
        )

        for idx, (path, content) in enumerate(files):
            backend, stripped_path = self._get_backend_and_key(path)
            backend_batches[backend].append((idx, stripped_path, content))

        for backend, batch in backend_batches.items():
            indices, stripped_paths, contents = zip(*batch, strict=False)
            batch_files = list(zip(stripped_paths, contents, strict=False))

            batch_responses = await backend.aupload_files(batch_files)

            for i, orig_idx in enumerate(indices):
                results[orig_idx] = FileUploadResponse(
                    path=files[orig_idx][0],
                    error=batch_responses[i].error if i < len(batch_responses) else None,
                )

        return results  # type: ignore

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download multiple files, batching by backend."""
        results: list[Optional[FileDownloadResponse]] = [None] * len(paths)

        backend_batches: dict[BackendProtocol, list[tuple[int, str]]] = (
            defaultdict(list)
        )

        for idx, path in enumerate(paths):
            backend, stripped_path = self._get_backend_and_key(path)
            backend_batches[backend].append((idx, stripped_path))

        for backend, batch in backend_batches.items():
            indices, stripped_paths = zip(*batch, strict=False)

            batch_responses = backend.download_files(list(stripped_paths))

            for i, orig_idx in enumerate(indices):
                results[orig_idx] = FileDownloadResponse(
                    path=paths[orig_idx],
                    content=batch_responses[i].content if i < len(batch_responses) else None,
                    error=batch_responses[i].error if i < len(batch_responses) else None,
                )

        return results  # type: ignore

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Async version of download_files."""
        results: list[Optional[FileDownloadResponse]] = [None] * len(paths)

        backend_batches: dict[BackendProtocol, list[tuple[int, str]]] = (
            defaultdict(list)
        )

        for idx, path in enumerate(paths):
            backend, stripped_path = self._get_backend_and_key(path)
            backend_batches[backend].append((idx, stripped_path))

        for backend, batch in backend_batches.items():
            indices, stripped_paths = zip(*batch, strict=False)

            batch_responses = await backend.adownload_files(list(stripped_paths))

            for i, orig_idx in enumerate(indices):
                results[orig_idx] = FileDownloadResponse(
                    path=paths[orig_idx],
                    content=batch_responses[i].content if i < len(batch_responses) else None,
                    error=batch_responses[i].error if i < len(batch_responses) else None,
                )

        return results  # type: ignore


def create_composite_backend(
    default: BackendProtocol,
    routes: dict[str, BackendProtocol],
) -> CompositeBackend:
    """Create a CompositeBackend.

    Args:
        default: Default backend for unmatched paths.
        routes: Dict mapping path prefixes to backends.

    Returns:
        CompositeBackend instance.

    Example:
        ```python
        backend = create_composite_backend(
            default=state_backend,
            routes={
                "/db/": store_backend,
                "/files/": fs_backend,
            }
        )
        ```
    """
    return CompositeBackend(default=default, routes=routes)
