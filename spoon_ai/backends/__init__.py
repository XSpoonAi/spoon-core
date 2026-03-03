"""Pluggable Memory Backends for Deep Agents.

This module provides a unified interface for file operations across different
storage backends. Compatible with LangChain DeepAgents backend architecture.

Backend Types:
- StateBackend: Ephemeral in-memory storage (per-thread)
- FilesystemBackend: Real filesystem access (optionally sandboxed)
- StoreBackend: Persistent key-value storage (cross-thread)
- CompositeBackend: Route operations to multiple backends by path prefix
- BaseSandbox: Abstract base class for remote sandboxes (Docker, Modal, etc.)

Example Usage:
    ```python
    from spoon_ai.backends import (
        StateBackend,
        FilesystemBackend,
        StoreBackend,
        CompositeBackend,
        BaseSandbox,
        BackendRuntime,
        create_state_backend,
        create_filesystem_backend,
        create_store_backend,
        create_composite_backend,
    )

    # 1. Simple ephemeral storage
    backend, runtime = create_state_backend()
    backend.write("/notes.txt", "Hello!")
    print(backend.read("/notes.txt"))

    # 2. Real filesystem access
    backend = create_filesystem_backend(
        root_dir="/workspace",
        virtual_mode=True  # Sandbox to root_dir
    )

    # 3. Persistent database storage
    backend = create_store_backend(db_path="agent.db")

    # 4. Mixed storage with routing
    state_backend, _ = create_state_backend()
    store_backend = create_store_backend()
    fs_backend = create_filesystem_backend()

    backend = create_composite_backend(
        default=state_backend,
        routes={
            "/persistent/": store_backend,
            "/local/": fs_backend,
        }
    )

    # Operations route automatically
    backend.write("/temp.txt", "Ephemeral")         # -> state
    backend.write("/persistent/note.txt", "Saved")  # -> database
    backend.write("/local/code.py", "# Code")       # -> filesystem

    # 5. Remote sandbox (Docker, Modal, etc.)
    class DockerSandbox(BaseSandbox):
        def __init__(self, container_id: str):
            self._container_id = container_id

        @property
        def id(self) -> str:
            return f"docker-{self._container_id}"

        def execute(self, command: str) -> ExecuteResponse:
            # Run command in Docker container
            result = docker_exec(self._container_id, command)
            return ExecuteResponse(output=result.output, exit_code=result.exit_code)

    sandbox = DockerSandbox("my-container")
    sandbox.write("/app/config.json", '{"key": "value"}')
    content = sandbox.read("/app/config.json")
    ```
"""

# Protocol and data types
from spoon_ai.backends.protocol import (
    # Base protocol
    BackendProtocol,
    SandboxBackendProtocol,

    # Runtime context
    BackendRuntime,

    # Result types
    WriteResult,
    EditResult,
    ExecuteResponse,
    FileUploadResponse,
    FileDownloadResponse,

    # Data types
    FileInfo,
    GrepMatch,
    FileOperationError,

    # Type aliases
    BackendFactory,
    BACKEND_TYPES,
)

# State backend (ephemeral)
from spoon_ai.backends.state import (
    StateBackend,
    create_state_backend,
)

# Filesystem backend
from spoon_ai.backends.filesystem import (
    FilesystemBackend,
    create_filesystem_backend,
)

# Store backend (persistent)
from spoon_ai.backends.store import (
    StoreBackend,
    BaseStore,
    InMemoryStore,
    SQLiteStore,
    create_store_backend,
)

# Composite backend (routing)
from spoon_ai.backends.composite import (
    CompositeBackend,
    create_composite_backend,
)

# Base sandbox (abstract class for remote sandboxes)
from spoon_ai.backends.sandbox import (
    BaseSandbox,
)

# Utilities
from spoon_ai.backends.utils import (
    format_content_with_line_numbers,
    create_file_data,
    update_file_data,
    file_data_to_string,
    perform_string_replacement,
    validate_path,
    glob_match,
    grep_matches_from_files,
    glob_search_files,
)

__all__ = [
    # Protocol
    "BackendProtocol",
    "SandboxBackendProtocol",
    "BackendRuntime",

    # Result types
    "WriteResult",
    "EditResult",
    "ExecuteResponse",
    "FileUploadResponse",
    "FileDownloadResponse",

    # Data types
    "FileInfo",
    "GrepMatch",
    "FileOperationError",

    # Type aliases
    "BackendFactory",
    "BACKEND_TYPES",

    # State backend
    "StateBackend",
    "create_state_backend",

    # Filesystem backend
    "FilesystemBackend",
    "create_filesystem_backend",

    # Store backend
    "StoreBackend",
    "BaseStore",
    "InMemoryStore",
    "SQLiteStore",
    "create_store_backend",

    # Composite backend
    "CompositeBackend",
    "create_composite_backend",

    # Base sandbox (abstract class for remote sandboxes)
    "BaseSandbox",

    # Utilities
    "format_content_with_line_numbers",
    "create_file_data",
    "update_file_data",
    "file_data_to_string",
    "perform_string_replacement",
    "validate_path",
    "glob_match",
    "grep_matches_from_files",
    "glob_search_files",
]
