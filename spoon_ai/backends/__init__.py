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
