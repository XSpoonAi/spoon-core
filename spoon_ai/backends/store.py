"""StoreBackend: Persistent key-value store backend (cross-thread).

Uses a simple key-value store interface for persistent, cross-conversation storage.
Files persist across all threads and sessions.
"""

import abc
from typing import Any, Optional

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
    glob_search_files,
    create_file_data,
    file_data_to_string,
    format_read_response,
    grep_matches_from_files,
    perform_string_replacement,
    update_file_data,
)


# ============================================================================
# Store Protocol
# ============================================================================

class BaseStore(abc.ABC):
    """Abstract base class for persistent stores.

    Implementations can use SQLite, Redis, S3, or any other storage backend.
    """

    @abc.abstractmethod
    def get(self, namespace: tuple[str, ...], key: str) -> Optional[dict[str, Any]]:
        """Get a value by key.

        Args:
            namespace: Hierarchical namespace tuple.
            key: The key to retrieve.

        Returns:
            The stored value dict, or None if not found.
        """
        ...

    @abc.abstractmethod
    def put(self, namespace: tuple[str, ...], key: str, value: dict[str, Any]) -> None:
        """Store a value by key.

        Args:
            namespace: Hierarchical namespace tuple.
            key: The key to store under.
            value: The value dict to store.
        """
        ...

    @abc.abstractmethod
    def delete(self, namespace: tuple[str, ...], key: str) -> None:
        """Delete a value by key.

        Args:
            namespace: Hierarchical namespace tuple.
            key: The key to delete.
        """
        ...

    @abc.abstractmethod
    def search(
        self,
        namespace: tuple[str, ...],
        query: Optional[str] = None,
        filter: Optional[dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Search for values in a namespace.

        Args:
            namespace: Hierarchical namespace tuple.
            query: Optional search query.
            filter: Optional key-value filter.
            limit: Maximum results to return.
            offset: Number of results to skip.

        Returns:
            List of matching items with 'key' and 'value' fields.
        """
        ...


# ============================================================================
# In-Memory Store Implementation
# ============================================================================

class InMemoryStore(BaseStore):
    """Simple in-memory store implementation.

    Useful for testing and development. Data is lost when process exits.
    """

    def __init__(self):
        self._data: dict[str, dict[str, dict[str, Any]]] = {}

    def _namespace_key(self, namespace: tuple[str, ...]) -> str:
        return "/".join(namespace)

    def get(self, namespace: tuple[str, ...], key: str) -> Optional[dict[str, Any]]:
        ns_key = self._namespace_key(namespace)
        ns_data = self._data.get(ns_key, {})
        return ns_data.get(key)

    def put(self, namespace: tuple[str, ...], key: str, value: dict[str, Any]) -> None:
        ns_key = self._namespace_key(namespace)
        if ns_key not in self._data:
            self._data[ns_key] = {}
        self._data[ns_key][key] = value

    def delete(self, namespace: tuple[str, ...], key: str) -> None:
        ns_key = self._namespace_key(namespace)
        if ns_key in self._data and key in self._data[ns_key]:
            del self._data[ns_key][key]

    def search(
        self,
        namespace: tuple[str, ...],
        query: Optional[str] = None,
        filter: Optional[dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        ns_key = self._namespace_key(namespace)
        ns_data = self._data.get(ns_key, {})

        results = []
        for key, value in ns_data.items():
            # Apply filter if provided
            if filter:
                match = all(
                    value.get(k) == v for k, v in filter.items()
                )
                if not match:
                    continue

            # Apply query if provided (simple substring match)
            if query:
                content = value.get("content", [])
                content_str = "\n".join(content) if isinstance(content, list) else str(content)
                if query.lower() not in content_str.lower():
                    continue

            results.append({"key": key, "value": value})

        # Apply pagination
        return results[offset:offset + limit]


# ============================================================================
# SQLite Store Implementation
# ============================================================================

class SQLiteStore(BaseStore):
    """SQLite-based persistent store.

    Data persists across process restarts.
    """

    def __init__(self, db_path: str = "store.db"):
        import sqlite3
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_table()

    def _create_table(self):
        cursor = self._conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS store (
                namespace TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                PRIMARY KEY (namespace, key)
            )
        """)
        self._conn.commit()

    def _namespace_key(self, namespace: tuple[str, ...]) -> str:
        return "/".join(namespace)

    def get(self, namespace: tuple[str, ...], key: str) -> Optional[dict[str, Any]]:
        import json
        ns_key = self._namespace_key(namespace)
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT value FROM store WHERE namespace = ? AND key = ?",
            (ns_key, key)
        )
        row = cursor.fetchone()
        if row:
            return json.loads(row[0])
        return None

    def put(self, namespace: tuple[str, ...], key: str, value: dict[str, Any]) -> None:
        import json
        ns_key = self._namespace_key(namespace)
        value_json = json.dumps(value)
        cursor = self._conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO store (namespace, key, value) VALUES (?, ?, ?)",
            (ns_key, key, value_json)
        )
        self._conn.commit()

    def delete(self, namespace: tuple[str, ...], key: str) -> None:
        ns_key = self._namespace_key(namespace)
        cursor = self._conn.cursor()
        cursor.execute(
            "DELETE FROM store WHERE namespace = ? AND key = ?",
            (ns_key, key)
        )
        self._conn.commit()

    def search(
        self,
        namespace: tuple[str, ...],
        query: Optional[str] = None,
        filter: Optional[dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        import json
        ns_key = self._namespace_key(namespace)

        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT key, value FROM store WHERE namespace = ?",
            (ns_key,)
        )

        results = []
        for row in cursor.fetchall():
            key, value_json = row
            value = json.loads(value_json)

            # Apply filter
            if filter:
                match = all(value.get(k) == v for k, v in filter.items())
                if not match:
                    continue

            # Apply query
            if query:
                content = value.get("content", [])
                content_str = "\n".join(content) if isinstance(content, list) else str(content)
                if query.lower() not in content_str.lower():
                    continue

            results.append({"key": key, "value": value})

        return results[offset:offset + limit]

    def close(self):
        self._conn.close()


# ============================================================================
# Store Backend
# ============================================================================

class StoreBackend(BackendProtocol):
    """Backend that stores files in a persistent store (cross-thread).

    Uses a key-value store for persistent, cross-conversation storage.
    Files are organized via namespaces and persist across all threads.

    Example:
        ```python
        store = SQLiteStore("agent_files.db")
        backend = StoreBackend(store)

        # Write persists across sessions
        backend.write("/notes.txt", "Important notes")

        # Read from any thread
        content = backend.read("/notes.txt")
        ```
    """

    def __init__(
        self,
        store: BaseStore,
        namespace: Optional[tuple[str, ...]] = None,
        assistant_id: Optional[str] = None,
    ):
        """Initialize StoreBackend.

        Args:
            store: BaseStore implementation.
            namespace: Optional namespace tuple. Defaults to ("filesystem",).
            assistant_id: Optional assistant ID for multi-agent isolation.
        """
        self.store = store
        self._namespace = namespace or ("filesystem",)
        self._assistant_id = assistant_id

    def _get_namespace(self) -> tuple[str, ...]:
        """Get the namespace for store operations."""
        if self._assistant_id:
            return (self._assistant_id,) + self._namespace
        return self._namespace

    def _convert_item_to_file_data(self, item: dict[str, Any]) -> dict[str, Any]:
        """Convert store item to FileData format."""
        value = item.get("value", item)
        return {
            "content": value.get("content", []),
            "created_at": value.get("created_at", ""),
            "modified_at": value.get("modified_at", ""),
        }

    def _search_all(self) -> list[dict[str, Any]]:
        """Search all items in namespace with pagination."""
        namespace = self._get_namespace()
        all_items = []
        offset = 0
        page_size = 100

        while True:
            page = self.store.search(namespace, limit=page_size, offset=offset)
            if not page:
                break
            all_items.extend(page)
            if len(page) < page_size:
                break
            offset += page_size

        return all_items

    def ls_info(self, path: str) -> list[FileInfo]:
        """List files and directories in the specified directory."""
        namespace = self._get_namespace()
        items = self._search_all()

        infos: list[FileInfo] = []
        subdirs: set[str] = set()

        normalized_path = path if path.endswith("/") else path + "/"

        for item in items:
            key = item.get("key", "")
            if not key.startswith(normalized_path):
                continue

            relative = key[len(normalized_path):]

            if "/" in relative:
                subdir_name = relative.split("/")[0]
                subdirs.add(normalized_path + subdir_name + "/")
                continue

            try:
                fd = self._convert_item_to_file_data(item)
                size = len("\n".join(fd.get("content", [])))
                infos.append({
                    "path": key,
                    "is_dir": False,
                    "size": int(size),
                    "modified_at": fd.get("modified_at", ""),
                })
            except (ValueError, KeyError):
                continue

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
        """Read file content with line numbers."""
        namespace = self._get_namespace()
        item = self.store.get(namespace, file_path)

        if item is None:
            return f"Error: File '{file_path}' not found"

        try:
            file_data = {
                "content": item.get("content", []),
                "created_at": item.get("created_at", ""),
                "modified_at": item.get("modified_at", ""),
            }
        except (ValueError, KeyError) as e:
            return f"Error: {e}"

        return format_read_response(file_data, offset, limit)

    def write(
        self,
        file_path: str,
        content: str,
    ) -> WriteResult:
        """Create a new file with content."""
        namespace = self._get_namespace()

        existing = self.store.get(namespace, file_path)
        if existing is not None:
            return WriteResult(
                error=f"Cannot write to {file_path} because it already exists. "
                      f"Read and then make an edit, or write to a new path."
            )

        file_data = create_file_data(content)
        self.store.put(namespace, file_path, file_data)

        return WriteResult(path=file_path, files_update=None)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Edit a file by replacing string occurrences."""
        namespace = self._get_namespace()

        item = self.store.get(namespace, file_path)
        if item is None:
            return EditResult(error=f"Error: File '{file_path}' not found")

        try:
            file_data = {
                "content": item.get("content", []),
                "created_at": item.get("created_at", ""),
                "modified_at": item.get("modified_at", ""),
            }
        except (ValueError, KeyError) as e:
            return EditResult(error=f"Error: {e}")

        content = file_data_to_string(file_data)
        result = perform_string_replacement(content, old_string, new_string, replace_all)

        if isinstance(result, str):
            return EditResult(error=result)

        new_content, occurrences = result
        new_file_data = update_file_data(file_data, new_content)

        self.store.put(namespace, file_path, new_file_data)

        return EditResult(
            path=file_path,
            files_update=None,
            occurrences=int(occurrences)
        )

    def grep_raw(
        self,
        pattern: str,
        path: Optional[str] = None,
        glob: Optional[str] = None,
    ) -> list[GrepMatch] | str:
        """Search for pattern in files."""
        items = self._search_all()
        files: dict[str, Any] = {}

        for item in items:
            try:
                key = item.get("key", "")
                files[key] = self._convert_item_to_file_data(item)
            except (ValueError, KeyError):
                continue

        return grep_matches_from_files(files, pattern, path or "/", glob)

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        """Find files matching glob pattern."""
        items = self._search_all()
        files: dict[str, Any] = {}

        for item in items:
            try:
                key = item.get("key", "")
                files[key] = self._convert_item_to_file_data(item)
            except (ValueError, KeyError):
                continue

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

    def upload_files(
        self, files: list[tuple[str, bytes]]
    ) -> list[FileUploadResponse]:
        """Upload multiple files to the store."""
        namespace = self._get_namespace()
        responses: list[FileUploadResponse] = []

        for path, content in files:
            try:
                content_str = content.decode("utf-8")
                file_data = create_file_data(content_str)
                self.store.put(namespace, path, file_data)
                responses.append(FileUploadResponse(path=path, error=None))
            except Exception:
                responses.append(FileUploadResponse(path=path, error="invalid_path"))

        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download multiple files from the store."""
        namespace = self._get_namespace()
        responses: list[FileDownloadResponse] = []

        for path in paths:
            item = self.store.get(namespace, path)

            if item is None:
                responses.append(
                    FileDownloadResponse(path=path, content=None, error="file_not_found")
                )
                continue

            try:
                file_data = {
                    "content": item.get("content", []),
                    "created_at": item.get("created_at", ""),
                    "modified_at": item.get("modified_at", ""),
                }
                content_str = file_data_to_string(file_data)
                content_bytes = content_str.encode("utf-8")
                responses.append(
                    FileDownloadResponse(path=path, content=content_bytes, error=None)
                )
            except Exception:
                responses.append(
                    FileDownloadResponse(path=path, content=None, error="invalid_path")
                )

        return responses


def create_store_backend(
    store: Optional[BaseStore] = None,
    db_path: str = "store.db",
    use_sqlite: bool = True,
    namespace: Optional[tuple[str, ...]] = None,
    assistant_id: Optional[str] = None,
) -> StoreBackend:
    """Create a StoreBackend.

    Args:
        store: Optional BaseStore instance. If not provided, creates one.
        db_path: Path to SQLite database (if using SQLite).
        use_sqlite: If True, use SQLite. Otherwise, use in-memory store.
        namespace: Optional namespace tuple.
        assistant_id: Optional assistant ID for isolation.

    Returns:
        StoreBackend instance.

    Example:
        ```python
        # Use SQLite for persistence
        backend = create_store_backend(db_path="agent.db")

        # Use in-memory store for testing
        backend = create_store_backend(use_sqlite=False)

        # With assistant isolation
        backend = create_store_backend(assistant_id="agent-001")
        ```
    """
    if store is None:
        if use_sqlite:
            store = SQLiteStore(db_path)
        else:
            store = InMemoryStore()

    return StoreBackend(
        store=store,
        namespace=namespace,
        assistant_id=assistant_id
    )
